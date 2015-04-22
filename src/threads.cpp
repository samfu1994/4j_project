#include "threads.h"

threads::threads(){
    threads(8);
}

threads::threads(int _max){
    max_thhreadNum =  _max;
    jobCount = 0;
    unsigned int err = 0;
    numMaxJobs = _max * 2;

    err |= sem_init(&queueMutex,0,1);
    err |= sem_init(&semTasks,0,0);
    err |= sem_init(&alljobs,0,0);
    err |= sem_init(&maxJobs,0, numMaxJobs);

    stopThread = false;
    tids        = new pthread_t[max_thhreadNum];
    threadParam = new threadP[max_thhreadNum];

    for(int i = 0; i < max_thhreadNum; i++){
        threadParam[i].threadNum = i;
        threadParam[i].th = this;
        err |= pthread_create(&tids[i],NULL,threadFunc,&threadParam[i]);
    }
    if(err){
        printf("ERROR. ERR when create thread poll\n");
    }else{
        printf("threads created\n");
    }
}


void* threads::threadFunc(void *p){
    threads     * th = ((threadP*)p)->th;
    void        * param;
    void*(*func)(void *);
    while(1){
        sem_wait(&th->semTasks);
        if(th->stopThread){
            sem_post(&th->semTasks);
            break;
        }
        sem_wait(&th->queueMutex);
            param = th->taskParam.back();
            func  = th->tasks.back();
            th->taskParam.pop_back();
            th->tasks.pop_back();
        sem_post(&th->queueMutex);
        func(param);
        sem_post(&th->alljobs);
        sem_post(&th->maxJobs);
    }
    return NULL;
}


void threads::addJob(void*(*func)(void *) ,void * param){
    if(stopThread)
        return;
    sem_wait(&maxJobs); // block when there is too many jobs in queue
    sem_wait(&queueMutex);
        tasks.push_front(func);
        taskParam.push_front(param);
    sem_post(&queueMutex);
    sem_post(&semTasks);
    jobCount++;
}


void threads::wait(){
    for(;jobCount > 0; jobCount--){
        sem_wait(&alljobs);
    }
}


void threads::stop(void){
    wait();
    stopThread = true;
    sem_post(&semTasks);
    for(int i = 0; i < max_thhreadNum; i++){
        pthread_join(tids[i],NULL);
    }
}















