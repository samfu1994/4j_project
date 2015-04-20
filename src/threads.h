#ifndef THREADS_CLASS_HEAD
#define THREADS_CLASS_HEAD


#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <vector>
#include <list>
#include <semaphore.h>

struct threadP;

class threads {
private:
    std::list<void*(*)(void*)> tasks;
    std::list<void *>       taskParam;
    int                     max_thhreadNum;
    sem_t                   queueMutex;
    sem_t                   semTasks;
    sem_t                   alljobs;
    pthread_t               *tids;
    int                     jobCount;
    threadP                 *threadParam;

    static void * threadFunc(void *);
public:
    threads(void);
    threads(int _max_thhreadNum);
    void addJob( void*(*)(void *) , void * );
    void wait(void);
};

struct threadP{
    int threadNum;
    threads * th;
};




#endif // THREADS_CLASS_HEAD
