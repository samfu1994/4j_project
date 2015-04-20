#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include "liblinear/linear.h"
#include "src/threads.h"

struct threpara{
    int param;
    int retval;
};

void * func(void *p){
    threpara * pp = (threpara*) p;
    pp->retval = pp->param * pp->param;
    printf("%d\n",pp->retval);
    return NULL;
}

int main(){
    printf("asf\n");
    threads poll(8);
    threpara tasks[100];
    for(int i = 0; i < 100; i++){
        tasks[i].param = i;
        poll.addJob(func,&tasks[i]);
    }
    poll.wait();
    return 0;
}
