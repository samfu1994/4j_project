#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <pthread.h>
#include <semaphore.h>
#include <queue>
#include <string.h>
#include <math.h>
#include "liblinear/linear.h"

using namespace std;
#define NUM_POSITIVE 8
#define NUM_NEGATIVE 24
#define CONTAINER_SIZE 3600
#define PREDICT_THREAD_NUM 8
#define SCAN_LABLE(fin,l) \
        (fscanf(fin, \
            "%[A-Z] %d %[A-Z]/%d/%d,",\
            &((l).Section),\
            &((l).Class),\
            &((l).SubClass),\
            &((l).group1),\
            &((l).group2)))
#define PRINT_LABLE(l) \
        printf("%c %d %c %d %d\n",\
            (l).Section,\
            (l).Class,\
            (l).SubClass,\
            (l).group1,\
            (l).group2)

#define SCAN_FEATURE(fin,f)\
        (fscanf(fin,  \
            "%d:%lf", \
            &((f).index), \
            &((f).value)))

#define PRINT_FEATURE(f)\
        printf("%-7d %.15lf\n",f.index, f.value)

struct lable_node{
    char Section;
    int  Class;
    char SubClass;
    int  group1;
    int  group2;
};

const int NUM_FEATURE = 5001;
const int MAX_TRAINDATA = 120000;
const int MAX_TESTDATA =40000;
const double BIAS = 1;
const feature_node endOfFeature = {-1,0};
const feature_node biasFeature = {NUM_FEATURE,BIAS};
int num_sub_positive[NUM_POSITIVE];
int num_sub_negative[NUM_NEGATIVE];
double positive_targetVal[NUM_POSITIVE][CONTAINER_SIZE];
double negative_targetVal[NUM_NEGATIVE][CONTAINER_SIZE];
model * mod[NUM_POSITIVE][NUM_NEGATIVE];
sem_t mutex;
sem_t count_predict;
int min_array[NUM_POSITIVE];
bool finish;
int tN;
int sum;

void down(sem_t &x){
    sem_wait(&x);
}
void up(sem_t &x){
    sem_post(&x);
}
int readData(const char * fileName, \
        vector< vector<lable_node> > &lables, \
        vector< vector<feature_node> >&features);
void wrapData(const vector< vector<lable_node> > &lables, \
        const vector< vector<feature_node> >&features,\
        struct feature_node **featureMat,\
        double *targetVal);
model* train(const problem *prob, const parameter *param);
    static vector< vector<lable_node> >    lables;
    vector< vector<feature_node> >  features;
    struct feature_node *           featureMat[MAX_TRAINDATA];
    double                          targetVal[MAX_TRAINDATA] ;

    vector< vector<lable_node> >    test_lables;
    vector< vector<feature_node> >  test_features;
    struct feature_node *           test_featureMat[MAX_TESTDATA];
    double                          test_targetVal[MAX_TESTDATA] ;
    double                          test_predictRes[MAX_TESTDATA];
struct s{
    struct problem * prob;
    struct parameter * param;
    int i;
    int j;
};
class func{
public:
    static void * train_subproblem_helper(void * args){
        return ((func *)args) -> train_subproblem(args);
    }
    void * train_subproblem(void * args){
        struct s * ar = (struct s *)args;
        problem * prob =  ar -> prob;
        parameter * param = ar -> param;
        printf("enter train sub, i = %d, j = %d\n", ar->i, ar->j);
        mod[ar->i][ar->j] = train(prob,param);
        printf("quit train sub\n");
        pthread_exit(NULL);
        return NULL;
    };
    static void * predict_helper(void * args){
        return ((func *)args) -> predict_problem(args);
    }
    void * predict_problem(void * args){
        int *index = (int *)args;
        //printf("enter predict_problem index = %d\n",*index);
        int lock;
        down(mutex);
        sem_getvalue(&count_predict, &lock);
        up(count_predict);
        up(mutex);
        double tmp_result = 1;
        for(int j = 0; j < NUM_NEGATIVE; j++){
            tmp_result = min(tmp_result, predict(mod[lock][j],test_featureMat[*index]));
        }
        min_array[lock] = tmp_result;
        pthread_exit(NULL);
        return NULL;
    }
};
feature_node ** combine(feature_node ** &ptr_f, struct feature_node * positive_featureMat[NUM_POSITIVE][CONTAINER_SIZE],int i,struct feature_node * negative_featureMat[NUM_NEGATIVE][CONTAINER_SIZE],int j){
    int p = num_sub_positive[i] - 1, q = num_sub_negative[j] - 1;
    ptr_f = new feature_node* [p + q];
    sum = p + q;
    size_t offset = sizeof(feature_node*) * p;
    feature_node ** tmp = positive_featureMat[i];
    memcpy(ptr_f, tmp ,offset);
    memcpy(ptr_f+offset, negative_featureMat[j],sizeof(feature_node *) * q);
    return ptr_f;
}
double * getY(double * &f, double positive_targetVal[NUM_POSITIVE][CONTAINER_SIZE],int i,double negative_targetVal[NUM_NEGATIVE][CONTAINER_SIZE],int j){
    int p = num_sub_positive[i] - 1, q = num_sub_negative[j] - 1;
    f  = new double [p + q];
    size_t offset = sizeof(double) * p;
    memcpy(f, positive_targetVal,offset);
    memcpy(f+offset, negative_targetVal,sizeof(double) * q);
    return f;
}
double predict_subproblem(int index){
    int tmp;
    sem_getvalue(&count_predict, &tmp);
    for(int i = 0; i < tmp; i++)
        sem_trywait(&count_predict);
    for(int i = 0; i < NUM_POSITIVE;i++)
        min_array[i] = 1;
    pthread_t predict_thread[PREDICT_THREAD_NUM];
    for(int i = 0; i < PREDICT_THREAD_NUM; i++){
        pthread_create(&predict_thread[i], NULL, func::predict_helper, &index);
    }
    for(int i = 0; i < PREDICT_THREAD_NUM; i++){
        pthread_join(predict_thread[i],NULL);
    }
    int max_result = 0;
    for(int i = 0; i < NUM_POSITIVE; i++){
        max_result = max(max_result, min_array[i]);
    }
    test_predictRes[index] = max_result;
    return 0;
}
void mytrain(struct feature_node **featureMat, double *targetVal, int N){
    struct problem prob;
    prob.l = N;
    prob.n = NUM_FEATURE;
    prob.y = targetVal;
    prob.x = featureMat;
    prob.bias = 1;

    struct parameter param;
    param.solver_type = L2R_L2LOSS_SVC_DUAL;
    param.C = 1;
    param.eps = 0.1;
    param.p = 0.1;
    param.nr_weight = 0;
    param.weight = NULL;
    param.weight_label = NULL;
    int num_lines_train = lables.size();
    int num_train_negative = 0, num_train_positive = 0;
    feature_node * positive_featureMat[NUM_POSITIVE][CONTAINER_SIZE] = {0};
    feature_node * negative_featureMat[NUM_NEGATIVE][CONTAINER_SIZE] = {0};
    for(int i = 0; i < NUM_POSITIVE; i++){
        for(int j = 0; j < CONTAINER_SIZE; j++){
            positive_targetVal[i][j] = 1;
        }
    }
    for(int i = 0; i < NUM_NEGATIVE; i++){
        for(int j = 0; j < CONTAINER_SIZE; j++){
            negative_targetVal[i][j] = 0;
        }
    }
    int thread_count = 0;
    int pos_data[NUM_POSITIVE][CONTAINER_SIZE];//27876 / 8 = 3485
    int neg_data[NUM_NEGATIVE][CONTAINER_SIZE];
    for(int i = 0; i < num_lines_train; i++){
        if(targetVal[i] == 1){
            positive_featureMat[num_train_positive % NUM_POSITIVE][num_sub_positive[num_train_positive % NUM_POSITIVE]] = featureMat[i];
            num_sub_positive[num_train_positive % NUM_POSITIVE]++;
            num_train_positive++;
        }
        else if(targetVal[i] == 0){
            negative_featureMat[num_train_negative % NUM_NEGATIVE][num_sub_negative[num_train_negative % NUM_NEGATIVE]] = featureMat[i];
            num_sub_negative[num_train_negative % NUM_NEGATIVE]++;
            num_train_negative++;
        }
        else{
            printf("error in partition!\n");
        }
    }
    const char * errcode = check_parameter(&prob,&param);
    if(errcode){
        printf("%s\n",errcode);
    }
    printf("start trainning\n");
    //mytrain(param, prob,positive_featureMat,negative_featureMat);
    pthread_t thread[NUM_POSITIVE][NUM_NEGATIVE];
    struct s s1[NUM_NEGATIVE];
    struct problem sub_prob[NUM_NEGATIVE];
    for(int i = 0; i < NUM_POSITIVE; i++){
        for(int j = 0; j < NUM_NEGATIVE; j++){
            feature_node ** ptr_f;
            double * f;
            s1[j].param = &param;
            sub_prob[j].x = combine(ptr_f, positive_featureMat,i,negative_featureMat,j);
            delete [] ptr_f;
            prob.y = getY(f,positive_targetVal,i,negative_targetVal,j);
            delete [] f;
            s1[j].prob = &sub_prob[j];
            s1[j].i = i;
            s1[j].j = j;
            pthread_create(&thread[i][j],NULL,func::train_subproblem_helper,&s1[j]);
        }
        printf("waiting for threads\n");
        for(int j = 0; j < NUM_NEGATIVE; j++){
            pthread_join(thread[i][j],NULL);
        }
        printf("for i = %d, threads are finished\n", i);
    }
    printf("training is over\n");
    return;
}
int main(){
    sem_init(&mutex, 0, 1);
    for(int i = 0 ; i < NUM_POSITIVE; i++)
        num_sub_positive[i] = 0;
    for(int i = 0;i < NUM_NEGATIVE; i++)
        num_sub_negative[i] = 0;
    int N = readData("data/train.txt",lables,features);
    wrapData(lables,features,featureMat,targetVal);

    tN = readData("data/test.txt",test_lables,test_features);
    wrapData(test_lables,test_features,test_featureMat,test_targetVal);

    // TRAIN procss
    mytrain(featureMat,targetVal, N);
    // PREDICT procss

    for(int i = 0; i < tN; i++){
        predict_subproblem(i);
    }
    int precise = 0;
    for(int i = 0; i < tN; i++){
        if(test_predictRes[i] == test_targetVal[i])
            precise++;
    }
    printf("num %d / %d\n",precise,tN );
    return 0;
}

void wrapData(const vector< vector<lable_node> > &lables, \
        const vector< vector<feature_node> >&features, \
        struct feature_node **featureMat, \
        double *targetVal)
{
    int N = lables.size();
    for(int i = 0; i < N; i++){
        featureMat[i] = (feature_node*)features[i].data();
        if(lables[i][0].Section == 'A')
            targetVal[i] = 1;
        else
            targetVal[i] = 0;
    }
}


int readData(const char * fileName, \
        vector< vector<lable_node> > &lables, \
        vector< vector<feature_node> >&features)
{
    FILE * fin = fopen(fileName,"r");
    // read Data From File
    int N = 0;
    feature_node tfeature;
    lable_node tlable;
    while(feof(fin) == 0){
        lables.push_back(vector<lable_node>());
        features.push_back(vector<feature_node>());
        while(SCAN_LABLE(fin,tlable)) lables[N].push_back(tlable);
        while(SCAN_FEATURE(fin,tfeature) && feof(fin) == 0) features[N].push_back(tfeature);
        if(BIAS >= 0) features[N].push_back(biasFeature);
        features[N].push_back(endOfFeature);
        N++;
    }
    printf("finished read file. %d records\n",N);
    fclose(fin);
    return N;
}













