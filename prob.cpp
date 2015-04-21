#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include "liblinear/linear.h"
#include "src/threads.h"
#include <string.h>

using namespace std;
/* micro defination */
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
        printf("%-7d %.15lf\n",(f).index, (f).value)

/* data structure defination */
struct lable_node{
    char Section;
    int  Class;
    char SubClass;
    int  group1;
    int  group2;
};
struct trainThreadParams{
    int         groupNum;
    problem*    prob;
    parameter*  param;
    model**     retData;
    trainThreadParams(int _gn, \
        problem* _prob, \
        parameter* _param, \
        model** _retData )
    {
        groupNum = _gn;
        prob = _prob;
        param = _param;
        retData = _retData;
    }
};

/* const defination */
const int NUM_FEATURE = 5001;
const int NUM_POSITIVE = 2;
const int NUM_NEGATIVE = 4;
const int NUM_GROUP    = NUM_NEGATIVE*NUM_POSITIVE;
const double BIAS = 1;
const feature_node endOfFeature = {-1,0};
const feature_node biasFeature = {NUM_FEATURE,BIAS};

/* global variables */
vector<vector<feature_node> >     features;
vector<vector<lable_node> >       lables;
vector<double>                         targetVal;

/* function prototype */

int readData(const char * fileName, \
        vector< vector<lable_node> > &lables, \
        vector< vector<feature_node> >&features);
int getTargetVal(vector<vector<lable_node> > & labs, \
        vector<double>& retVal);
int classify(vector< vector<lable_node> > &lables, \
        vector< vector<feature_node> >&features, \
        vector< vector<feature_node* > > & retFeature , \
        vector<vector<double> > &retTargetval);
int getGroupParam(vector< vector<feature_node* > > &gFeature , \
        vector<vector<double> > &gTargetval , \
        vector<parameter> &retParam, \
        vector<problem>  &retProb \
);
void * trainThreadFunc(void *);

/* functions */
int main(){
    vector<vector<double> >             gTargetval;
    vector< vector<feature_node* > >    gFeature;
    vector<parameter>                   gParam;
    vector<problem>                     gProb;
    vector<model*>                      gmodel;
    threads poll(8);

    // read data
    int N = readData("data/train.txt",lables,features);
    getTargetVal(lables,targetVal);
    classify(lables,features,gFeature,gTargetval);
    printf("Finished classify\n");

    // predict
    getGroupParam(gFeature,gTargetval,gParam,gProb);
    gmodel.reserve(NUM_GROUP);
    for(int i = 0; i < NUM_GROUP; i++){
        poll.addJob(trainThreadFunc,new trainThreadParams(i,&gProb[i],&gParam[i],&gmodel[i]));
    }

    poll.stop();
    return 0;
}
void * trainThreadFunc(void * p){
    trainThreadParams * pa = (trainThreadParams*)p;
    *(pa->retData) = train(pa->prob,pa->param);
    return NULL;
}
int getGroupParam(vector< vector<feature_node* > > &gFeature , \
        vector<vector<double> > &gTargetval , \
        vector<parameter> &retParam, \
        vector<problem>  &retProb \
)
{
    retParam.clear();
    retProb.clear();
    retParam.reserve(NUM_GROUP);
    retProb.reserve(NUM_GROUP);
    for(int i = 0; i < NUM_GROUP; i++){
        retProb[i].l = gFeature[i].size();
        retProb[i].n = NUM_FEATURE;
        retProb[i].y = gTargetval[i].data();
        retProb[i].x = gFeature[i].data();
        retProb[i].bias = 1;
        retParam[i].solver_type = L2R_L2LOSS_SVC_DUAL;
        retParam[i].C = 1;
        retParam[i].eps = 0.1;
        retParam[i].p = 0.1;
        retParam[i].nr_weight = 0;
        retParam[i].weight = NULL;
        retParam[i].weight_label = NULL;
        const char * c = check_parameter(&retProb[i],&retParam[i]);
        if(c){
            printf("%s\n", c);
            return -1;
        }
    }
    return 0;
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
        while(SCAN_LABLE(fin,tlable))
            lables[N].push_back(tlable);
        while(SCAN_FEATURE(fin,tfeature) && feof(fin) == 0)
            features[N].push_back(tfeature);
        if(BIAS >= 0) features[N].push_back(biasFeature);
        features[N].push_back(endOfFeature);
        N++;
    }
    printf("finished read file. %d records\n",N);
    fclose(fin);
    return N;
}

int getTargetVal(vector<vector<lable_node> > & labs, \
        vector<double>& retVal)
{
    retVal.clear();
    for(int i = 0; i < (int)labs.size(); i++){
        if(labs[i][0].Section == 'A'){
            retVal.push_back(1);
        }else{
            retVal.push_back(-1);
        }
    }
    return 0;
}

/*
    Use a code techque to code and decode classfications;
    For possitive class i and negtive class j as a group,
    The group number is k = j + i * NUM_POSITIVE;
    That is a posstive origented code. for convenienct of
    latter MAX and MIN operations.
*/
int classify(vector< vector<lable_node> > &lables, \
        vector< vector<feature_node> >&features, \
        vector< vector<feature_node* > > & retFeature , \
        vector<vector<double> > &retTargetval)
{
    int counterP = 0;
    int counterN = 0;
    retFeature.clear();
    retTargetval.clear();
    // create data streucture
    for(int i = 0; i < NUM_GROUP; i++){
        retTargetval.push_back(vector<double> ());
        retFeature.push_back(vector<feature_node*>());
    }
    // classify data
    for (unsigned int i = 0; i < features.size(); ++i){
        if(lables[i][0].Section == 'A'){
            for(int j = counterP; j < NUM_GROUP; j+=NUM_POSITIVE){
                retFeature[j].push_back(features[i].data());
                retTargetval[j].push_back(1);
            }
            counterP++;
            counterP = counterP % NUM_POSITIVE;
        }else{
            for(int j = counterN*NUM_POSITIVE; j < counterN*NUM_POSITIVE+NUM_POSITIVE; j++){
                retFeature[j].push_back(features[i].data());
                retTargetval[j].push_back(-1);
            }
            counterN++;
            counterN = counterN % NUM_NEGATIVE;
        }
    }
    return 0;
}
