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
        printf("%-7d %.15lf\n",f.index, f.value)

/* data structure defination */
struct lable_node{
    char Section;
    int  Class;
    char SubClass;
    int  group1;
    int  group2;
};

/* const defination */
const int NUM_FEATURE = 5001;
const int NUM_POSITIVE = 8;
const int NUM_NEGATIVE = 24;
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
        vector< std::vector<feature_node* > > & retFeature , \
        std::vector<std::vector<double> > &retTargetval);

/* functions */
int main(){
    std::vector<std::vector<double> >     gTargetval;
    
    int N = readData("data/train.txt",lables,features);
    getTargetVal(lables,targetVal);
    vector< std::vector<feature_node* > > gFeature;
    classify(lables,features,gFeature,gTargetval);
    printf("Finished classi\n");

    threads poll(8);
    poll.stop();
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
        vector< std::vector<feature_node* > > & retFeature , \
        std::vector<std::vector<double> > &retTargetval)
{
    int counterP = 0;
    int counterN = 0;
    int numGroup = NUM_POSITIVE*NUM_NEGATIVE;
    // count possitives
    retFeature.clear();
    retTargetval.clear();
    for(int i = 0; i < numGroup; i++){
        retTargetval.push_back(std::vector<double> ());
        retFeature.push_back(std::vector<feature_node*>());
    }
    for (unsigned int i = 0; i < features.size(); ++i){
        if(lables[i][0].Section == 'A'){


            retFeature[counterP].push_back(features[i].data());
            retTargetval[counterP].push_back(1);
            counterP++;
            counterP = counterP % numGroup;
        }else{
            retFeature[counterN].push_back(features[i].data());
            retTargetval[counterN].push_back(-1);
            counterN++;
            counterN = counterN % numGroup;
        }
    }
    return 0;
}
