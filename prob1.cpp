/*#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include "liblinear/linear.h"

using namespace std;

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

int readData(const char * fileName, \
        vector< vector<lable_node> > &lables, \
        vector< vector<feature_node> >&features);
void wrapData(const vector< vector<lable_node> > &lables, \
        const vector< vector<feature_node> >&features,\
        struct feature_node **featureMat,\
        double *targetVal);

int main(){
    vector< vector<lable_node> >    lables;
    vector< vector<feature_node> >  features;
    struct feature_node *           featureMat[MAX_TRAINDATA];
    double                          targetVal[MAX_TRAINDATA] ;

    vector< vector<lable_node> >    test_lables;
    vector< vector<feature_node> >  test_features;
    struct feature_node *           test_featureMat[MAX_TESTDATA];
    double                          test_targetVal[MAX_TESTDATA] ;
    double                          test_predictRes[MAX_TESTDATA];

    int N = readData("data/train.txt",lables,features);
    wrapData(lables,features,featureMat,targetVal);

    int tN = readData("data/test.txt",test_lables,test_features);
    wrapData(test_lables,test_features,test_featureMat,test_targetVal);

    // TRAIN procss
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

    const char * errcode = check_parameter(&prob,&param);
    if(errcode){
        printf("%s\n",errcode);
    }

    printf("start trainning\n");
    model * mod = train(&prob,&param);

    // PREDICT procss
    for(int i = 0; i < tN; i++){
        test_predictRes[i] = predict(mod,test_featureMat[i]);
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
            targetVal[i] = 0;
        else
            targetVal[i] = 1;
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

*/













