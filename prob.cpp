#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include "liblinear/linear.h"
#include "src/threads.h"
#include <string.h>
#include <map>
#include <algorithm>
#include <time.h>
 #include <sys/time.h>

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
struct preadictThreadParams{
    vector<model*> * mod;
    feature_node* f;
    double * retVal;
    double bias;
    preadictThreadParams(vector<model*> *_mod, \
            feature_node* _f, \
            double*ret, \
            double _bias = 0){
        mod = _mod;
        f = _f;
        retVal = ret;
        bias = _bias;
    }
};
struct predictResult
{
    double TP;
    double FP;
    double TN;
    double FN;
    double TPR;
    double FPR;
    double F1;
    double p;
    double r;
    vector<double> roc_tpr;
    vector<double> roc_fpr;
    predictResult(){
        TP = 0;
        FP = 0;
        TN = 0;
        FN = 0;
        roc_fpr.clear();
        roc_tpr.clear();
    }
    void calculate(){
        TPR = TP / (TP + FN);
        FPR = FP / (FP + TN);
        p   = TP / (TP + FP);
        r   = TP / (TP + FN);
        F1  = 2 * r * p / (r + p);
    }
};

/* const defination */
const int NUM_FEATURE = 5001;
      int NUM_POSITIVE =20;
      int NUM_NEGATIVE = 60;
      int NUM_GROUP    = NUM_NEGATIVE*NUM_POSITIVE;
const double BIAS = 1;
const feature_node endOfFeature = {-1,0};
const feature_node biasFeature = {NUM_FEATURE,BIAS};

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
int classify_1(vector< vector<lable_node> > &lables, \
        vector< vector<feature_node> >&features, \
        vector< vector<feature_node* > > & retFeature , \
        vector<vector<double> > &retTargetval);
int classify_2(vector< vector<lable_node> > &lables, \
        vector< vector<feature_node> >&features, \
        vector< vector<feature_node* > > & retFeature , \
        vector<vector<double> > &retTargetval);
int classify_3(vector< vector<lable_node> > &lables, \
        vector< vector<feature_node> >&features, \
        vector< vector<feature_node* > > & retFeature , \
        vector<vector<double> > &retTargetval);
int getGroupParam(vector< vector<feature_node* > > &gFeature , \
        vector<vector<double> > &gTargetval , \
        vector<parameter> &retParam, \
        vector<problem>  &retProb );
void * trainThreadFunc(void *);
void * predictThreadFunc(void *);
predictResult getPredictRes(vector<double> &targetVal, \
        vector<double> &predictTargetVal);
void getRoc(threads &poll, \
        vector<vector<feature_node> > &tFeatures, \
        vector<double> &tTargetval,\
        vector<model*> &gmodel, \
        void*(*func)(void*), \
        vector<double> &bias ,\
        predictResult & retVal );
vector<double> autuRoc(threads &poll, \
        vector<vector<feature_node> > &tFeatures, \
        vector<double> &tTargetval,\
        vector<model*> &gmodel, \
        void*(*func)(void*) );
void new_getRoc(threads &poll, \
        vector<vector<feature_node> > &tFeatures, \
        vector<double> &tTargetval,\
        vector<model*> &gmodel, \
        predictResult & retVal );

/* functions */
int main(){
    // rew train data
    vector<vector<feature_node> >       features;
    vector<vector<lable_node> >         lables;
    vector<double>                      targetVal;
    // raw test data
    vector<double>                      tTargetval;
    vector<vector<feature_node> >       tFeatures;
    vector<vector<lable_node> >         tLables;
    vector<double>                      predictTargetVal;
    // train var
    vector<vector<double> >             gTargetval;
    vector< vector<feature_node* > >    gFeature;
    vector<parameter>                   gParam;
    vector<problem>                     gProb;
    vector<model*>                      gmodel;
    // test var
    threads                             poll(8);
    // time measure
    clock_t start;
    clock_t end;
    timeval starttime,endtime;
    // read data
    readData("data/train.txt",lables,features);
    readData("data/test.txt",tLables,tFeatures);

    start = clock();
    gettimeofday(&starttime,0);

    getTargetVal(lables,targetVal);
    getTargetVal(tLables,tTargetval);
    // classify trainning data
    classify(lables,features,gFeature,gTargetval);
    printf("Finished classify\n");
    // train
    getGroupParam(gFeature,gTargetval,gParam,gProb);
    gmodel.reserve(NUM_GROUP);
    for(int i = 0; i < NUM_GROUP; i++){
        poll.addJob(trainThreadFunc,\
            new trainThreadParams(i,&gProb[i],&gParam[i],&gmodel[i]));
    }
    poll.wait();
    printf("predict\n");
    // predict
    // predictTargetVal.reserve(tTargetval.size());
    // for(unsigned int i = 0; i < tFeatures.size(); i++){
    //     poll.addJob(predictThreadFunc,\
    //         new preadictThreadParams(&gmodel,tFeatures[i].data(),&(predictTargetVal[i]) ));
    // }
    // poll.wait();
    // predictResult pr = getPredictRes(tTargetval,predictTargetVal);
    predictResult pr;

/*    vector<double> bias = autuRoc(poll,tFeatures,tTargetval,gmodel,predictThreadFunc);
    // get roc
    printf("GETROC\n");
    getRoc(poll,tFeatures,tTargetval,gmodel,predictThreadFunc,bias,pr);
    for(int i = 0; i < (int)pr.roc_tpr.size(); i++){
        printf("%f %f \n", pr.roc_tpr[i], pr.roc_fpr[i]);
    }
 */   
    new_getRoc(poll,tFeatures,tTargetval,gmodel,pr);
    end = clock();
    gettimeofday(&endtime,0);
    double timeuse = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
    timeuse /=1000;
    
    printf("FINAL RESULT %f %f %f\n",pr.r, pr.p, pr.F1);
    printf("%f\n", (pr.TP+pr.TN)/(pr.TP+pr.TN+pr.FP+pr.FN));
    printf("Time1: %f\n", ((double)(end- start)) / ((double)CLOCKS_PER_SEC) );
    printf("Time2: %f\n", timeuse );
    poll.stop();
    return 0;
}
void * predictThreadFunc_raw(void *p){
    preadictThreadParams *  pp = (preadictThreadParams*)p;
    vector<model*>&         mod = *(pp->mod);
    vector<double>          mins;
    mins.reserve(NUM_POSITIVE);
    // MIN
    double t;
    for(int i = 0; i < NUM_POSITIVE; i++){
        t = predict_raw(mod[i],pp->f);
        for(int j = i+NUM_POSITIVE; j < NUM_GROUP;j+=NUM_POSITIVE){
            t = min(t,predict_raw(mod[j],pp->f));
        }
        mins[i] = t;
    }
    // MAX
    t = mins[0];
    for(int i = 1;i < NUM_POSITIVE;i++){
        t = max(t,mins[i]);
    }
    *(pp->retVal) = t;
    delete pp;
    return NULL;
}
void new_getRoc(threads &poll, \
        vector<vector<feature_node> > &tFeatures, \
        vector<double> &tTargetval,\
        vector<model*> &gmodel, \
        predictResult & retVal )
{
    int numPoints = 500;
    int pstep,nstep;
    vector<double> predictTargetVal;
    predictTargetVal.reserve(tTargetval.size());
    std::vector<double> pp;
    std::vector<double> pn;
    for(unsigned int i = 0; i < tTargetval.size(); i++){
        predictTargetVal.push_back(0);
        poll.addJob(predictThreadFunc_raw,\
            new preadictThreadParams(&gmodel,tFeatures[i].data(),&(predictTargetVal[i]) ));
    }
    poll.wait();
    for(int i = 0; i < (int)predictTargetVal.size(); i++){
        if(tTargetval[i] == 1){
            pp.push_back(predictTargetVal[i]);
        }else{
            pn.push_back(predictTargetVal[i]);
        }
    }
    // printf("Poss %d NEG %d\n", pp.size(),pn.size());
    sort(pp.begin(),pp.end());
    sort(pn.begin(),pn.end());
    // for(int i = 0; i < 100; i++){
    //     printf("%f %f \n", pp[i],pn[i]);
    // }
    pstep = pp.size()/numPoints;
    nstep = pn.size()/numPoints;
    // printf("steps %d %d \n", pstep,nstep);
    // return ;
    retVal.roc_tpr.push_back(1);
    retVal.roc_fpr.push_back(1);
    int n = 0, p = 0;
    while(1){
        p += pstep;
        n += nstep;
        if(p >= (int)pp.size() || n >= (int)pn.size()){
            break;
        }
        p = min(p,(int)pp.size()-1);
        n = min(n,(int)pn.size()-1);
        // printf("size %d %d\n", p,n);
        if(pp[p]>pn[n]){
            while(pp[p] > pn[n] && p > 0) p--;
        }else{
            while(pp[p] < pn[n] && n > 0) n--;
        }
        retVal.roc_tpr.push_back((double)(pp.size()-p)/(pp.size() ));
        retVal.roc_fpr.push_back((double)(pn.size()-n)/(pn.size() ));
    }
    retVal.roc_tpr.push_back(0);
    retVal.roc_fpr.push_back(0);
    for(int i = 0; i <(int)retVal.roc_tpr.size(); i++){
        printf("%f  %f\n", retVal.roc_tpr[i], retVal.roc_fpr[i]);
    }
    p = 0;
    n = 0;
    while(pp[p] < 0) p++;
    while(pn[n] < 0) n++;
    retVal.TP = pp.size()-p;
    retVal.FN = p;
    retVal.TN = n;
    retVal.FP = pn.size()-n;
    retVal.calculate();
}
// make sure that there are enough 
vector<double> autuRoc(threads &poll, \
        vector<vector<feature_node> > &tFeatures, \
        vector<double> &tTargetval,\
        vector<model*> &gmodel, \
        void*(*func)(void*) )
{
    vector< feature_node* > sampleFeatures;
    vector<double> sampleTargetval;
    vector<double> bias;
    vector<double> sampleTPR;
    vector<double> sampleFPR;
    sampleFPR.push_back(1);
    sampleFPR.push_back(0);
    sampleTPR.push_back(1);
    sampleTPR.push_back(0);
    bias.push_back(-4);
    bias.push_back(4);
    int perm = 49597; // a large prime number to get the permution
    int numTotalSize = (int)tFeatures.size();
    int numSamples = numTotalSize / 40;
    for(int i = 0; i < numSamples; i++){
        sampleFeatures.push_back(tFeatures[(i * perm) % numTotalSize].data());
        sampleTargetval.push_back(tTargetval[(i * perm) % numTotalSize]);
    }
    double diff;
    double b;
    int sapa;
    vector<double> predictTargetVal;
    predictTargetVal.reserve(sampleFeatures.size());
    while(1){
        sapa = 0;
        for(int i = 1; i < (int)sampleTPR.size(); i++){
            diff = sampleTPR[i-1] - sampleTPR[i];
            diff = max(diff,  sampleFPR[i-1] - sampleFPR[i]);
            if(diff > 0.2){
                sapa = i;
                break;
            }
        }
        if(sapa){
            b = (bias[sapa]+bias[sapa-1])/2;
        }else{
            break;
        }
        for(unsigned int i = 0; i < sampleFeatures.size(); i++){
            poll.addJob(func,\
                    new preadictThreadParams(&gmodel, \
                            sampleFeatures[i], \
                            &(predictTargetVal[i]), \
                            b));
        }
        poll.wait();
        predictResult pr = getPredictRes(sampleTargetval,predictTargetVal);
        // printf("%f %f %f \n", b, pr.TPR, pr.FPR);
        sampleFPR.insert(sampleFPR.begin()+sapa,pr.FPR);
        sampleTPR.insert(sampleTPR.begin()+sapa,pr.TPR);
        bias.insert(bias.begin()+sapa,b);
    }
    for(int i = 0; i < (int)bias.size();){
        if((sampleFPR[i] == 1 && sampleTPR[i]==1) || (sampleFPR[i] == 0 && sampleTPR[i]==0)){
            sampleFPR.erase(sampleFPR.begin()+i);
            sampleTPR.erase(sampleTPR.begin()+i);
            bias.erase(bias.begin()+i);
        }else{
            i++;
        }
    }
    return bias;
}
void getRoc(threads &poll, \
        vector<vector<feature_node> > &tFeatures, \
        vector<double> &tTargetval,\
        vector<model*> &gmodel, \
        void*(*func)(void*), \
        vector<double> &bias ,\
        predictResult & retVal )
{
    vector<double> predictTargetVal;
    predictTargetVal.clear();
    predictTargetVal.reserve(tTargetval.size());
    for(int j = 0; j < (int)bias.size(); j++){
        for(unsigned int i = 0; i < tFeatures.size(); i++){
            poll.addJob(func,\
                    new preadictThreadParams(&gmodel, \
                            tFeatures[i].data(), \
                            &(predictTargetVal[i]), \
                            bias[j]));
        }
        poll.wait();
        predictResult pr = getPredictRes(tTargetval,predictTargetVal);
        printf("rounf %d of %d ", j+1, (int)bias.size());
        printf(" %f %f \n", pr.TPR, pr.FPR);
        retVal.roc_tpr.push_back(pr.TPR);
        retVal.roc_fpr.push_back(pr.FPR);
    }
}
predictResult getPredictRes(vector<double> &targetVal, \
        vector<double> &predictTargetVal)
{
    predictResult pr;
    for(int i = 0; i < (int)targetVal.size(); i++){
        if(targetVal[i] == predictTargetVal[i]){
            if(targetVal[i] > 0){
                pr.TP++;
            }else{
                pr.TN++;
            }
        }else{
            if(targetVal[i]>0){
                pr.FN++;
            }else{
                pr.FP++;
            }
        }
    }
    pr.calculate();
    return pr;
}
void * predictThreadFunc(void *p){
    preadictThreadParams *  pp = (preadictThreadParams*)p;
    vector<model*>&         mod = *(pp->mod);
    vector<double>          mins;
    mins.reserve(NUM_POSITIVE);
    // MIN
    double t;
    for(int i = 0; i < NUM_POSITIVE; i++){
        t = predict_roc(mod[i],pp->f,pp->bias);
        for(int j = i+NUM_POSITIVE; j < NUM_GROUP;j+=NUM_POSITIVE){
            t = min(t,predict_roc(mod[j],pp->f,pp->bias));
        }
        mins[i] = t;
    }
    // MAX
    t = mins[0];
    for(int i = 1;i < NUM_POSITIVE;i++){
        t = max(t,mins[i]);
    }
    *(pp->retVal) = t;
    delete pp;
    return NULL;
}
void * trainThreadFunc(void * p){
    trainThreadParams * pa = (trainThreadParams*)p;
    *(pa->retData) = train(pa->prob,pa->param);
    delete pa;
    return NULL;
}
int getGroupParam(vector< vector<feature_node* > > &gFeature , \
        vector<vector<double> > &gTargetval , \
        vector<parameter> &retParam, \
        vector<problem>  &retProb )
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
    for (int i = 0; i < (int)features.size(); ++i){
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

// this group the data with the same section together.
// So there is only one possitive group, and serveral negiive group
int classify_1(vector< vector<lable_node> > &lables, \
        vector< vector<feature_node> >&features, \
        vector< vector<feature_node* > > & retFeature , \
        vector<vector<double> > &retTargetval)
{
    retFeature.clear();
    retTargetval.clear();
    // tmpval
    map<char, int>                  secToIndex;
    vector<vector<feature_node*> >  negGroups;
    vector<vector<double> >         negVals;
    vector<feature_node*>           possGroups;
    vector<double>                  possVals;
    // get single poss and negtive groups
    for(int i = 0; i < (int)lables.size(); i++){
        if(lables[i][0].Section == 'A'){
            possGroups.push_back(features[i].data());
            possVals.push_back(1);
        }else{
            if(secToIndex.find(lables[i][0].Section) == secToIndex.end()){
                secToIndex[lables[i][0].Section] = (int)negGroups.size();
                negGroups.push_back(vector<feature_node*>());
                negVals.push_back(vector<double> ());
            }
            negGroups[secToIndex[lables[i][0].Section]].push_back(features[i].data());
            negVals[secToIndex[lables[i][0].Section]].push_back(-1);
        }
    }
    NUM_POSITIVE = 1;
    NUM_NEGATIVE = (int)negGroups.size();
    NUM_GROUP = NUM_POSITIVE * NUM_NEGATIVE;
    // retFeature.reserve(NUM_GROUP);
    // retTargetval.reserve(NUM_GROUP);
    for(int i = 0; i < NUM_GROUP; i++){
        retFeature.push_back(possGroups);
        retTargetval.push_back(possVals);
    }
    for(int i = 0; i < NUM_NEGATIVE; i++){
        for(int j = 0; j < (int)negGroups[i].size(); j++){
            retFeature[i].push_back(negGroups[i][j]);
            retTargetval[i].push_back(negVals[i][j]);
        }
    }
    return 0;
}

void cardProduct(vector<vector<feature_node*> >  &negGroups,\
        vector<vector<double> >         &negVals,\
        vector<vector<feature_node*> >  &possGroups,\
        vector<vector<double> >         &possVals,\
        vector< vector<feature_node* > > & retFeature,\
        vector<vector<double> > &retTargetval        )
{
    NUM_POSITIVE = (int)possGroups.size();
    NUM_NEGATIVE = (int)negGroups.size();
    NUM_GROUP = NUM_POSITIVE * NUM_NEGATIVE;
    for(int i = 0; i < NUM_GROUP; i++){
        retFeature.push_back(vector<feature_node*>());
        retTargetval.push_back(vector<double>());
    }
    for(int i = 0; i < NUM_POSITIVE; i++){
        for(int j = 0; j < (int)possGroups[i].size(); j++){
            for(int k = i; k < NUM_GROUP; k+=NUM_POSITIVE){
                retFeature[k].push_back(possGroups[i][j]);
                retTargetval[k].push_back(1);
            }
        }
    }
    for(int i = 0; i < NUM_NEGATIVE; i++){
        for(int j = 0; j < (int)negGroups[i].size(); j++){
            for(int k = i * NUM_POSITIVE; k < i*NUM_POSITIVE+NUM_POSITIVE; k++){
                retFeature[k].push_back(negGroups[i][j]);
                retTargetval[k].push_back(-1);
            }
        }
    }

}
/*
this group the data with the same Class together.
So there is only one possitive group, and serveral negiive group
the code of class is like this:
    numOfClass = SectionNUm * 100 + classNum;
Since classNUm is a two digit number, it is enough to do so
*/
int classify_2(vector< vector<lable_node> > &lables, \
        vector< vector<feature_node> >&features, \
        vector< vector<feature_node* > > & retFeature , \
        vector<vector<double> > &retTargetval)
{
    retFeature.clear();
    retTargetval.clear();
    // tmpval
    map<int, int>                   secToIndex;
    vector<vector<feature_node*> >  negGroups;
    vector<vector<double> >         negVals;
    vector<vector<feature_node*> >  possGroups;
    vector<vector<double> >         possVals;
    printf("classify\n");
    // get single poss and negtive groups
    int code;
    for(int i = 0; i < (int)lables.size(); i++){
        for(int j = 0; j < (int)lables[i].size(); j++){
           code = ((int)lables[i][j].Section)*100 + lables[i][j].Class;
           if(lables[i][0].Section == 'A'){
                if(secToIndex.find(code)==secToIndex.end()){
                    secToIndex[code] = (int)possGroups.size();
                    possGroups.push_back(vector<feature_node*>());
                    possVals.push_back(vector<double>());
                }
                possGroups[secToIndex[code]].push_back(features[i].data());
                possVals[secToIndex[code]].push_back(1);
           }else{
                if(secToIndex.find(code)==secToIndex.end()){
                    secToIndex[code] = (int)negGroups.size();
                    negGroups.push_back(vector<feature_node*>());
                    negVals.push_back(vector<double>());
                }
                negGroups[secToIndex[code]].push_back(features[i].data());
                negVals[secToIndex[code]].push_back(1);
           }
        }
    }
    cardProduct(negGroups,negVals,possGroups,possVals,retFeature,retTargetval);
    printf("group finished\n");
    return 0;
}

bool compareVectorSize(vector<feature_node*>i, vector<feature_node*> j ){
    return i.size() < j.size();
}
// a <-- a U b;
// b = empty;
template <class T>
void mergeVector(vector<T> &a, vector<T> &b){
    a.insert(a.end(), b.begin(),b.end());
    b.clear();
}
// arrange group that each group is 0.75size ~~ 1.5 size
void makeGroup(vector<vector<feature_node*> > &t,int size){
    for(int i = 1; i < (int)t.size(); i++){
        if(t[i-1].size() + t[i].size() < size * 1.5){
            mergeVector(t[i],t[i-1]);
        }
    }
    for(int i = 0; i < (int)t.size(); ){
        if(t[i].size() > size*1.5){
            t.push_back(vector<feature_node*>());
            t.back().insert(t.back().end(),t[i].end()-t[i].size()/2,t[i].end());
            t[i].erase(t[i].end()-t[i].size()/2,t[i].end());
        }else{
            i++;
        }
    }
    for(vector<vector<feature_node*> >::iterator i = t.begin();i != t.end();){
        if(i->size()){
            i++;
        }else{
            i = t.erase(i);
        }
    }
}
/*
    classify using class information, balanced way
*/
int classify_3(vector< vector<lable_node> > &lables, \
        vector< vector<feature_node> >&features, \
        vector< vector<feature_node* > > & retFeature , \
        vector<vector<double> > &retTargetval)
{
    retFeature.clear();
    retTargetval.clear();
    // tmpval
    map<int, int>                   secToIndex;
    vector<vector<feature_node*> >  negGroups;
    vector<vector<double> >         negVals;
    vector<vector<feature_node*> >  possGroups;
    vector<vector<double> >         possVals;
    printf("classify\n");
    // get single poss and negtive groups
    int code;
    for(int i = 0; i < (int)lables.size(); i++){
        for(int j = 0; j < (int)lables[i].size(); j++){
           code = ((int)lables[i][j].Section)*100 + lables[i][j].Class;
           if(lables[i][0].Section == 'A'){
                if(secToIndex.find(code)==secToIndex.end()){
                    secToIndex[code] = (int)possGroups.size();
                    possGroups.push_back(vector<feature_node*>());
                    // possVals.push_back(vector<double>());
                }
                possGroups[secToIndex[code]].push_back(features[i].data());
                // possVals[secToIndex[code]].push_back(1);
           }else{
                if(secToIndex.find(code)==secToIndex.end()){
                    secToIndex[code] = (int)negGroups.size();
                    negGroups.push_back(vector<feature_node*>());
                    // negVals.push_back(vector<double>());
                }
                negGroups[secToIndex[code]].push_back(features[i].data());
                // negVals[secToIndex[code]].push_back(1);
           }
        }
    }
    sort(possGroups.begin(),possGroups.end(),compareVectorSize);
    sort(negGroups.begin(),negGroups.end(),compareVectorSize);
    // int meansize = (int)possGroups[possGroups.size()/2].size();
    int meansize = 16000;
    makeGroup(negGroups,meansize);
    makeGroup(possGroups,meansize);
    cardProduct(negGroups,negVals,possGroups,possVals,retFeature,retTargetval);
    // printf("group finished\n");
    return 0;
}

