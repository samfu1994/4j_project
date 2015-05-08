#ifndef _MLP_HEAD_
#define _MLP_HEAD_

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include "../liblinear/linear.h"

using namespace std;

class mlp{
private:
	// mata data
	int nInput;
	int nLayer;
    int nHide;
    int nOutput;

    // calculation data
    vector<double> 			input;
    vector<vector<double> > wh;
    vector<vector<double> > wo;
    vector<double > 		valh;
    vector<double> 			output;


    void PropagateSignal(feature_node *x);
    void BackPropagateError();
    double _random(double start = 0, double end = 1);
    double phi(double);

public:
	mlp(int _nInput, int _nHide ,int _nOutput);
	void train(const problem *prob, const parameter *param);
	void predict(const feature_node *x);

};

#endif