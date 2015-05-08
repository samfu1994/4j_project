#include "mlp.h"
#include <math.h>

double mlp::_random(double start, double end){
    return start+(end-start)*rand()/(RAND_MAX + 1.0);
}

double mlp::phi(double x){
    double dGain = 1.0;
    return 1.0 / (1.0 + exp(-dGain * x));
}

mlp::mlp(int _nInput, int _nHide ,int _nOutput){
    nInput = _nInput;
    nHide = _nHide;
    nOutput = _nOutput;
    for (int i = 0; i < nHide; ++i){
        wh.push_back(std::vector<double>(nInput+1));
        for (int j = 0; j < nInput; ++j){
            wh[i][j] = _random();
        }
    }
    for (int i = 0; i < nOutput; ++i){
        wo.push_back(std::vector<double>(nHide+1));
        for (int j = 0; j < nHide; ++j){
            wo[i][j] = _random();
        }
    }
    valh.reserve(nHide+1);
    input.reserve(nInput+1);
    output.reserve(nOutput);
}

void mlp::PropagateSignal(feature_node *x){
    for (int i = 0; i < nInput; ++i){
        input[i] = 0;
    }
    while(x->index != -1){
        input[x->index] = x->value;
        x++;
    }
    // compute val of hiden layer
    input[nInput] = 1; // bias
    for (int i = 0; i < nHide; ++i){
        valh[i] = 0;
        for (int j = 0; j < nInput+1; ++j){
            valh[i]+=wh[i][j]*input[j];
            valh[i] = phi(valh[i]);
        }
    }
    // conpute output layer
    valh[nHide] = 1;// bias
    for (int i = 0; i < nOutput; ++i){
        output[i] = 0;
        for (int j = 0; j < nHide+1; ++j){
            output[i]+=wo[i][j]*valh[j];
        }
    }
}

