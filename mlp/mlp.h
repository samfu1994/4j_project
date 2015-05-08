/*********************************************************************
 * File  : mlp.h
 * Author: Sylvain BARTHELEMY
 *         mailto:sylvain@sylbarth.com
 *         http://www.sylbarth.com
 * Date  : 2000-08
 *********************************************************************/


#ifndef _MLP_H_
#define _MLP_H_

#include <vector>
#include "../liblinear/linear.h"
using namespace std;


struct Neuron {
  double  x;     /* sortie */
  double  e;     /* erreur */
  double* w;     /* poids  */
  double* dw;    /* dernier poids pour les momentum  */
  double* wsave; /* poids sauvegard� */
};

struct Layer {
  int     nNumNeurons;
  Neuron* pNeurons;
};

class MultiLayerPerceptron {

  int    nNumLayers;
  Layer* pLayers;

  double dMSE;
  double dMAE;

  void RandomWeights();

  void SetInputSignal (double* input);
  void GetOutputSignal(double* output);

  void SaveWeights();
  void RestoreWeights();

  void PropagateSignal();
  void ComputeOutputError(double* target);
  void BackPropagateError();
  void AdjustWeights();

  void Simulate(double* input, double* output, double* target, bool training);

  void ConvertFeatureNode(const struct feature_node *x, double *t);

public:

  double dEta;
  double dAlpha;
  double dGain;
  double dAvgTestError;
  
  MultiLayerPerceptron(int nl, int npl[]);
  ~MultiLayerPerceptron();

  int Train(const char* fnames);
  int Train(const struct problem *prob,const struct parameter *param);
  int Test (const char* fname);
  int Test (const struct problem *prob,const struct parameter *param);
  double predict(const struct feature_node *x);

  void Run(const char* fname, const int maxiter);
  void Run(const struct problem *prob,const struct parameter *param, int maxiter);
};

#endif