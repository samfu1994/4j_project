/*********************************************************************
 * File  : mlp.h
 * Author: Sylvain BARTHELEMY
 *         mailto:sylvain@sylbarth.com
 *         http://www.sylbarth.com
 * Date  : 2000-08
 *********************************************************************/


#ifndef _MLP_H_
#define _MLP_H_


struct Neuron {
  double  x;     /* sortie */
  double  e;     /* erreur */
  double* w;     /* poids  */
  double* dw;    /* dernier poids pour les momentum  */
  double* wsave; /* poids sauvegardé */
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


public:

  double dEta;
  double dAlpha;
  double dGain;
  double dAvgTestError;
  
  MultiLayerPerceptron(int nl, int npl[]);
  ~MultiLayerPerceptron();

  int Train(const char* fnames);
  int Test (const char* fname);
  int Evaluate();

  void Run(const char* fname, const int& maxiter);

};

#endif
