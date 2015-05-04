/*********************************************************************
 * File  : test.cpp
 * Author: Sylvain BARTHELEMY
 *         mailto:sylvain@sylbarth.com
 *         http://www.sylbarth.com
 * Date  : 2000-08
 *********************************************************************/

#include "mlp.h"

int main(int argc, char* argv[])
{

  // int layers1[] = {2,2,1};
  // MultiLayerPerceptron mlp1(3,layers1);
  // mlp1.Run("xor.dat",10);

 int layers2[] = {1,100,1};
 MultiLayerPerceptron mlp2(3,layers2);
 mlp2.Run("sin.dat",5000);

  return 0;
}

