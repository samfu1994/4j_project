/*
	This file is part of NNF2.

	NNF2 is free software: you can redistribute it and/or modify it
	under the terms of the GNU General Public License as published 
	by the Free Software Foundation, either version 3 of the License,
	or (at your option) any later version.

	NNF2 is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with NNF2.  If not, see <http://www.gnu.org/licenses/>.
*/

// example: XOR function
// this simple application shows how to create layers of neurons, connect them
// and build a neural network to perform some task

#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <math.h>
#include "MultiLayerPerceptron.h"
#include "Sigmoid.h"
#include "Heaviside.h"

using namespace std;
using namespace neural;

#define XOR(a, b) (((a) && !(b)) || (!(a) && (b)))

int points = 100;

int main()
{
	// seed random number generator
	srand(time(NULL));
	
	// transfer functions
	Sigmoid sigmoid(1); // sigmoid of parameter 2.0
	Heaviside heaviside; // heaviside of default parameter 0
	Identity  identity;
	
	// learning rate and initial weight range
	float eps = 0.5f;
	float range = 1.0f;
	
	// input layer has 2 neurons, uses sigmoid transfer function
	InputLayer il(1, identity);
	
	// hidden layer is connected to input layer, has 4 neurons, uses sigmoid, 
	// has learning rate 'eps' and initial weights in (-range, range)
	Layer hl(&il, 10, sigmoid, eps, range);
	
	// output layer is connected to hidden layer, has 1 neuron, uses heaviside
	// transfer function
	OutputLayer ol(&hl, 1, sigmoid, eps, range);
	
	// MLP network constructor takes input and output layers and 
	// a NULL-terminated list of hidden layers in the same order 
	// they were connected
	MultiLayerPerceptron mlp(&il, &ol, &hl, NULL);
	
	// a simple way to train the network is to generate some examples
	// of input-output couples
	for (int epochs = 0; epochs < 10; ++epochs)
		for (int i = 0; i <= points; ++i)
		{
			float input[1] = {(float)i / points * (float)3.14};
			float desired_output = (float)((float)i / points * 3.14);
			mlp.train(input, &desired_output);	
		}
	
	// then we can test the network fitness so far
	float input[2] = {1, 0};
	float output;
	
	// let's see if the network has actually learned the XOR function
	for (int i = 0; i <= points; ++i)
	{
		input[0] = (float)i / points * (float)3.14;
		mlp.compute(input, &output);
		cout << (float)i / points * (float)3.14 << "\t" << output << endl;
	}
	
	return 0;
}
