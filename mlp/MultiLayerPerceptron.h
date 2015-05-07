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

#ifndef MULTILAYERPERCEPTRON_H_
#define MULTILAYERPERCEPTRON_H_

#include "Layer.h"
#include "InputLayer.h"
#include "OutputLayer.h"
#include <iostream>
#include <vector>

namespace neural {
	using std::istream;
	using std::ostream;
	using std::vector;

	// MLP network class
	class MultiLayerPerceptron {
		private:
			InputLayer* input_layer;
			OutputLayer* output_layer;
			vector<Layer*> hidden_layer;
			
		public:
			// class constructor accepts a variable number of layers
			MultiLayerPerceptron(InputLayer* il, OutputLayer* ol, ...);
			
			// wrappers for input and output layer routines
			void setInput(float* input_vector);
			void setDesired(float* desired_output_vector);
			void getOutput(float* output_vector);
			
			// execution and back-propagation algorithms
			void compute();
			void compute(float* input_vector);
			void compute(float* input_vector, float* output_vector);
			
			void train();
			void train(float* desired_output_vector);
			void train(float* input_vector, float* desired_output_vector);
			
			// mean square error of network output
			float getError();
			
			// default training routine
			// returns true in case of success, false otherwise
			bool optimize(istream& data, float tolerance, int max_epochs, bool batch = false);
			
			// file i/o
			void load(istream& is);
			void save(ostream& os);
	};
	
}

#endif /*MULTILAYERPERCEPTRON_H_*/
