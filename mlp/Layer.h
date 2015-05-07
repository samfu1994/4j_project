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

#ifndef LAYER_H_
#define LAYER_H_

#include "Neuron.h"
#include "TransferFunction.h"
#include <iostream>

namespace neural {
	using std::istream;
	using std::ostream;
	
	// base layer class
	class Layer {
		private:
			Layer* input_layer;
			float** w; // weights of synapses from input layer
			float* bw; // bias neuron weights
			float learning_rate;
			
		protected:
			int	nneurons;
			Neuron* neuron;
			TransferFunction& transfer_function;
			
		public:
			Layer(Layer* il, int nn, TransferFunction& tf, float lr, float wabsmax = 1.0f);
			~Layer();
			
			// execution and back-propagation algorithms
			void compute();
			void train();
			
			// file i/o
			void load(istream& is);
			void save(ostream& os);
	};
			
}

#endif /*LAYER_H_*/
