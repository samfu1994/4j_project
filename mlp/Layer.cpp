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

#include "Layer.h"
#include <cstdlib>

#define RANDOM_WEIGHT(absmax) (2.0f * ((float) (rand()) / (RAND_MAX) - 0.5f) * (absmax))

namespace neural {
	using std::rand;
	using std::endl;

	Layer::Layer(Layer* il, int nn, TransferFunction& tf, float lr, float wabsmax) : input_layer(il), learning_rate(lr), nneurons(nn), transfer_function(tf) 
	{
		neuron = new Neuron [nneurons];
		if (input_layer) {
			w = new float* [nneurons];
			bw = new float [nneurons];
			for (int i = 0; i < nneurons; ++i) {
				w[i] = new float [input_layer->nneurons];
				for (int j = 0; j < input_layer->nneurons; ++j)
					w[i][j] = RANDOM_WEIGHT(wabsmax);
				bw[i] = RANDOM_WEIGHT(wabsmax);
			}
		}
	}
	
	Layer::~Layer()
	{
		delete neuron;
		if (input_layer) {
			for (int i = 0; i < nneurons; ++i)
				delete[] w[i];
			delete[] w;
			delete[] bw;
		}
	}
	
	void Layer::compute()
	{
		if (input_layer) {
			for (int i = 0; i < nneurons; ++i) {
				neuron[i].input = 0.0f;
				for (int j = 0; j < input_layer->nneurons; ++j)
					neuron[i].input += input_layer->neuron[j].output * w[i][j];
				neuron[i].input += bw[i];
			}
		}
		for (int i = 0; i < nneurons; ++i)
			neuron[i].output = transfer_function(neuron[i].input);
	}
	
	void Layer::train()
	{
		if (input_layer) {
			for (int i = 0; i < nneurons; ++i) {
				neuron[i].delta = transfer_function.derivative(neuron[i].input, neuron[i].output) * neuron[i].error;	
				for (int j = 0; j < input_layer->nneurons; ++j)
					w[i][j] -= learning_rate * neuron[i].delta * input_layer->neuron[j].input;
				bw[i] -= learning_rate * neuron[i].delta;
			}
			for (int j = 0; j < input_layer->nneurons; ++j) {
				input_layer->neuron[j].error = 0;
				for (int i = 0; i < nneurons; ++i)
					input_layer->neuron[j].error += neuron[i].delta * w[i][j];	
			}
		}
	}
	
	void Layer::load(istream& is)
	{
		if (input_layer) {
			for (int i = 0; i < nneurons; ++i) {
				for (int j = 0; j < input_layer->nneurons; ++j)
					is >> w[i][j];
				is >> bw[i];
			}	
		}
	}
	
	void Layer::save(ostream& os)
	{
		if (input_layer) {
			for (int i = 0; i < nneurons; ++i) {
				for (int j = 0; j < input_layer->nneurons; ++j)
					os << w[i][j] << endl;
				os << bw[i] << endl;
			}	
		}
	}
	
}
