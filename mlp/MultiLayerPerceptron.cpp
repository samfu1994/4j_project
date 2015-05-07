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

#include "MultiLayerPerceptron.h"
#include <cstdarg>
#include <cmath>

namespace neural {
	using std::va_list;

	MultiLayerPerceptron::MultiLayerPerceptron(InputLayer* il, OutputLayer* ol, ...)
	{
		input_layer = il;
		output_layer = ol;
		va_list args;
		va_start(args, ol);
		Layer* next_arg;
		while ((next_arg = va_arg(args, Layer*)) != NULL)
			hidden_layer.push_back(next_arg);
		va_end(args);
	}
	
	void MultiLayerPerceptron::setInput(float* input_vector)
	{
		input_layer->setInput(input_vector);	
	}
	
	void MultiLayerPerceptron::setDesired(float* desired_output_vector)
	{
		output_layer->setDesired(desired_output_vector);	
	}
	
	void MultiLayerPerceptron::getOutput(float* output_vector)
	{
		output_layer->getOutput(output_vector);	
	}
	
	void MultiLayerPerceptron::compute()
	{
		input_layer->compute();
		vector<Layer*>::iterator next_layer;
		for (next_layer = hidden_layer.begin(); next_layer != hidden_layer.end(); ++next_layer)
			(*next_layer)->compute();
		output_layer->compute();
	}
	
	void MultiLayerPerceptron::compute(float* input_vector)
	{
		setInput(input_vector);
		compute();	
	}
	
	void MultiLayerPerceptron::compute(float* input_vector, float* output_vector)
	{
		setInput(input_vector);
		compute();
		getOutput(output_vector);	
	}
	
	void MultiLayerPerceptron::train()
	{
		output_layer->train();
		vector<Layer*>::reverse_iterator next_layer;
		for (next_layer = hidden_layer.rbegin(); next_layer != hidden_layer.rend(); ++next_layer)
			(*next_layer)->train();
	}
	
	void MultiLayerPerceptron::train(float* desired_output_vector)
	{
		setDesired(desired_output_vector);
		train();	
	}
	
	void MultiLayerPerceptron::train(float* input_vector, float* desired_output_vector)
	{
		setInput(input_vector);
		compute();
		setDesired(desired_output_vector);
		train();	
	}
	
	float MultiLayerPerceptron::getError()
	{
		return output_layer->getError();
	}
	
	bool MultiLayerPerceptron::optimize(istream& data, float tolerance, int max_epochs, bool batch)
	{
		int ninput, noutput, ntestcases;
		data >> ninput >> noutput >> ntestcases;
		float input_vector[ntestcases][ninput], desired_output_vector[ntestcases][noutput];
		for (int i = 0; i < ntestcases; ++i) {
			for (int j = 0; j < ninput; ++j)
				data >> input_vector[i][j];
			for (int j = 0; j < noutput; ++j)
				data >> desired_output_vector[i][j];
		}
		float global_error;
		int epochs = 0;
		do {
			global_error = 0;
			for (int i = 0; i < ntestcases; ++i) {
				compute(input_vector[i]);
				train(desired_output_vector[i]);
				global_error += getError();
			}
			global_error /= ntestcases;
			++epochs;
		} while (global_error > tolerance && epochs < max_epochs);
		return (global_error <= tolerance);
	}
	
	void MultiLayerPerceptron::load(istream& is)
	{
		vector<Layer*>::iterator next_layer;
		for (next_layer = hidden_layer.begin(); next_layer != hidden_layer.end(); ++next_layer)
			(*next_layer)->load(is);
		output_layer->load(is);
	}
	
	void MultiLayerPerceptron::save(ostream& os)
	{
		vector<Layer*>::iterator next_layer;
		for (next_layer = hidden_layer.begin(); next_layer != hidden_layer.end(); ++next_layer)
			(*next_layer)->save(os);
		output_layer->save(os);
	}
	
}
