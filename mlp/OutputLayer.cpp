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

#include "OutputLayer.h"
#include <cmath>

namespace neural {
	using std::pow;
	
	void OutputLayer::setDesired(float* desired_output_vector)
	{
		for (int i = 0; i < nneurons; ++i)
			neuron[i].error = neuron[i].output - desired_output_vector[i];	
	}
	
	void OutputLayer::getOutput(float* output_vector)
	{
		for (int i = 0; i < nneurons; ++i)
			output_vector[i] = neuron[i].output;	
	}

	float OutputLayer::getError()
	{
		float error = 0;
		for (int i = 0; i < nneurons; ++i)
			error += pow(neuron[i].error, 2);
		error /= nneurons;
		return error;
	}
	
}
