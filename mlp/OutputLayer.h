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

#ifndef OUTPUTLAYER_H_
#define OUTPUTLAYER_H_

#include "Layer.h"

namespace neural {

	// output layer class
	class OutputLayer : public Layer {
		public:
			OutputLayer(Layer* il, int nn, TransferFunction& tf, float lr, float wabsmax = 1.0f) : Layer(il, nn, tf, lr, wabsmax) { };
			
			void setDesired(float* desired_output_vector);
			
			void getOutput(float* output_vector);
	
			// mean square error
			float getError();		
	};
	
}

#endif /*OUTPUTLAYER_H_*/
