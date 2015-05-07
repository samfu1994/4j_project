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

#ifndef INPUTLAYER_H_
#define INPUTLAYER_H_

#include "Layer.h"

namespace neural {

	// input layer class
	class InputLayer : public Layer {
		public:
			InputLayer(int nn, TransferFunction& tf) : Layer(NULL, nn, tf, 0.0f, 0.0f) { };
			
			void setInput(float* input_vector);
	};
	
}

#endif /*INPUTLAYER_H_*/
