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

#ifndef SIGMOID_H_
#define SIGMOID_H_

#include "TransferFunction.h"
#include <cmath>

namespace neural {
	using std::exp;
	
	// sigmoid function of parameter k
	class Sigmoid : public TransferFunction {
		private:
			float k;
			
		public:
			Sigmoid(float _k = 1.0f) : k(_k) { }
			
			inline virtual float operator ()(float x)
			{
				return 	1.0f / (1.0f + exp(-k * x));
			}
			
			inline virtual float derivative(float x, float y)
			{
				return k * y * (1 - y);	
			}
	};
	
}

#endif /*SIGMOID_H_*/
