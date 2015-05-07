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

#ifndef HYPERBOLICTANGENT_H_
#define HYPERBOLICTANGENT_H_

#include "TransferFunction.h"
#include <cmath>

namespace neural {
	using std::tanh;
	using std::pow;
	
	// hyperbolic tangent function
	class HyperbolicTangent : public TransferFunction {
		public:
			inline virtual float operator ()(float x)
			{
				return 	tanh(x);
			}
			
			inline virtual float derivative(float x, float y)
			{
				return 1 - pow(y, 2);	
			}
	};
	
}

#endif /*HYPERBOLICTANGENT_H_*/
