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

#ifndef HEAVISIDE_H_
#define HEAVISIDE_H_

#include "TransferFunction.h"

namespace neural {

	// Heaviside "step" function of parameter k
	class Heaviside : public TransferFunction {
		private:
			float k;
			
		public:
			Heaviside(float _k = 0) : k(_k) { }
			
			inline virtual float operator ()(float x)
			{
				return 	x <= k ? 0 : 1;
			}
			
			// this is not really the derivative, but in this case Widrow-Off
			// algorithm falls back to delta-rule
			inline virtual float derivative(float x, float y)
			{
				return 1;
			}
	};
	
}

#endif /*HEAVISIDE_H_*/
