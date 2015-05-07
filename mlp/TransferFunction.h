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

#ifndef TRANSFERFUNCTION_H_
#define TRANSFERFUNCTION_H_

namespace neural {

	// default base class for transfer functions; default is identity
	class TransferFunction {
		public:
			inline virtual float operator ()(float x)
			{
				return x;	
			}
			
			// it should be quicker to calculate the derivative from the value
			// of the function when it satisfies a simple differential equation
			inline virtual float derivative(float x, float y)
			{
				return 1;	
			}
	};
	
	typedef TransferFunction Identity;
}

#endif /*TRANSFERFUNCTION_H_*/
