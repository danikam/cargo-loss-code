# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.
#
# Description: power demand needed to sail different speeds for given vessel.
# Unit: x = knots ; y = MW

Curve "speed_power_curve_gas_carrier_100k_cbm" {
	Table [
		10	2.0
		11	2.8
		12	3.6
		13	4.6
		14	5.8
		15	7.0
		16	8.5
		17	10.2
		18	12.0
	]
	
	Extrapolate=LINEAR
}
