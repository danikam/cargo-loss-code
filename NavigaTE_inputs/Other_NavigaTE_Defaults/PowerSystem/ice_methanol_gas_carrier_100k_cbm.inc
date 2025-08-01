# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

PowerSystem "ice_methanol_gas_carrier_100k_cbm" {
	Setup = INDIVIDUAL
	Propulsion = Converter("propulsion_ice_methanol_gas_carrier_100k_cbm")
	Electrical = Converter("electrical_ice_methanol_gas_carrier_100k_cbm")
	Heat = Converter("heat_boiler_oil_gas_carrier_100k_cbm")
	
	CAPEX = 0.65e6
	OPEX = 0.65e4
}







