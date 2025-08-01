# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

PowerSystem "ice_methane_tanker_300k_dwt" {
	Setup = INDIVIDUAL
	Propulsion = Converter("propulsion_ice_methane_tanker_300k_dwt")
	Electrical = Converter("electrical_ice_methane_tanker_300k_dwt")
	Heat = Converter("heat_boiler_oil_tanker_300k_dwt")
	
	CAPEX = 9.9e6
	OPEX = 9.9e4
}







