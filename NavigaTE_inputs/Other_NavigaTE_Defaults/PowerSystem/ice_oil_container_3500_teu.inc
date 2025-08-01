# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

PowerSystem "ice_oil_container_3500_teu" {
	Setup = INDIVIDUAL
	Propulsion = Converter("propulsion_ice_oil_container_3500_teu")
	Electrical = Converter("electrical_ice_oil_container_3500_teu")
	Heat = Converter("heat_boiler_oil_container_3500_teu")
	
	CAPEX = 0.46e6
	OPEX = 0.46e4
}







