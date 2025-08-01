# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

PowerSystem "ice_ammonia_bulk_carrier_panamax" {
	Setup = INDIVIDUAL
	Propulsion = Converter("propulsion_ice_ammonia_bulk_carrier_panamax")
	Electrical = Converter("electrical_ice_ammonia_bulk_carrier_panamax")
	Heat = Converter("heat_boiler_oil_bulk_carrier_panamax")
	
	CAPEX = 3.448e6
	OPEX = 3.448e4
}







