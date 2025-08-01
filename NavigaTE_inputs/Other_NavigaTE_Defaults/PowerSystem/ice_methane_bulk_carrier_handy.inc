# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

PowerSystem "ice_methane_bulk_carrier_handy" {
	Setup = INDIVIDUAL
	Propulsion = Converter("propulsion_ice_methane_bulk_carrier_handy")
	Electrical = Converter("electrical_ice_methane_bulk_carrier_handy")
	Heat = Converter("heat_boiler_oil_bulk_carrier_handy")
	
	CAPEX = 3.6e6
	OPEX = 3.6e4
}







