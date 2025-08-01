# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Converter "electrical_ice_oil_gas_carrier_100k_cbm" {
	PowerCapacity=3.
	MainFuelTypes = [HFO, DIESEL]
	Efficiency = 0.5

	set_consumption_TTW(HFO, "black_carbon", Variable("converter_hfo_black_carbon"))
	set_consumption_TTW(DIESEL, "black_carbon", Variable("converter_diesel_black_carbon"))
	
	CAPEX = 400000
	OPEX = 40000
}
