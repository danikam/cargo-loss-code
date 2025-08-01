# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Converter "propulsion_ice_oil_tanker_100k_dwt" {
	PowerCapacity=13.9
	MinimumLoad=0.2
	MainFuelTypes = [HFO, DIESEL]
	Efficiency = Forecast("propulsion_ice_oil_thermal_efficiency")

	set_consumption_TTW(HFO, "black_carbon", Variable("converter_hfo_black_carbon"))
	set_consumption_TTW(DIESEL, "black_carbon", Variable("converter_diesel_black_carbon"))
	
	CAPEX = 520000
	OPEX = <520000 * 0.01>
}
