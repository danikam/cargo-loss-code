# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Converter "propulsion_ice_ammonia_tanker_35k_dwt" {
	PowerCapacity=8.5
	MinimumLoad=0.2
	MainFuelTypes = [AMMONIA]
	PilotFuelTypes = [HFO, DIESEL]
	MinimumPilotFuel = 0.05
	Efficiency = Forecast("propulsion_ice_ammonia_thermal_efficiency")

	set_consumption_TTW(AMMONIA, "nitrous_oxide", 0.)
	
	set_consumption_TTW(HFO, "black_carbon", Variable("converter_hfo_black_carbon"))
	set_consumption_TTW(DIESEL, "black_carbon", Variable("converter_diesel_black_carbon"))
	set_consumption_TTW(AMMONIA, "black_carbon", Variable("converter_ammonia_black_carbon"))
	
	CAPEX = 700000
	OPEX = <700000 * 0.015>
}
