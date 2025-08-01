# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Converter "propulsion_ice_methane_container_15000_teu" {
	PowerCapacity=68.6
	MinimumLoad=0.15
	MainFuelTypes = [METHANE]
	PilotFuelTypes = [HFO, DIESEL]
	MinimumPilotFuel = 0.01
	Efficiency = Forecast("propulsion_ice_methane_thermal_efficiency")

	set_slip_TTW(METHANE, "methane", Forecast("methane_slip_propulsion_ice"))
	
	set_consumption_TTW(HFO, "black_carbon", Variable("converter_hfo_black_carbon"))
	set_consumption_TTW(DIESEL, "black_carbon", Variable("converter_diesel_black_carbon"))
	set_consumption_TTW(METHANE, "black_carbon", Variable("converter_methane_black_carbon"))
	
	CAPEX = 570000
	OPEX = <570000 * 0.015>
}
