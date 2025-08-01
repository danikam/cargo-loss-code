# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Converter "propulsion_ice_methanol_bulk_carrier_panamax" {
	PowerCapacity=10.3
	MinimumLoad=0.2
	MainFuelTypes = [METHANOL]
	PilotFuelTypes = [DIESEL, HFO]
	MinimumPilotFuel = 0.05
	Efficiency = Forecast("propulsion_ice_methanol_thermal_efficiency")

	set_consumption_TTW(METHANOL, "black_carbon", Variable("converter_methanol_black_carbon"))

	
	CAPEX = 540000
	OPEX = <540000 * 0.015>
}
