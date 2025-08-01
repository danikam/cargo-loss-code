# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Converter "electrical_ice_methane_bulk_carrier_capesize" {
	PowerCapacity=2.
	MainFuelTypes = [METHANE]
	PilotFuelTypes = [DIESEL, HFO]
	MinimumPilotFuel = 0.01
	Efficiency = 0.48

	set_slip_TTW(METHANE, "methane", Forecast("methane_slip_electrical_ice"))
	
	set_consumption_TTW(HFO, "black_carbon", Variable("converter_hfo_black_carbon"))
	set_consumption_TTW(DIESEL, "black_carbon", Variable("converter_diesel_black_carbon"))
	set_consumption_TTW(METHANE, "black_carbon", Variable("converter_methane_black_carbon"))
	
	CAPEX = 480000
	OPEX = 48000
}
