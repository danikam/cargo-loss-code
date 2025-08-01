# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Converter "heat_boiler_oil_container_3500_teu" {
	PowerCapacity=1.5
	MainFuelTypes = [HFO, DIESEL]
	Efficiency = 0.825

	CAPEX = 55e3
	OPEX = 0
}
