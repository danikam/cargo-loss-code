# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping.
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.
#
# Description: Thermal efficiency of the propulsion (main engine) ICE using methanol
# Unit: GJ / GJ

Forecast "propulsion_ice_methanol_thermal_efficiency" {
	Table [
		"01-01-2024"	0.5279
		"01-01-2025"	0.5295
		"01-01-2026"	0.5311
		"01-01-2027"	0.5327
		"01-01-2028"	0.5343
		"01-01-2029"	0.5359
		"01-01-2030"	0.5375
		"01-01-2031"	0.5391
		"01-01-2032"	0.5407
		"01-01-2033"	0.5424
		"01-01-2034"	0.5440
		"01-01-2035"	0.5456
		"01-01-2036"	0.5472
		"01-01-2037"	0.5489
		"01-01-2038"	0.5505
		"01-01-2039"	0.5522
		"01-01-2040"	0.5538
		"01-01-2041"	0.5555
		"01-01-2042"	0.5572
		"01-01-2043"	0.5588
		"01-01-2044"	0.5605
		"01-01-2045"	0.5622
		"01-01-2046"	0.5639
		"01-01-2047"	0.5656
		"01-01-2048"	0.5673
		"01-01-2049"	0.5690
		"01-01-2050"	0.5707
	]
	Extrapolate=LINEAR
}
