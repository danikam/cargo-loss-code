# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping.
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.
#
# Description: Thermal efficiency of the propulsion (main engine) ICE using methane
# Unit: GJ / GJ

Forecast "propulsion_ice_methane_thermal_efficiency" {
	Table [
		"01-01-2024"	0.5301
		"01-01-2025"	0.5317
		"01-01-2026"	0.5333
		"01-01-2027"	0.5349
		"01-01-2028"	0.5365
		"01-01-2029"	0.5381
		"01-01-2030"	0.5397
		"01-01-2031"	0.5413
		"01-01-2032"	0.5430
		"01-01-2033"	0.5446
		"01-01-2034"	0.5462
		"01-01-2035"	0.5479
		"01-01-2036"	0.5495
		"01-01-2037"	0.5511
		"01-01-2038"	0.5528
		"01-01-2039"	0.5545
		"01-01-2040"	0.5561
		"01-01-2041"	0.5578
		"01-01-2042"	0.5595
		"01-01-2043"	0.5611
		"01-01-2044"	0.5628
		"01-01-2045"	0.5645
		"01-01-2046"	0.5662
		"01-01-2047"	0.5679
		"01-01-2048"	0.5696
		"01-01-2049"	0.5713
"01-01-2050"	0.5730
	]
	Extrapolate=LINEAR
}
