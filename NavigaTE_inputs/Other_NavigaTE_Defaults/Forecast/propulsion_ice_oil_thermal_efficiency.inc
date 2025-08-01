# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping.
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.
#
# Description: Thermal efficiency of the propulsion (main engine) ICE using oil
# Unit: GJ / GJ

Forecast "propulsion_ice_oil_thermal_efficiency" {
	Table [
		"01-01-2024"	0.5143
		"01-01-2025"	0.5158
		"01-01-2026"	0.5174
		"01-01-2027"	0.5189
		"01-01-2028"	0.5205
		"01-01-2029"	0.5220
		"01-01-2030"	0.5236
		"01-01-2031"	0.5252
		"01-01-2032"	0.5268
		"01-01-2033"	0.5283
		"01-01-2034"	0.5299
		"01-01-2035"	0.5315
		"01-01-2036"	0.5331
		"01-01-2037"	0.5347
		"01-01-2038"	0.5363
		"01-01-2039"	0.5379
		"01-01-2040"	0.5395
		"01-01-2041"	0.5411
		"01-01-2042"	0.5428
		"01-01-2043"	0.5444
		"01-01-2044"	0.5460
		"01-01-2045"	0.5477
		"01-01-2046"	0.5493
		"01-01-2047"	0.5510
		"01-01-2048"	0.5526
		"01-01-2049"	0.5543
		"01-01-2050"	0.5559
	]
	Extrapolate=LINEAR
}
