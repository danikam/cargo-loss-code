# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping.
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.
#
# Description: Thermal efficiency of the propulsion (main engine) ICE using ammonia
# Unit: GJ / GJ

Forecast "propulsion_ice_ammonia_thermal_efficiency" {
	Table [
		"01-01-2024"	0.4925
		"01-01-2025"	0.4940
		"01-01-2026"	0.4954
		"01-01-2027"	0.4969
		"01-01-2028"	0.4984
		"01-01-2029"	0.4999
		"01-01-2030"	0.5014
		"01-01-2031"	0.5029
		"01-01-2032"	0.5044
		"01-01-2033"	0.5059
		"01-01-2034"	0.5075
		"01-01-2035"	0.5090
		"01-01-2036"	0.5105
		"01-01-2037"	0.5120
		"01-01-2038"	0.5136
		"01-01-2039"	0.5151
		"01-01-2040"	0.5167
		"01-01-2041"	0.5182
		"01-01-2042"	0.5198
		"01-01-2043"	0.5213
		"01-01-2044"	0.5229
		"01-01-2045"	0.5244
		"01-01-2046"	0.5260
		"01-01-2047"	0.5276
		"01-01-2048"	0.5292
		"01-01-2049"	0.5308
		"01-01-2050"	0.5324
	]
	Extrapolate=LINEAR
}
