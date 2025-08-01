
Route "globalized_gas_carrier_100k_cbm" {
	RouteType=REGIONAL_TRIP
	
	Ports=[Port("port_europe"), Port("port_asia"), Port("port_africa"), Port("port_americas"), Port("port_middle_east")]
	PortCalls = [6, 6, 6, 6, 6]
	
	TimeAtSea = 0.684 #250 days at sea
	ConditionDistribution = 				[1]
	#Speeds = 				[17]
	Speeds = 				[16.6]
	CapacityUtilizations = 	[0.5]
}
