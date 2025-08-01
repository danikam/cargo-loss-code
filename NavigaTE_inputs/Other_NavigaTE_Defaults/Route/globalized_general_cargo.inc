
Route "globalized_general_cargo" {
	RouteType=REGIONAL_TRIP
	
	Ports=[Port("port_europe"), Port("port_asia"), Port("port_africa"), Port("port_americas"), Port("port_middle_east")]
	PortCalls = [4, 4, 4, 4, 4]
	
	TimeAtSea = 0.548 #200 days at sea
	ConditionDistribution = [1]
	#Speeds = [11, 11]
	Speeds = [11.2]
	CapacityUtilizations = 	[0.5]
}