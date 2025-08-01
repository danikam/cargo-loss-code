
Route "globalized_tanker_100k_dwt" {
	RouteType=REGIONAL_TRIP
	
	Ports=[Port("port_europe"), Port("port_asia"), Port("port_africa"), Port("port_americas"), Port("port_middle_east")]
	PortCalls = [8, 8, 8, 8, 8]
	
	TimeAtSea = 0.616 #225 days at sea
	ConditionDistribution = [0.4, 0.6]
	#Speeds = [12, 13]
	#Speeds = [13.1, 14.1]
	Speeds = [12.5, 13.5]
	CapacityUtilizations = [0.0, 1.0]
}
