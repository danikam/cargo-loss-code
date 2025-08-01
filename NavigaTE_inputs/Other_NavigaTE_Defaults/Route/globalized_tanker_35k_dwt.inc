
Route "globalized_tanker_35k_dwt" {
	RouteType=REGIONAL_TRIP
	
	Ports=[Port("port_europe"), Port("port_asia"), Port("port_africa"), Port("port_americas"), Port("port_middle_east")]
	PortCalls = [8, 8, 8, 8, 8]
	
	TimeAtSea = 0.5
	ConditionDistribution = [0.4, 0.6]
	#Speeds = [12, 13]
	Speeds = [10, 10]
	CapacityUtilizations = [0.0, 1.0]
}
