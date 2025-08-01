
Route "globalized_tanker_300k_dwt" {
	RouteType=REGIONAL_TRIP
	
	Ports=[Port("port_europe"), Port("port_asia"), Port("port_africa"), Port("port_americas"), Port("port_middle_east")]
	PortCalls = [8, 8, 8, 8, 8]
	
	TimeAtSea = 0.821 # 300 days a year
	ConditionDistribution = [0.4, 0.6]
	#Speeds = [12, 13]
	#Speeds = [13.4, 14.4]
	Speeds = [12.7, 14]
	CapacityUtilizations = [0.0, 1.0]
}
