
Route "globalized_container_15000_teu" {
	RouteType=REGIONAL_TRIP
	Ports=[Port("port_europe"), Port("port_asia"), Port("port_africa"), Port("port_americas"), Port("port_middle_east")]
	PortCalls = [10, 10, 10, 10, 10]

	TimeAtSea = 0.657 # 240 days a year
	#Times = [0.10, 0.45, 0.45]
	ConditionDistribution = [0.10, 0.65, 0.25]
	Speeds = [12, 18, 21]
	#Speeds = [12.3, 17.3, 21.3]
	CapacityUtilizations = [0.65, 0.65, 0.65]
}
