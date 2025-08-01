
Route "globalized_container_8000_teu" {
	RouteType=REGIONAL_TRIP
	Ports=[Port("port_europe"), Port("port_asia"), Port("port_africa"), Port("port_americas"), Port("port_middle_east")]
	PortCalls = [20, 20, 20, 20, 20]

	TimeAtSea = 0.657 # 240 days a year
	#Times = [0.2, 0.8]
	ConditionDistribution = [0.15, 0.65, 0.20]
	Speeds = [12, 18, 21]
	#Speeds = [14, 19, 23]
	#Speeds = [12, 20]
	CapacityUtilizations = [0.65, 0.65, 0.65]
}
