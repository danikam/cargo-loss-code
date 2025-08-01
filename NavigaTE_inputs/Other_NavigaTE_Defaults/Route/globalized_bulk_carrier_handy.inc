
Route "globalized_bulk_carrier_handy" {
	RouteType=REGIONAL_TRIP
	Ports=[Port("port_europe"), Port("port_asia"), Port("port_africa"), Port("port_americas"), Port("port_middle_east")]
	PortCalls = [4, 4, 4, 4, 4]
	
	TimeAtSea = 0.575 # 210 days a year
	ConditionDistribution = [0.5, 0.5]
	#Speeds = [11.5, 12]
	Speeds = [11.1, 11.6]
	CapacityUtilizations = [1.0, 0.0]
}
