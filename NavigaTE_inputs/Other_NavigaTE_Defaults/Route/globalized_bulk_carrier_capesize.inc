
Route "globalized_bulk_carrier_capesize" {
	RouteType=REGIONAL_TRIP
	Ports=[Port("port_europe"), Port("port_asia"), Port("port_africa"), Port("port_americas"), Port("port_middle_east")]
	PortCalls = [2, 2, 2, 2, 2]
	
	TimeAtSea = 0.698 # 255 days a year
	ConditionDistribution = [0.5, 0.5]
	#Speeds = [11, 14]
	Speeds = [11.2, 13.6]
	CapacityUtilizations = [1.0, 0.0]
}
