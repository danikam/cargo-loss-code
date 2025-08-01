Report "excel_report" {
    Directory="../../all_outputs_full_fleet"
    
    # Vessel properties
    add_vessel_property("*", Miles)                     # miles/year
    add_vessel_property("*", CargoMiles)                # tonne/miles per year
        
    add_vessel_property("*", CAPEX)
    add_vessel_property("*", OPEX)
    add_vessel_property("*", BaseCAPEX)
    add_vessel_property("*", BaseOPEX)
    add_vessel_property("*", TankCAPEX)
    add_vessel_property("*", TankOPEX)
    add_vessel_property("*", PowerCAPEX)
    add_vessel_property("*", PowerOPEX)
    add_vessel_property("*", FuelOPEX)
    add_vessel_property("*", TotalCost)
    add_vessel_property("*", ConsumedEnergy)            # Dictionary for each fuel
}
