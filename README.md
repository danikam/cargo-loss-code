# Calculation and Analysis of Cargo Losses with Alternative Fuels
This repo contains input files and analysis code used to calculate and analyze tank scaling factors and resulting vessel cargo losses and cargo transport cost impacts associated with transitioning vessels to alternative fuels. 

The code leverages default inputs (version 2, time stamped on June 18, 2025) used to define representative vessels used by the [NavigaTE techno-economic fleet model](https://www.zerocarbonshipping.com/publications/navigate-explainer) developed by Frederik Lehn at the Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping. Currently, the full NavigaTE source code and executable are not publicly available. If the user has the NavigaTE command-line client installed, the code in this repo can additionally use the client to calculate cost impacts of cargo losses. Otherwise, pre-calculated model outputs are used instead.

The code in this repo is used to produce plots and tables in a working paper entitled "A Generalized Framework for Assessing Cargo Loss Impacts of Alternative Maritime Fuels" (preparing for submission to Transportation Research Part D).

# Pre-requisites
* Python 3.12
* If available, the `navigate` command-line client for the NavigaTE model (generated from the following commit hash on July 7, 2025: `be3692d810726948295c193000ab764173c475f8`)

# Environment setup

To install needed python packages, run:

```bash
pip install -r requirements.txt 
```

# Running the code

The bash file `source/run_all.sh` runs the full code pipeline. To execute:

```bash
bash source/run_all.sh
```

This executes the python scripts and NavigaTE model in sequence to produce a set of output tables and plots output to the `tables` and `plots` directory. The specific plots included in the working paper are automatically copied to the `cargo_loss_paper_figures` directory. 

As discussed above, `run_all.sh` will automatically check if the `navigate` command-line client is available. If the client is available, it will be used to generate model outputs that are subsequently analyzed by `source/analyze_cost_impact.py` to compare cargo transport costs with vs. without accounting for cargo loss. If the client is unavailable, pre-generated model outputs located in the `processed_results_*` directories will be used instead.

# Additional Input File Info
1. Default NavigaTE model inputs (version 2, time stamped on June 18, 2025) are located in the following directories:

* `NavigaTE_inputs/Vessel_Nominal`: Default include (.inc) files used by the NavigaTE model to define representative bulk carrier, container, tanker, and gas carrier vessels. Cargo capacities of all alternative fuel vessels are equal to that of the conventional fuel oil vessel for the nominal vessels. Note that the liquid hydrogen and compressed hydrogen vessels are not present in the default NavigaTE inputs, and vessel design parameters for these vessels are currently copied from the NavigaTE defaults for ammonia. 
* `NavigaTE_inputs/Tank_Nominal`: Default include (.inc) files used by the NavigaTE model to define tanks for the representative vessels defined in `NavigaTE_inputs/Vessel_Nominal`. Nominal capacities of alternative fuel tanks are equal to that of the conventional fuel oil vessel for the nominal vessels. As above, liquid hydrogen and compressed hydrogen tanks are not present in the default NavigaTE inputs, and tank design parameters for these vessels are currently copied from the NavigaTE defaults for ammonia. 

    Note that `NavigaTE_inputs/Vessel_Modified` and `NavigaTE_inputs/Tank_Modified` are automatically populated when `source/modify_tanks_and_cargo_capacity.py` is executed.

* `NavigaTE_inputs/Other_NavigaTE_Defaults`: Other default NavigaTE model input files used to calculate and analyze cargo losses and associated cargo transport cost impacts. These include:
  * `Converter` and `PowerSystem` directories: Contain files used to define costs and design specifications for each vessel's propulsion, electrical, and heat power systems.
  * `Route` directory: Contains inputs used to define speed and capacity utilization distributions for each representative vessel, along with the fraction of time at sea.
  * `Curve` and `Surface` directories: Contain inputs used to define speed-power curves or speed-utilization-power surfaces for representative vessels.
  * `Forecast` directory: Contains the projected thermal efficiency of vessel propulsion systems operating on different fuels.
  * `Variable` directory: Contains the assumed cost of capital for vessel newbuilds. 

2. The `single_pathway_full_fleet` and `includes_global` contain custom steering files input to the NavigaTE model to calculate the costs associated with operating each representative vessel on conventional oil and alternative fuels, either with or without accounting for tank scaling and cargo loss. Files produced by NavigaTE are output to the `all_outputs_full_fleet*` directories, which are subsequently processed by `source/convert_excel_files_to_csv.py` and `make_output_csvs.py` to produce csv-formatted outputs in the `processed_results_*` directories. If the NavigaTE executable is not detected on the user's system, the pre-calculated outputs in `processed_results_*` are automatically used by the subsequent analysis code (`source/analyze_cost_impact.py`), rather than attempting to re-generate the NavigaTE outputs.  

3. The `data` directory contains additional csv-formatted info used by the code, including:
* Fuel properties (`fuel_info.csv`)
* Assumed cargo densities used to calculate non-default vessel cargo capacities (`assumed_cargo_density.csv`)
* Stowage factor ranges of each commodity category included in the Freight Analysis Framework version 5  (`faf5_commodity_info.csv`)
* Freight Analysis Framework version 5.6.1 (FAF5) commodity flow data (`FAF5.6.1`), downloaded from https://faf.ornl.gov/faf5/. Note: to save space, only version pre-filtered for U.S. imports and exports carried by ship (`data/FAF5.6.1/FAF5.6.1_import_export_by_ship.csv`) is included in the repo. 
* Crude oil production data by API category (`Crude_Oil_and_Lease_Condensate_Production_by_API_Gravity.csv`), downloaded from https://www.eia.gov/dnav/pet/pet_crd_api_adc_mbblpd_m.htm 
