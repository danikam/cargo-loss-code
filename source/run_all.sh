# Infer the top-level directory from the source dir
TOP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Make output directories, if needed
mkdir -p "${TOP_DIR}/all_outputs_full_fleet_nominal" "${TOP_DIR}/all_outputs_full_fleet_modified"

# Clear out all output figures and csv files produced by the code
rm "${TOP_DIR}/all_outputs_full_fleet_nominal/"* "${TOP_DIR}/all_outputs_full_fleet_modified/"*
rm "${TOP_DIR}/plots/"* "${TOP_DIR}/tables/"*


###################### Get SF distributions for each vessel class ######################
python "${TOP_DIR}/source/get_sf_distributions.py"
########################################################################################

###################### Get tank and cargo capacity modifications #######################
python "${TOP_DIR}/source/modify_tanks_and_cargo_capacity.py"
########################################################################################

## Analyze cargo loss with respect to design parameters for MC representative vessels ##
python "${TOP_DIR}/source/analyze_vessel_design_impacts.py"
########################################################################################

################# Make plot of boil-off factor vs. days to empty tank ##################
python "${TOP_DIR}/source/plot_beta_boiloff.py"
########################################################################################

############################## Make plot of vessel ranges ##############################
python "${TOP_DIR}/source/plot_vessel_ranges.py"
########################################################################################

########### Calculate the cost impact of tank scaling and cargo capacity loss ##########

# Check if the 'navigate' CLI tool is available
if command -v navigate &> /dev/null; then
  echo "Found 'navigate' CLI. Running simulations..."

  # Fuel oil (LSFO)
  navigate --suppress-plots "${TOP_DIR}/single_pathway_full_fleet/lsfo/lsfo_nominal.nav"
  mv "${TOP_DIR}/all_outputs_full_fleet/lsfo_nominal_excel_report.xlsx" "${TOP_DIR}/all_outputs_full_fleet_nominal/lsfo_excel_report.xlsx"

  navigate --suppress-plots "${TOP_DIR}/single_pathway_full_fleet/lsfo/lsfo_modified.nav"
  mv "${TOP_DIR}/all_outputs_full_fleet/lsfo_modified_excel_report.xlsx" "${TOP_DIR}/all_outputs_full_fleet_modified/lsfo_excel_report.xlsx"

  # Methanol
  navigate --suppress-plots "${TOP_DIR}/single_pathway_full_fleet/methanol/methanol_nominal.nav"
  mv "${TOP_DIR}/all_outputs_full_fleet/methanol_nominal_excel_report.xlsx" "${TOP_DIR}/all_outputs_full_fleet_nominal/methanol_excel_report.xlsx"

  navigate --suppress-plots "${TOP_DIR}/single_pathway_full_fleet/methanol/methanol_modified.nav"
  mv "${TOP_DIR}/all_outputs_full_fleet/methanol_modified_excel_report.xlsx" "${TOP_DIR}/all_outputs_full_fleet_modified/methanol_excel_report.xlsx"

  # Ammonia
  navigate --suppress-plots "${TOP_DIR}/single_pathway_full_fleet/ammonia/ammonia_nominal.nav"
  mv "${TOP_DIR}/all_outputs_full_fleet/ammonia_nominal_excel_report.xlsx" "${TOP_DIR}/all_outputs_full_fleet_nominal/ammonia_excel_report.xlsx"

  navigate --suppress-plots "${TOP_DIR}/single_pathway_full_fleet/ammonia/ammonia_modified.nav"
  mv "${TOP_DIR}/all_outputs_full_fleet/ammonia_modified_excel_report.xlsx" "${TOP_DIR}/all_outputs_full_fleet_modified/ammonia_excel_report.xlsx"

  # LNG
  navigate --suppress-plots "${TOP_DIR}/single_pathway_full_fleet/lng/lng_nominal.nav"
  mv "${TOP_DIR}/all_outputs_full_fleet/lng_nominal_excel_report.xlsx" "${TOP_DIR}/all_outputs_full_fleet_nominal/lng_excel_report.xlsx"

  navigate --suppress-plots "${TOP_DIR}/single_pathway_full_fleet/lng/lng_modified.nav"
  mv "${TOP_DIR}/all_outputs_full_fleet/lng_modified_excel_report.xlsx" "${TOP_DIR}/all_outputs_full_fleet_modified/lng_excel_report.xlsx"

  # Liquid Hydrogen
  navigate --suppress-plots "${TOP_DIR}/single_pathway_full_fleet/liquid_hydrogen/liquid_hydrogen_nominal.nav"
  mv "${TOP_DIR}/all_outputs_full_fleet/liquid_hydrogen_nominal_excel_report.xlsx" "${TOP_DIR}/all_outputs_full_fleet_nominal/liquid_hydrogen_excel_report.xlsx"

  navigate --suppress-plots "${TOP_DIR}/single_pathway_full_fleet/liquid_hydrogen/liquid_hydrogen_modified.nav"
  mv "${TOP_DIR}/all_outputs_full_fleet/liquid_hydrogen_modified_excel_report.xlsx" "${TOP_DIR}/all_outputs_full_fleet_modified/liquid_hydrogen_excel_report.xlsx"

  # Compressed Hydrogen
  navigate --suppress-plots "${TOP_DIR}/single_pathway_full_fleet/compressed_hydrogen/compressed_hydrogen_nominal.nav"
  mv "${TOP_DIR}/all_outputs_full_fleet/compressed_hydrogen_nominal_excel_report.xlsx" "${TOP_DIR}/all_outputs_full_fleet_nominal/compressed_hydrogen_excel_report.xlsx"

  navigate --suppress-plots "${TOP_DIR}/single_pathway_full_fleet/compressed_hydrogen/compressed_hydrogen_modified.nav"
  mv "${TOP_DIR}/all_outputs_full_fleet/compressed_hydrogen_modified_excel_report.xlsx" "${TOP_DIR}/all_outputs_full_fleet_modified/compressed_hydrogen_excel_report.xlsx"

  # Convert Excel outputs to csv
  python "${TOP_DIR}/source/convert_excel_files_to_csv.py" --input_dir "${TOP_DIR}/all_outputs_full_fleet_nominal" --output_dir "${TOP_DIR}/all_outputs_full_fleet_nominal_csv"
  python "${TOP_DIR}/source/convert_excel_files_to_csv.py" --input_dir "${TOP_DIR}/all_outputs_full_fleet_modified" --output_dir "${TOP_DIR}/all_outputs_full_fleet_modified_csv"

  # Convert output csv files into a format that can be used by analyze_cost_impact.py
  python "${TOP_DIR}/source/make_output_csvs.py" -i nominal -o nominal_tanks_no_bog_fuel_loss
  python "${TOP_DIR}/source/make_output_csvs.py" -i modified -o modified_tanks_no_bog_fuel_loss
  python "${TOP_DIR}/source/make_output_csvs.py" -b -i modified -o modified_tanks_with_bog_fuel_loss

else
  echo "'navigate' command-line tool not found. Skipping simulations and using pre-generated outputs located in the processed_results_* directories..."
fi

# Calculate and plot the cost impacts
python "${TOP_DIR}/source/analyze_cost_impact.py"

########################################################################################

################### Copy plots used in paper to a dedicated directory ##################
bash "${TOP_DIR}/source/copy_plots.sh"
########################################################################################
