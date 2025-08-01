#!/bin/bash

# Source and destination directories
SRC_DIR="plots"
DEST_DIR="cargo_loss_paper_figures"

mkdir -p $DEST_DIR

# List of figure files to copy
figures=(
  "vessel_ranges.pdf"
  "boiloff_tank_size_factor_liquid_hydrogen.pdf"
  "tank_size_scaling_factors_V.pdf"
  "tank_size_scaling_factors_m.pdf"
  "V_c1_vs_V_o_tank1_methanol.png"
  "cargo_loss_mass_vs_m_o_tank1_m_c-1_methanol_legend.pdf"
  "cargo_loss_volume_vs_V_o_tank1_V_c-1_methanol.png"
  "cargo_loss_mass_vs_m_o_tank1_m_c-1_methanol.png"
  "cargo_loss_volume_vs_V_o_tank1_V_c-1_liquid_hydrogen.png"
  "cargo_loss_mass_vs_m_o_tank1_m_c-1_liquid_hydrogen.png"
  "cargo_loss_volume_vs_V_o_tank1_V_c-1_lng.png"
  "cargo_loss_mass_vs_m_o_tank1_m_c-1_lng.png"
  "cargo_loss_mass_vs_P_av1_m_c-1_methanol_with_legend.png"
  "cargo_loss_volume_vs_N_days1_liquid_hydrogen_with_legend.png"
  "boiloff_vs_N.pdf"
  "cargo_loss_mass_vs_f_p1_lng_with_legend.png"
  "cargo_loss_vs_sf_bulk_carrier_capesize_legend_only.pdf"
  "cargo_loss_vs_sf_bulk_carrier_handy_no_legend.pdf"
  "cargo_loss_vs_sf_container_15000_teu_no_legend.pdf"
  "modified_capacities_container_legend_stacked.pdf"
  "modified_capacities_gas_carrier_mass_central_with_sf_no_legend.pdf"
  "modified_capacities_container_mass_central_with_sf_no_legend.pdf"
  "modified_capacities_bulk_carrier_mass_central_with_sf_no_legend.pdf"
  "modified_capacities_tanker_mass_central_with_sf_no_legend.pdf"
  "sf_distribution_container.pdf"
  "modified_capacities_distribution_bulk_carrier_panamax_lsfo_mass_central_legendonly.pdf"
  "modified_capacities_distribution_bulk_carrier_panamax_lsfo_mass_central_nolegend.pdf"
  "modified_capacities_distribution_bulk_carrier_panamax_liquid_hydrogen_mass_central_nolegend.pdf"
  "modified_capacities_container_legend_stacked.pdf"
  "modified_capacities_bulk_carrier_mass_lower_with_sf_no_legend.pdf"
  "modified_capacities_bulk_carrier_mass_upper_with_sf_no_legend.pdf"
  "legend_only_lng_per_tonne_mile_final.pdf"
  "total_cost_comparison_liquid_hydrogen_per_tonne_mile_final_nolegend.pdf"
  "total_cost_comparison_compressed_hydrogen_per_tonne_mile_final_nolegend.pdf"
  "total_cost_comparison_lng_per_tonne_mile_final_nolegend.pdf"
  "cargo_loss_cost_summary.pdf"
  "sf_distribution_crude_oil.pdf"
  "commodity_sf_range_bulk.pdf"
  "bulk_carrier_commodity_tmiles.pdf"
  "commodity_sf_range_container.pdf"
  "container_commodity_tmiles.pdf"
  "commodity_sf_range_tanker.pdf"
  "tanker_commodity_tmiles.pdf"
)


# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Copy each file
for file in "${figures[@]}"; do
    if [ -f "$SRC_DIR/$file" ]; then
        cp "$SRC_DIR/$file" "$DEST_DIR/"
        echo "Copied $file"
    else
        echo "Warning: $file not found in $SRC_DIR"
    fi
done

