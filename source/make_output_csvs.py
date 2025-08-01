"""
Date: July 25, 2024
Author: danikae
Purpose: Reads in NavigaTE outputs from a 2024 run including tankers, bulk vessels, containerships and gas carriers, and produces CSV files to process and structure the outputs for visualization.
"""

import functools
import glob
import os
import time

import numpy as np
import pandas as pd
from common_tools import get_top_dir
from parse import parse
import argparse

# Constants
TONNES_PER_TEU = 14
LB_PER_GAL_LNG = 3.49
GAL_PER_M3 = 264.172
LB_PER_TONNE = 2204.62
TONNES_PER_M3_LNG = LB_PER_GAL_LNG * GAL_PER_M3 / LB_PER_TONNE

# Get the path to the top level of the Git repo
top_dir = get_top_dir()

# Vessel type and size information
vessels = {
    "bulk_carrier_ice": [
        "bulk_carrier_capesize_ice",
        "bulk_carrier_handy_ice",
        "bulk_carrier_panamax_ice",
    ],
    "container_ice": [
        "container_15000_teu_ice",
        "container_8000_teu_ice",
        "container_3500_teu_ice",
    ],
    "tanker_ice": [
        "tanker_100k_dwt_ice",
        "tanker_300k_dwt_ice",
        "tanker_35k_dwt_ice",
    ],
    "gas_carrier_ice": ["gas_carrier_100k_cbm_ice"],
}

# Number of vessels for each type and size
vessel_size_number = {
    "bulk_carrier_capesize_ice": 2013,
    "bulk_carrier_handy_ice": 4186,
    "bulk_carrier_panamax_ice": 6384,
    "container_15000_teu_ice": 464,
    "container_8000_teu_ice": 1205,
    "container_3500_teu_ice": 3896,
    "tanker_100k_dwt_ice": 3673,
    "tanker_300k_dwt_ice": 866,
    "tanker_35k_dwt_ice": 8464,
    "gas_carrier_100k_cbm_ice": 2156,
}

# Quantities of interest
quantities = [
    "ConsumedEnergy_lsfo",
    "ConsumedEnergy_main",
    "CAPEX",
    "OPEX",
    "BaseCAPEX",
    "BaseOPEX",
    "TankCAPEX",
    "TankOPEX",
    "PowerCAPEX",
    "PowerOPEX",
    "FuelOPEX",
    "TotalCost",
]

# Evaluation choices
# per_*_mile_orig: Prior to tank size modification
evaluation_choices = [
    "per_year",
    "per_mile",
    "per_tonne_mile_lsfo",
    "per_tonne_mile_lsfo_final",
    "per_tonne_mile",
    "per_tonne_mile_final",
    "per_cbm_mile_lsfo",
    "per_cbm_mile_lsfo_final",
    "per_cbm_mile",
    "per_cbm_mile_final",
    "per_gj_fuel"
]

def time_function(func):
    """A decorator that logs the time a function takes to execute."""

    @functools.wraps(func)
    def wrapper_time_function(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds")
        return result

    return wrapper_time_function


def read_results(fuel, region, filename, all_results_df):
    """
    Reads the results from an Excel file and extracts relevant data for each vessel type and size.

    Parameters
    ----------
    fuel : str
        The type of fuel being used (eg. ammonia, hydrogen)

    region : str
        The region associated with the results.

    filename : str
        The path to the Excel file containing the results.

    all_results_df : pandas.DataFrame
        The DataFrame to which results will be appended.

    Returns
    -------
    all_results_df : pandas.DataFrame
        The updated DataFrame with the new results appended.
    """

    # Replace 'compressed_hydrogen' and 'liquid_hydrogen' in all_results_df with compressedhydrogen and liquidhydrogen to facilitate vessel name parsing
    fuel_orig = fuel
    fuel = fuel.replace("compressed_hydrogen", "compressedhydrogen").replace(
        "liquid_hydrogen", "liquidhydrogen"
    )

    # Define columns to read based on the fuel type
    if fuel == "lsfo":
        results_df_columns = [
            "Date",
            "Time (days)",
            "Miles",
            "CargoMiles",
            "CAPEX",
            "OPEX",
            "BaseCAPEX",
            "BaseOPEX",
            "TankCAPEX",
            "TankOPEX",
            "PowerCAPEX",
            "PowerOPEX",
            "FuelOPEX",
            "TotalCost",
            "ConsumedEnergy_lsfo",
        ]
    else:
        results_df_columns = [
            "Date",
            "Time (days)",
            "Miles",
            "CargoMiles",
            "CAPEX",
            "OPEX",
            "BaseCAPEX",
            "BaseOPEX",
            "TankCAPEX",
            "TankOPEX",
            "PowerCAPEX",
            "PowerOPEX",
            "FuelOPEX",
            "TotalCost",
            f"ConsumedEnergy_{fuel_orig}",
            "ConsumedEnergy_lsfo",
        ]
    # Read the results from the csv file
    results_df = pd.read_csv(filename)

    # Extract relevant data for each vessel type and size
    results_dict = {}
    for vessel_type in vessels:
        for vessel in vessels[vessel_type]:
            results_df_vessel = results_df.filter(regex=f"Date|Time|{vessel}").drop(
                [0, 1, 2]
            )
            results_df_vessel.columns = results_df_columns
            results_df_vessel = results_df_vessel.set_index("Date")

            results_dict["Vessel"] = f"{vessel}_{fuel}"
            results_dict["Fuel"] = fuel
            results_dict["Region"] = region
            results_dict["CAPEX"] = float(
                results_df_vessel["CAPEX"].loc["2025-01-01"]
            )
            results_dict["BaseCAPEX"] = float(
                results_df_vessel["BaseCAPEX"].loc["2025-01-01"]
            )
            results_dict["TankCAPEX"] = float(
                results_df_vessel["TankCAPEX"].loc["2025-01-01"]
            )
            results_dict["PowerCAPEX"] = float(
                results_df_vessel["PowerCAPEX"].loc["2025-01-01"]
            )
            results_dict["FuelOPEX"] = float(
                results_df_vessel["FuelOPEX"].loc["2025-01-01"]
            )
            results_dict["OPEX"] = float(
                results_df_vessel["OPEX"].loc["2025-01-01"]
            )
            results_dict["BaseOPEX"] = float(
                results_df_vessel["BaseOPEX"].loc["2025-01-01"]
            )
            results_dict["TankOPEX"] = float(
                results_df_vessel["TankOPEX"].loc["2025-01-01"]
            )
            results_dict["PowerOPEX"] = float(
                results_df_vessel["PowerOPEX"].loc["2025-01-01"]
            )
            results_dict["TotalCost"] = (
                float(results_df_vessel["CAPEX"].loc["2025-01-01"])
                + float(results_df_vessel["FuelOPEX"].loc["2025-01-01"])
                + float(results_df_vessel["OPEX"].loc["2025-01-01"])
            )
            results_dict["Miles"] = float(results_df_vessel["Miles"].loc["2025-01-01"])
            if "container" in vessel_type:
                results_dict["CargoMiles"] = (
                    float(results_df_vessel["CargoMiles"].loc["2025-01-01"])
                    * TONNES_PER_TEU
                )
            elif "gas_carrier" in vessel_type:
                results_dict["CargoMiles"] = (
                    float(results_df_vessel["CargoMiles"].loc["2025-01-01"])
                    * TONNES_PER_M3_LNG
                )
            else:
                results_dict["CargoMiles"] = float(
                    results_df_vessel["CargoMiles"].loc["2025-01-01"]
                )
            if fuel == "lsfo":
                results_dict["ConsumedEnergy_main"] = float(
                    results_df_vessel["ConsumedEnergy_lsfo"].loc["2025-01-01"]
                )
                results_dict["ConsumedEnergy_lsfo"] = (
                    float(results_df_vessel["ConsumedEnergy_lsfo"].loc["2025-01-01"])
                    * 0
                )
            else:
                results_dict["ConsumedEnergy_main"] = float(
                    results_df_vessel[f"ConsumedEnergy_{fuel_orig}"].loc["2025-01-01"]
                )
                results_dict["ConsumedEnergy_lsfo"] = float(
                    results_df_vessel["ConsumedEnergy_lsfo"].loc["2025-01-01"]
                )

            results_row_df = pd.DataFrame([results_dict])
            all_results_df = pd.concat(
                [all_results_df, results_row_df], ignore_index=True
            )

    return all_results_df


def extract_info_from_filename(filename):
    """
    Extracts the fuel from the given filename.

    Parameters
    ----------
    filename : str
        The filename from which to extract the information.

    Returns
    -------
    result.named : dict
        A dictionary containing the extracted information, or None if the pattern doesn't match.
    """
    pattern = "{fuel}_excel_report.csv"
    result = parse(pattern, filename)
    if result:
        return result.named
    return None


@time_function
def collect_all_results(label=None):
    """
    Collects all results from Excel files in the specified directory and compiles them into a DataFrame.

    Parameters
    ----------
    None

    Returns
    -------
    all_results_df : pandas.DataFrame
        A DataFrame containing all the collected results.
    """
    # List all files in the output directory
    input_dir = f"{top_dir}/all_outputs_full_fleet_csv"
    if label:
        input_dir = f"{top_dir}/all_outputs_full_fleet_{label}_csv"
    files = os.listdir(f"{input_dir}/")
    filename_info = [
        extract_info_from_filename(file)
        for file in files
        if extract_info_from_filename(file)
    ]

    # Initialize DataFrame to store all results
    columns = [
        "Vessel",
        "Fuel",
        "Region",
        "CAPEX",
        "FuelOPEX",
        "OPEX",
        "TotalCost",
        "Miles",
        "CargoMiles",
        "ConsumedEnergy_main",
        "ConsumedEnergy_lsfo",
    ]
    
    dtypes = {
    "Vessel": str, "Fuel": str, "Region": str,
    "CAPEX": float, "FuelOPEX": float, "OPEX": float, "TotalCost": float,
    "Miles": float, "CargoMiles": float,
    "ConsumedEnergy_main": float, "ConsumedEnergy_lsfo": float,
    }

    all_results_df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtypes.items()})

    # Read results for each file and add to the DataFrame
    for info in filename_info:
        fuel = info["fuel"]
        results_filename = f"{input_dir}/{fuel}_excel_report.csv"
        all_results_df = read_results(
            fuel, "Global", results_filename, all_results_df
        )
    return all_results_df


@time_function
def add_number_of_vessels(all_results_df):
    """
    Maps the number of vessels to each row in the DataFrame.

    Parameters
    ----------
    all_results_df : pandas.DataFrame
        The DataFrame containing the results to which the number of vessels will be added.

    Returns
    -------
    all_results_df : pandas.DataFrame
        The updated DataFrame with the number of vessels added.
    """

    def extract_base_vessel_name(vessel_name):
        return "_".join(vessel_name.split("_")[:-1])

    # Map the number of vessels to each row in the DataFrame
    all_results_df["base_vessel_name"] = all_results_df["Vessel"].apply(
        extract_base_vessel_name
    )
    all_results_df["n_vessels"] = (
        all_results_df["base_vessel_name"].map(vessel_size_number).astype(float)
    )
    all_results_df.drop("base_vessel_name", axis=1, inplace=True)

    return all_results_df

@time_function
def add_quantity_modifiers(all_results_df):
    """
    Adds quantity modifiers (e.g., per year, per mile, per tonne-mile) to the DataFrame based on the existing quantities.

    Parameters
    ----------
    all_results_df : pandas.DataFrame
        The DataFrame containing the results to which evaluated quantities will be added.

    Returns
    -------
    None
    """
    new_cols = {}
    for quantity in quantities:
        for evaluation_choice in evaluation_choices:
            if evaluation_choice == "per_mile":
                column_divide = "Miles"
            elif evaluation_choice == "per_tonne_mile":
                column_divide = "TonneMiles"
            elif evaluation_choice == "per_tonne_mile_final":
                column_divide = "FinalTonneMiles"
            elif evaluation_choice == "per_tonne_mile_lsfo":
                column_divide = "TonneMiles_lsfo"
            elif evaluation_choice == "per_tonne_mile_lsfo_final":
                column_divide = "FinalTonneMiles_lsfo"
            elif evaluation_choice == "per_cbm_mile":
                column_divide = "CbmMiles"
            elif evaluation_choice == "per_cbm_mile_final":
                column_divide = "FinalCbmMiles"
            elif evaluation_choice == "per_cbm_mile_lsfo":
                column_divide = "CbmMiles_lsfo"
            elif evaluation_choice == "per_cbm_mile_lsfo_final":
                column_divide = "FinalCbmMiles_lsfo"
            elif evaluation_choice == "per_gj_fuel":
                column_divide = "ConsumedEnergy_total"
            else:
                continue

            # Compute evaluated quantity
            new_col_name = f"{quantity}-{evaluation_choice}"
            new_cols[new_col_name] = all_results_df[quantity] / all_results_df[column_divide]

    # Add all new columns using concat to avoid fragmentation
    new_cols_df = pd.DataFrame(new_cols)

    # Use concat instead of column assignment to avoid insert-based fragmentation
    all_results_df = pd.concat([all_results_df, new_cols_df], axis=1)

    # Defragment the DataFrame after all column additions
    all_results_df = all_results_df.copy()

    return all_results_df


@time_function
def scale_quantities_to_fleet(all_results_df):
    """
    Scales quantities to the global fleet within each vessel type and size class by multiplying by the number of vessels of that type and class.

    Parameters
    ----------
    all_results_df : pandas.DataFrame
        The DataFrame containing the results to which fleet quantities will be added.

    Returns
    -------
    all_results_df : pandas.DataFrame
        The updated DataFrame with fleet-level quantities added.
    """
    for quantity in quantities + [
        "Miles",
        "CargoMiles",
        "CbmMiles",
        "TonneMiles",
        "CbmMiles_lsfo",
        "TonneMiles_lsfo",
        "FinalCbmMiles",
        "FinalCbmMiles_lsfo",
        "FinalTonneMiles",
        "FinalTonneMiles_lsfo",
    ]:
        # Multiply by the number of vessels to sum the quantity to the full fleet
        all_results_df[f"{quantity}-fleet"] = (
            all_results_df[quantity] * all_results_df["n_vessels"]
        )

    return all_results_df


@time_function
def add_vessel_type_quantities(all_results_df):
    quantities_fleet = [col for col in all_results_df.columns if "-fleet" in col]

    # Perform groupby once
    grouped = all_results_df.groupby(["Fuel", "Region"])

    new_rows = []
    for (fuel, region), group_df in grouped:
        for vessel_type, vessel_names in vessels.items():
            vessel_type_df = group_df[
                group_df["Vessel"].str.contains("|".join(vessel_names))
            ]

            if not vessel_type_df.empty:
                vessel_type_row = vessel_type_df[quantities_fleet].sum()
                vessel_type_row["Fuel"] = fuel
                vessel_type_row["Region"] = region
                vessel_type_row["Vessel"] = f"{vessel_type}_{fuel}"
                vessel_type_row["n_vessels"] = vessel_type_df["n_vessels"].sum()

                # The value for an individual vessel in the fleet is its total value scaled up to the fleet, divided by the total numbef of vessels of the given type in the fleet
                for quantity_fleet in quantities_fleet:
                    quantity_vessel = quantity_fleet.replace("-fleet", "")
                    vessel_type_row[quantity_vessel] = (
                        vessel_type_row[quantity_fleet] / vessel_type_row["n_vessels"]
                    )

                # Add rows in bulk
                new_rows.append(vessel_type_row)

    new_rows_df = pd.DataFrame(new_rows)
    return pd.concat([all_results_df, new_rows_df], ignore_index=True)

@time_function
def add_fleet_level_quantities(all_results_df):
    """
    Sums quantities in DataFrame to the full fleet, aggregating over all vessel types and sizes considered in the global fleet

    Parameters
    ----------
    all_results_df : pandas.DataFrame
        The DataFrame containing the results to which fleet-level quantities will be added.

    Returns
    -------
    all_results_df : pandas.DataFrame
        The updated DataFrame with fleet-level quantities added.
    """
    # Get a list of all vessels considered in the global fleet
    all_vessels = list(vessel_size_number.keys())

    # List of quantities that have already been scaled to the fleet level for individual vessel types and sizes (searchable with the '-fleet' keyword)
    quantities_fleet = [
        column for column in all_results_df.columns if "-fleet" in column
    ]

    new_rows = []

    # Iterate over each fuel and region combination in the dataframe.
    # For each such combo, the rows in all_results_df with vessels that match this combo are grouped into a single dataframe group_df
    for (fuel, region), group_df in all_results_df.groupby(
        ["Fuel", "Region"]
    ):
        # This filter ensures that we're only including base vessels (defined by both vessel type and size) when summing to the global fleet
        # This is necessary because we previously grouped base vessels into vessel types in add_vessel_type_quantities
        fleet_df = group_df[group_df["Vessel"].str.contains("|".join(all_vessels))]

        if not fleet_df.empty:
            # Sum quantities for the full fleet
            fleet_row = fleet_df[quantities_fleet].sum()
            fleet_row["Fuel"] = fuel
            fleet_row["Region"] = region
            fleet_row["Vessel"] = f"fleet_{fuel}"
            fleet_row["n_vessels"] = fleet_df["n_vessels"].sum()

            # Evaluate the average based on the fleet sum for each vessel-level quantity
            for quantity in quantities + [
                "Miles",
                "CargoMiles",
                "CbmMiles",
                "CbmMiles_lsfo",
                "TonneMiles",
                "TonneMiles_lsfo",
                "FinalCbmMiles",
                "FinalCbmMiles_lsfo",
                "FinalTonneMiles",
                "FinalTonneMiles_lsfo",
            ]:
                fleet_row[f"{quantity}"] = (
                    fleet_row[f"{quantity}-fleet"] / fleet_row["n_vessels"]
                )

            # Append the new row to the list
            new_rows.append(fleet_row)

    # Convert the list of new rows to a DataFrame and concatenate with the original DataFrame
    new_rows_df = pd.DataFrame(new_rows)
    all_results_df = pd.concat([all_results_df, new_rows_df], ignore_index=True)

    return all_results_df

@time_function
def add_bog_fuel_loss(all_results_df):
    """
    Adds in fuel losses due to reliquefaction of boiloff gas (BOG) for liquid hydrogen, ammonia, and methanol.

    Parameters
    ----------
    all_results_df : pandas.DataFrame
        The DataFrame containing the results to which fleet-level quantities will be added.

    Returns
    -------
    all_results_df : pandas.DataFrame
        The updated DataFrame with the boiloff added
    """

    # Load the tank size factors
    tank_size_factors = pd.read_csv(f"{top_dir}/tables/tank_size_factors.csv")

    # Prepare mapping of boiloff factors
    def map_boiloff_factors(row):
        fuel = row["Fuel"]
        vessel = row["Vessel"].split(f"_ice")[0]

        if fuel == "liquidhydrogen":
            fuel = "liquid_hydrogen"
        if fuel == "compressedhydrogen":
            fuel = "compressed_hydrogen"

        try:
            return tank_size_factors.loc[
                (tank_size_factors["Fuel"] == fuel)
                & (tank_size_factors["Vessel Class"] == vessel),
                "f_boiloff",
            ].iloc[0]
        except IndexError:
            return 1.0  # Default boiloff factor if none found

    # Vectorized calculation of boiloff factors
    all_results_df["Boil-off Factor"] = all_results_df.apply(
        map_boiloff_factors, axis=1
    )
    
    # Update columns using vectorized operations
    all_results_df["ConsumedEnergy_main"] *= all_results_df["Boil-off Factor"]
    all_results_df["FuelOPEX"] *= all_results_df["Boil-off Factor"]

    # Recalculate TotalCost
    all_results_df["TotalCost"] = (
        all_results_df["CAPEX"]
        + all_results_df["FuelOPEX"]
        + all_results_df["OPEX"]
    )

    # Drop the temporary Boil-off Factor column
    all_results_df = all_results_df.drop(columns=["Boil-off Factor"])

    return all_results_df


@time_function
def add_cargo_miles(all_results_df):
    """
    Adds the cargo miles under both mass-constrained (tonne-miles) and volume-constrained (m^3-miles) scenarios

    Parameters
    ----------
    all_results_df : pandas.DataFrame
        The DataFrame containing the results to which fleet-level quantities will be added.

    Returns
    -------
    all_results_df : pandas.DataFrame
        The updated DataFrame with the cost of carbon abatement added.
    """

    # Read in the csv file containing per-vessel cargo miles
    cargo_miles_df = pd.read_csv(f"tables/cargo_miles.csv")

    # Ensure proper matching of Vessel types without the "_ice" suffix
    all_results_df["Vessel_type"] = all_results_df["Vessel"].str.split("_ice").str[0]
    cargo_miles_df["Vessel_type"] = cargo_miles_df["Vessel"]

    # Rename the liquid hydrogen and compressed hydrogen fuels in cargo_miles_df to match all_results_df
    cargo_miles_df["Fuel_merge"] = cargo_miles_df["Fuel"].replace(
        {
            "liquid_hydrogen": "liquidhydrogen",
            "compressed_hydrogen": "compressedhydrogen",
        }
    )
    all_results_df["Fuel_merge"] = all_results_df["Fuel"]

    # Merge cargo_miles_df with all_results_df on Vessel_type and Fuel_merge
    all_results_df = all_results_df.merge(
        cargo_miles_df,
        how="left",
        left_on=["Vessel_type", "Fuel_merge"],
        right_on=["Vessel_type", "Fuel_merge"],
    )

    # Add additional columns for lsfo-specific cargo miles
    lsfo_cargo_miles = (
        cargo_miles_df[cargo_miles_df["Fuel"] == "lsfo"]
        .set_index("Vessel_type")[
            [
                "Cargo miles (m^3-miles)",
                "Cargo miles (tonne-miles)",
                "Final Cargo miles (m^3-miles)",
                "Final Cargo miles (tonne-miles)",
            ]
        ]
        .rename(
            columns={
                "Cargo miles (m^3-miles)": "CbmMiles_lsfo",
                "Cargo miles (tonne-miles)": "TonneMiles_lsfo",
                "Final Cargo miles (m^3-miles)": "FinalCbmMiles_lsfo",
                "Final Cargo miles (tonne-miles)": "FinalTonneMiles_lsfo",
            }
        )
    )

    all_results_df = all_results_df.merge(
        lsfo_cargo_miles, how="left", left_on="Vessel_type", right_index=True
    )

    # Drop the temporary columns used for merging
    all_results_df.drop(
        columns=["Vessel_type", "Fuel_merge", "Vessel_y", "Fuel_y", "Unnamed: 0"],
        inplace=True,
    )

    # Rename the columns from the merged data for clarity
    all_results_df.rename(
        columns={
            "Cargo miles (m^3-miles)": "CbmMiles",
            "Cargo miles (tonne-miles)": "TonneMiles",
            "Final Cargo miles (m^3-miles)": "FinalCbmMiles",
            "Final Cargo miles (tonne-miles)": "FinalTonneMiles",
            "Vessel_x": "Vessel",
            "Fuel_x": "Fuel",
        },
        inplace=True,
    )

    return all_results_df
    
@time_function
def add_total_consumed_energy(all_results_df):
    """
    Adds the total consumed energy (sum of pilot and alternative fuel)

    Parameters
    ----------
    all_results_df : pandas.DataFrame
        The DataFrame containing the results to which total consumed energy will be added

    Returns
    -------
    all_results_df : pandas.DataFrame
        The updated DataFrame with the cost of carbon abatement added.
        
    """
    
    # Get a list of all modifiers to handle
    all_modifiers = ["per_mile", "per_tonne_mile", "fleet"]
    
    # Calculate the product of cost times emissions
    all_results_df["ConsumedEnergy_total"] = all_results_df["ConsumedEnergy_main"] + all_results_df["ConsumedEnergy_lsfo"]

    return all_results_df


def remove_all_files_in_directory(directory_path):
    """
    Removes all files in the specified directory.

    Parameters
    ----------
    directory_path : str
        The path to the directory where all files will be removed.

    Returns
    -------
    None
    """
    files = glob.glob(os.path.join(directory_path, "*"))
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Error removing {f}: {e}")


@time_function
def generate_csv_files(all_results_df, label=None):
    """
    Generates and saves CSV files from the processed results DataFrame, organized by fuel and quantity.

    Parameters
    ----------
    all_results_df : pandas.DataFrame
        The DataFrame containing the processed results.

    Returns
    -------
    None
    """
    quantities_of_interest = list(
        all_results_df.drop(
            columns=[
                "Vessel",
                "Fuel",
                "Region",
                "n_vessels",
            ]
        ).columns
    )

    # Remove the fuel name from the vessel name since it's included in the filename
    all_results_df["Vessel"] = all_results_df["Vessel"].str.replace(
        r"_[^_]*$", "", regex=True
    )

    unique_fuels = all_results_df["Fuel"].unique()

    output_dir = f"{top_dir}/processed_results"
    if label:
        output_dir = f"{top_dir}/processed_results_{label}"

    remove_all_files_in_directory(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for fuel in unique_fuels:
        filter = (all_results_df["Fuel"] == fuel)
        all_selected_results_df = all_results_df[filter]
        for quantity in quantities_of_interest:
            quantity_selected_results_df = all_selected_results_df[
                ["Vessel", "Region", quantity]
            ]

            # Pivot the DataFrame
            pivot_df = quantity_selected_results_df.pivot(
                index="Region", columns="Vessel", values=quantity
            )

            # Replace NaN with zeros or any other value as needed
            pivot_df = pivot_df.fillna(0)

            # Ensure all data are numeric
            pivot_df = pivot_df.apply(pd.to_numeric, errors="coerce").fillna(0)

            # Identify countries with multiple entries
            countries_with_multiple_entries = quantity_selected_results_df[
                "Region"
            ][quantity_selected_results_df["Region"].str.contains("_2")]
            base_countries_with_multiple_entries = (
                countries_with_multiple_entries.apply(
                    lambda x: x.split("_")[0]
                ).unique()
            )

            # Rename base region rows with multiple entries
            for base_region in base_countries_with_multiple_entries:
                if base_region in pivot_df.index:
                    pivot_df.rename(
                        index={base_region: f"{base_region}_1"}, inplace=True
                    )

            # Calculate the average for each base region with multiple entries and add as a new row
            avg_rows = []
            for base_region in base_countries_with_multiple_entries:
                matching_rows = pivot_df.loc[
                    pivot_df.index.str.startswith(base_region + "_")
                ]
                if not matching_rows.empty:
                    avg_values = matching_rows.mean()
                    avg_values.name = base_region
                    avg_rows.append(avg_values)
            if avg_rows:
                avg_df = pd.DataFrame(avg_rows)
                pivot_df = pd.concat([pivot_df, avg_df])

            # Calculate the weighted average for each column, excluding the 'Weight' column itself
            global_avg = pivot_df.loc[~pivot_df.index.str.contains("_")].mean()

            # Add the weighted averages as a new row
            pivot_df.loc["Global Average"] = global_avg

            # If no modifier specified, add a modifier to indicate that the quantity is per-vessel
            if "-" not in quantity:
                quantity = f"{quantity}-vessel"

            # Update the fuel name for compressed/liquified hydrogen back to its original form with a '_' for file saving
            fuel_save = fuel
            if fuel == "compressedhydrogen":
                fuel_save = "compressed_hydrogen"
            if fuel == "liquidhydrogen":
                fuel_save = "liquid_hydrogen"

            # Generate the filename
            filename = f"{fuel_save}-{quantity}.csv"
            filepath = f"{output_dir}/{filename}"

            # Save the DataFrame to a CSV file
            pivot_df.to_csv(filepath)
    print(f"Saved processed csv files to {output_dir}")


def make_lower_heating_values_dict():
    """
    Makes a dictionary of lower heating values for each fuel, including entries
    without underscores for any fuel names containing '_'

    Parameters
    ----------
    None

    Returns
    -------
    lower_heating_values : Dictionary
        Dictionary containing the lower heating value for each fuel, in MJ / kg
    """

    info_file = f"{top_dir}/data/fuel_info.csv"
    lhv_info_df = pd.read_csv(
        info_file, usecols=["Fuel", "Lower Heating Value (MJ / kg)"], index_col="Fuel"
    )

    # Convert the dataframe into a dictionary such that each fuel will have a float entry corresponding to its LHV
    lower_heating_values = lhv_info_df["Lower Heating Value (MJ / kg)"].to_dict()

    # Add additional entries without underscores for fuels whose names contain underscores (eg. liquid_hydrogen)
    for fuel, lhv in list(lower_heating_values.items()):
        if "_" in fuel:
            lower_heating_values[fuel.replace("_", "")] = lhv

    return lower_heating_values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--include_bog_loss", action="store_true", help="Include fuel loss to reliquefaction of boil-off gas (BOG) in calculating total fuel costs")
    parser.add_argument("-i", "--input_label", default=None, help="Label to modify name of input directory")
    parser.add_argument("-o", "--output_label", default=None, help="Label to modify name of output directory")
    args = parser.parse_args()
    
    # Collect all results from the Excel files
    all_results_df = collect_all_results(args.input_label)

    # Add the number of vessels to the DataFrame
    all_results_df = add_number_of_vessels(all_results_df)

    # Add fuel needed to offset fuel loss due to reliquefaction of boiloff
    if args.include_bog_loss:
        all_results_df = add_bog_fuel_loss(all_results_df)

    # Add cargo miles (both tonne-miles and m^3-miles) to the dataframe
    all_results_df = add_cargo_miles(all_results_df)

    # Multiply by number of vessels of each type+size the fleet to get fleet-level quantities
    all_results_df = scale_quantities_to_fleet(all_results_df)

    # Group vessels by type to get type-level quantities
    all_results_df = add_vessel_type_quantities(all_results_df)

    # Group all vessel together to get fleet-level quantities
    all_results_df = add_fleet_level_quantities(all_results_df)
    
    # Add total GJ of fuel consumed
    all_results_df = add_total_consumed_energy(all_results_df)

    # Add evaluated quantities (per mile and per tonne-mile) to the dataframe
    all_results_df = add_quantity_modifiers(all_results_df)

    # Generate CSV files for each combination of fuel pathway, quantity, and evaluation choice
    generate_csv_files(all_results_df, args.output_label)


main()
