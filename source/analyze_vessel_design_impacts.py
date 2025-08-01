"""
Date: Dec. 17, 2024
Author: danikae
Purpose: Analyze the impact of vessel design parameters on the cargo displacement due to increased tank sizes needed for alternative fuels.
"""

import os
import random
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common_tools import get_fuel_label, get_top_dir
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import differential_evolution, minimize

KG_PER_TONNE = 1000
L_PER_CBM = 1000
S_PER_DAY = 86400
H_PER_DAY = 24
S_PER_H = 3600

def collect_nominal_vessel_design_params(top_dir):
    """
    Collects the nominal values of vessel design parameters considered in this analysis for each vessel defined in NavigaTE.

    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.

    Returns
    -------
    vessel_design_params : pd.DataFrame
        DataFrame with the info collected for each vessel
    """
    vessel_design_params_df = pd.read_csv(f"{top_dir}/tables/vessel_info.csv")
    vessel_design_params_df["Nominal Cargo Capacity (kg)"] = (
        vessel_design_params_df["Nominal Cargo Capacity (tonnes)"] * KG_PER_TONNE
    )
    return vessel_design_params_df


def extract_vessel_design_params(vessel_design_params_df, vessel):
    """
    Extracts the vessel design parameters for a given vessel as a dictionary, using the dataframe containing parameters for all vessels.

    Parameters
    ----------
    vessel_design_params_df : pandas.DataFrame
        DataFrame containing the design parameters for all vessels.
    vessel : str
        Name of the vessel, as it's stored in the dataframe.

    Returns
    -------
    vessel_design_params_dict : dict
        Dictionary with the nominal design parameters for the given vessel.
    """
    # Filter the DataFrame for the specified vessel
    vessel_row = vessel_design_params_df[vessel_design_params_df["Vessel"] == vessel]

    # Check if the vessel exists in the DataFrame
    if vessel_row.empty:
        raise ValueError(f"Vessel '{vessel}' not found in the dataframe.")

    # Extract the row as a dictionary (reset index to ensure correct format)
    vessel_design_params_dict = vessel_row.iloc[0].to_dict()

    # Drop the index column (e.g., "Unnamed: 0")
    vessel_design_params_dict.pop("Unnamed: 0", None)

    return vessel_design_params_dict


def collect_fuel_params(top_dir):
    """
    Collects the vessel design parameters considered in this analysis for each vessel defined in NavigaTE.

    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.

    Returns
    -------
    fuel_params : pd.DataFrame
        DataFrame with the info collected for each vessel
    """
    fuel_params_df = pd.read_csv(f"{top_dir}/tables/fuel_info.csv")
    fuel_params_df["Mass density (kg/m^3)"] = (
        fuel_params_df["Mass density (kg/L)"] * L_PER_CBM
    )

    return fuel_params_df


def extract_fuel_params(fuel_params_df, fuel):
    """
    Extracts the fuel parameters for a given fuel as a dictionary, using the dataframe containing parameters for all fuels.

    Parameters
    ----------
    fuel_params_df : pandas.DataFrame
        DataFrame containing the fuel parameters for all fuels.
    fuel : str
        Name of the fuel, as it's stored in the dataframe.

    Returns
    -------
    fuel_params_dict : dict
        Dictionary with the parameters for the given fuel.
    """
    # Filter the DataFrame for the specified fuel
    fuel_row = fuel_params_df[fuel_params_df["Fuel"] == fuel]

    # Check if the vessel exists in the DataFrame
    if fuel_row.empty:
        raise ValueError(f"Fuel '{fuel}' not found in the dataframe.")

    # Extract the row as a dictionary (reset index to ensure correct format)
    fuel_params_dict = fuel_row.iloc[0].to_dict()

    # Drop the index column (e.g., "Unnamed: 0")
    fuel_params_dict.pop("Unnamed: 0", None)

    return fuel_params_dict


def calculate_days_to_empty_tank(R, s_av):
    """
    Calculates the total number of days needed to empty the vessel's tank

    Parameters
    ----------
    R : float
        Vessel design range, in nautical miles

    s_av : float
        Average vessel speed

    Returns
    -------
    N_days : float
        Average number of days the vessel takes to empty its tank
    """

    N_hours = R / s_av
    N_days = N_hours / H_PER_DAY

    return N_days

def calculate_fuel_term(L_l, rho_l, e_l, L_f, rho_f, e_f, cargo_type="mass"):
    """
    Calculates the dimensionless fuel term in the equation for fractional cargo capacity loss

    Parameters
    ----------
    L_l : float
        Lower heating value of LSFO fuel (MJ/kg)

    rho_l : float
        Mass density of LSFO fuel (kg/m^3)

    e_l : float
        Energy conversion efficiency of LSFO fuel by the main engine

    L_l : float
        Lower heating value of the given fuel (MJ/kg)

    rho_l : float
        Mass density of the given fuel (kg/m^3)

    e_l : float
        Energy conversion efficiency of the given fuel by the main engine

    Returns
    -------
    fuel_term : float
        Dimensionless fuel term
    """

    if cargo_type == "mass":
        fuel_term = L_l * e_l / (L_f * e_f)
        return fuel_term

    elif cargo_type == "volume":
        fuel_term = L_l * rho_l * e_l / (L_f * rho_f * e_f)
        return fuel_term

    else:
        print(
            f"Error: Cargo type {cargo_type} not recognized. Currently accepted cargo types are mass and volume."
        )
        return


def calculate_scaling_term(V_o_tank, rho_l, m_c, V_c, cargo_type="mass"):
    """
    Calculates the dimensionless scaling term in the equation for fractional cargo capacity loss

    Parameters
    ----------
    V_o_tank : float
        Nominal volume capacity of the vessel's tank

    rho_l : float
        Mass density of the given fuel (kg/m^3)

    e_l : float
        Energy conversion efficiency of the given fuel by the main engine

    m_c : float
        Mass cargo capacity of the vessel, in kg

    V_c : float
        Volume cargo capacity of the vessel, in m^3

    Returns
    -------
    fuel_term : float
        Dimensionless fuel term
    """

    if cargo_type == "mass":
        return V_o_tank * rho_l / m_c

    elif cargo_type == "volume":
        return V_o_tank / V_c

    else:
        print(
            f"Error: Cargo type {cargo_type} not recognized. Currently accepted cargo types are mass and volume."
        )
        return


def calculate_boiloff_term(r_f, N_days, f_port):
    """
    Calculates the dimensionless boil-off term in the equation for fractional cargo capacity loss

    Parameters
    ----------
    r_f : float
        Fractional loss of fuel to boil-off per day

    N_days : float
        Number of days the vessel takes to empty its fuel tank

    Returns
    -------
    boiloff_term : float
        Dimensionless boiloff term
    """
    
    r_f = r_f / 100 # Convert from percent to relative boiloff rate
    if r_f == 0:
        boiloff_term = 1
    else:
        boiloff_term = (1/N_days) * (
            (1 / (1 - r_f) ** (f_port * N_days))
            * (1 - (1 - r_f) ** (f_port * N_days))
            / (1 - (1 - r_f)**f_port)
        )
    return boiloff_term


def calculate_cargo_loss(
    R,
    V_o_tank,
    s_av,
    r_f,
    e_l,
    L_l,
    rho_l,
    e_f,
    L_f,
    rho_f,
    m_c,
    V_c,
    f_p,
    f_port,
    cargo_type="mass",
):
    """
    Calculates the fractional cargo loss due to tank displacement.

    Parameters
    ----------

    R : float
        Vessel design range (nautical miles)

    V_o_tank : float
        Nominal volume capacity of the vessel's main tank

    s_av : float
        Average vessel speed, in knots

    r_f : float
        Fractional loss of fuel to boil-off per day

    e_l : float
        Energy conversion efficiency of LSFO fuel by the main engine

    L_l : float
        Lower heating value of LSFO fuel (MJ/kg)

    rho_l : float
        Mass density of LSFO fuel (kg/m^3)

    e_f : float
        Energy conversion efficiency of the given fuel by the main engine

    L_f : float
        Lower heating value of the given fuel (MJ/kg)

    rho_f : float
        Mass density of the given fuel (kg/m^3)

    m_c : float
        Mass cargo capacity of the vessel, in kg

    V_c : float
        Volume cargo capacity of the vessel, in m^3
    
    f_p : float
        Pilot tank size ratio
    
    f_port : float
        Fraction of time spent at port

    Returns
    -------
    cargo_loss : float
        Fractional cargo loss
    """
    scaling_term = calculate_scaling_term(
        V_o_tank, rho_l, m_c, V_c, cargo_type
    )
    fuel_term = calculate_fuel_term(L_l, rho_l, e_l, L_f, rho_f, e_f, cargo_type)

    # If the daily fuel loss to boil-off is zero, don't bother calculating the boil-off term
    if np.absolute(r_f) < 0.000001:
        cargo_loss = scaling_term * (fuel_term + f_p - 1)
    else:
        N_days_to_empty = calculate_days_to_empty_tank(R, s_av)
        boiloff_term = calculate_boiloff_term(r_f, N_days_to_empty, f_port)
        cargo_loss = scaling_term * (fuel_term * boiloff_term + f_p - 1)
    
    return cargo_loss


def get_parameter_values(
    nominal_vessel_params_df, fuel_params_df, fuel="liquid_hydrogen"
):
    """
    Calculates all parameters related to cargo loss for all vessels for the given fuel, evaluates the maximum and minimum value of each parameter over all vessels, and saves the results to a dataframe.

    Parameters
    ----------
    nominal_vessel_params_df : pd.DataFrame
        Dataframe containing nominal parameters for each vessel

    fuel_params_df : pd.DataFrame
        Dataframe containing parameters for each fuel

    fuel : str
        Fuel to get parameter values for

    Returns
    -------
    parameter_values_df : pd.DataFrame
        Dataframe containing parameters related to cargo loss for each vessel, along with maxima and minima
    """
    top_dir = get_top_dir()
    fuel_params_lsfo = extract_fuel_params(fuel_params_df, "lsfo")
    fuel_params_fuel = extract_fuel_params(fuel_params_df, fuel)
    boiloff_params_df = pd.read_csv(f"{top_dir}/tables/effective_boiloff_rates.csv").set_index("Fuel")

    r_f = boiloff_params_df.loc[fuel, "Effective Boiloff Rate (%/day)"]
    e_l = fuel_params_lsfo["eta_eff_mean"]
    L_l = fuel_params_lsfo["Lower Heating Value (MJ / kg)"]
    rho_l = fuel_params_lsfo["Mass density (kg/m^3)"]
    e_f = fuel_params_fuel["eta_eff_mean"]
    L_f = fuel_params_fuel["Lower Heating Value (MJ / kg)"]
    rho_f = fuel_params_fuel["Mass density (kg/m^3)"]
    fuel_term_mass = calculate_fuel_term(
        L_l, rho_l, e_l, L_f, rho_f, e_f, cargo_type="mass"
    )
    fuel_term_volume = calculate_fuel_term(
        L_l, rho_l, e_l, L_f, rho_f, e_f, cargo_type="volume"
    )

    minima = {
        "Vessel": "Minimum",
        "R (nm)": 9e9,
        "P_av (MW)": 9e9,
        "V_o_tank (m^3)": 9e9,
        "m_o_tank (kg)": 9e9,
        "s_av (knots)": 9e9,
        "m_c (kg)": 9e9,
        "V_c (m^3)": 9e9,
        "f_p": 9e9,
        "f_port": 9e9,
        "N_days": 9e9,
        "r_f": 9e9,
        "scaling_term_volume": 9e9,
        "fuel_term_volume": 9e9,
        "scaling_term_mass": 9e9,
        "fuel_term_mass": 9e9,
        "boiloff_term": 9e9,
        "cargo_loss_volume": 9e9,
        "cargo_loss_mass": 9e9,
    }

    maxima = {
        "Vessel": "Maximum",
        "R (nm)": -9e9,
        "P_av (MW)": -9e9,
        "V_o_tank (m^3)": -9e9,
        "m_o_tank (kg)": -9e9,
        "s_av (knots)": -9e9,
        "m_c (kg)": -9e9,
        "V_c (m^3)": -9e9,
        "f_p": -9e9,
        "f_port": -9e9,
        "N_days": -9e9,
        "r_f": -9e9,
        "scaling_term_volume": -9e9,
        "fuel_term_volume": -9e9,
        "boiloff_term": -9e9,
        "scaling_term_mass": -9e9,
        "fuel_term_mass": -9e9,
        "cargo_loss_volume": -9e9,
        "cargo_loss_mass": -9e9,
    }

    data = []
    for vessel in nominal_vessel_params_df["Vessel"]:
        parameter_value_dict = {}
        nominal_vessel_params = extract_vessel_design_params(
            nominal_vessel_params_df, vessel
        )
        parameter_value_dict["Vessel"] = vessel
        parameter_value_dict["R (nm)"] = nominal_vessel_params[
            "Nominal Range (nautical miles)"
        ]
        parameter_value_dict["P_av (MW)"] = nominal_vessel_params[
            "Average Vessel Power (MW)"
        ]
        parameter_value_dict["s_av (knots)"] = nominal_vessel_params[
            "Average Speed (knots)"
        ]
        parameter_value_dict["m_c (kg)"] = nominal_vessel_params[
            "Nominal Cargo Capacity (kg)"
        ]
        parameter_value_dict["V_c (m^3)"] = nominal_vessel_params[
            "Nominal Cargo Capacity (m^3)"
        ]
        parameter_value_dict["V_o_tank (m^3)"] = nominal_vessel_params[
            "Nominal Tank Capacity (m^3)"
        ]
        
        parameter_value_dict["m_o_tank (kg)"] = parameter_value_dict["V_o_tank (m^3)"] * rho_f
        
        parameter_value_dict["f_p"] = nominal_vessel_params["Pilot Fuel Tank Capacity (m^3)"] / nominal_vessel_params["Nominal Tank Capacity (m^3)"]
        parameter_value_dict["f_port"] = 1 - nominal_vessel_params["Fraction Year at Sea"]
        parameter_value_dict["N_days"] = calculate_days_to_empty_tank(
            parameter_value_dict["R (nm)"],
            parameter_value_dict["s_av (knots)"],
        )
        parameter_value_dict["boiloff_term"] = calculate_boiloff_term(
            r_f, parameter_value_dict["N_days"], parameter_value_dict["f_port"]
        )
        parameter_value_dict["r_f"] = r_f
        parameter_value_dict["scaling_term_volume"] = calculate_scaling_term(
            parameter_value_dict["V_o_tank (m^3)"],
            rho_l,
            parameter_value_dict["m_c (kg)"],
            parameter_value_dict["V_c (m^3)"],
            cargo_type="volume",
        )
        parameter_value_dict["scaling_term_mass"] = calculate_scaling_term(
            parameter_value_dict["V_o_tank (m^3)"],
            rho_l,
            parameter_value_dict["m_c (kg)"],
            parameter_value_dict["V_c (m^3)"],
            cargo_type="mass",
        )
        parameter_value_dict["fuel_term_volume"] = fuel_term_volume
        parameter_value_dict["fuel_term_mass"] = fuel_term_mass
        parameter_value_dict["cargo_loss_volume"] = calculate_cargo_loss(
            parameter_value_dict["R (nm)"],
            parameter_value_dict["V_o_tank (m^3)"],
            parameter_value_dict["s_av (knots)"],
            r_f,
            e_l,
            L_l,
            rho_l,
            e_f,
            L_f,
            rho_f,
            parameter_value_dict["m_c (kg)"],
            parameter_value_dict["V_c (m^3)"],
            parameter_value_dict["f_p"],
            parameter_value_dict["f_port"],
            cargo_type="volume",
        )
        parameter_value_dict["cargo_loss_mass"] = calculate_cargo_loss(
            parameter_value_dict["R (nm)"],
            parameter_value_dict["V_o_tank (m^3)"],
            parameter_value_dict["s_av (knots)"],
            r_f,
            e_l,
            L_l,
            rho_l,
            e_f,
            L_f,
            rho_f,
            parameter_value_dict["m_c (kg)"],
            parameter_value_dict["V_c (m^3)"],
            parameter_value_dict["f_p"],
            parameter_value_dict["f_port"],
            cargo_type="mass",
        )

        data.append(parameter_value_dict)

        # Set the minimum and maximum values for each parameter over all vessels
        for parameter in minima:
            if parameter == "Vessel":
                continue
            if parameter_value_dict[parameter] < minima[parameter]:
                minima[parameter] = parameter_value_dict[parameter]

            if parameter_value_dict[parameter] > maxima[parameter]:
                maxima[parameter] = parameter_value_dict[parameter]

    data.append(minima)
    data.append(maxima)

    parameter_values_df = pd.DataFrame(data)
    parameter_values_df.to_csv(
        f"{top_dir}/tables/parameter_values_{fuel}.csv", index=False
    )
    print(f"Saved parameter values to {top_dir}/tables/parameter_values_{fuel}.csv")

    return parameter_values_df


def make_mc(
    nominal_vessel_params_df, fuel_params_df, fuel="liquid_hydrogen", N_events=1000
):
    """
    Make Monte Carlo parameter value dictionaries for cargo loss without boil-off, for either mass or volume cargo types.

    Parameters
    ----------

    fuel : str
        Fuel to get parameter values for

    cargo_type : str
        Type of cargo (mass or volume)

    Returns
    -------
    None
    """
    top_dir = get_top_dir()

    # Get a summary of parameter values, along with maxima and minima
    vessel_parameter_values_df = get_parameter_values(
        nominal_vessel_params_df, fuel_params_df, fuel
    )

    fuel_params_lsfo = extract_fuel_params(fuel_params_df, "lsfo")
    fuel_params_fuel = extract_fuel_params(fuel_params_df, fuel)
    r_f = fuel_params_fuel["Boil-off Rate (%/day)"]
    e_l = fuel_params_lsfo["eta_eff_mean"]
    L_l = fuel_params_lsfo["Lower Heating Value (MJ / kg)"]
    rho_l = fuel_params_lsfo["Mass density (kg/m^3)"]
    e_f = fuel_params_fuel["eta_eff_mean"]
    L_f = fuel_params_fuel["Lower Heating Value (MJ / kg)"]
    rho_f = fuel_params_fuel["Mass density (kg/m^3)"]
    fuel_term_mass = calculate_fuel_term(
        L_l, rho_l, e_l, L_f, rho_f, e_f, cargo_type="mass"
    )
    fuel_term_volume = calculate_fuel_term(
        L_l, rho_l, e_l, L_f, rho_f, e_f, cargo_type="volume"
    )

    # Set the relevant MC parameters
    parameters = [
        "R (nm)",
        "s_av (knots)",
        "P_av (MW)",
        "m_c (kg)",
        "V_c (m^3)",
        "f_p",
        "f_port"
    ]
    r_f = fuel_params_fuel["Boil-off Rate (%/day)"]

    # Get the bounds on all parameters
    param_mins = {}
    param_maxes = {}

    Rs, s_avs, P_avs, m_cs, V_cs, f_ps, f_ports = get_cargo_loss_parameter_extrema(
        vessel_parameter_values_df
    )
    param_mins["R (nm)"] = Rs[0]
    param_maxes["R (nm)"] = Rs[1]

    param_mins["s_av (knots)"] = s_avs[0]
    param_maxes["s_av (knots)"] = s_avs[1]

    param_mins["P_av (MW)"] = P_avs[0]
    param_maxes["P_av (MW)"] = P_avs[1]

    param_mins["m_c (kg)"] = m_cs[0]
    param_maxes["m_c (kg)"] = m_cs[1]

    param_mins["V_c (m^3)"] = V_cs[0]
    param_maxes["V_c (m^3)"] = V_cs[1]

    param_mins["f_p"] = f_ps[0]
    param_maxes["f_p"] = f_ps[1]

    param_mins["f_port"] = f_ports[0]
    param_maxes["f_port"] = f_ports[1]

    data = []
    for event in range(N_events):
        parameter_value_dict = {}
        for parameter in parameters:
            parameter_value_dict[parameter] = random.uniform(
                param_mins[parameter], param_maxes[parameter]
            )

        parameter_value_dict["N_days"] = calculate_days_to_empty_tank(
            parameter_value_dict["R (nm)"],
            parameter_value_dict["s_av (knots)"],
        )
        parameter_value_dict["V_o_tank (m^3)"] = parameter_value_dict["R (nm)"] * parameter_value_dict["P_av (MW)"] / (L_l * e_l * rho_l * parameter_value_dict["s_av (knots)"]) * S_PER_H
        parameter_value_dict["m_o_tank (kg)"] = parameter_value_dict["V_o_tank (m^3)"] * rho_f
        #print(parameter_value_dict["V_o_tank (m^3)"])
        parameter_value_dict["scaling_term_volume"] = calculate_scaling_term(
            parameter_value_dict["V_o_tank (m^3)"],
            rho_l,
            parameter_value_dict["m_c (kg)"],
            parameter_value_dict["V_c (m^3)"],
            cargo_type="volume",
        )
        parameter_value_dict["scaling_term_mass"] = calculate_scaling_term(
            parameter_value_dict["V_o_tank (m^3)"],
            rho_l,
            parameter_value_dict["m_c (kg)"],
            parameter_value_dict["V_c (m^3)"],
            cargo_type="mass",
        )
        parameter_value_dict["fuel_term_volume"] = fuel_term_volume
        parameter_value_dict["fuel_term_mass"] = fuel_term_mass

        parameter_value_dict["cargo_loss_volume"] = calculate_cargo_loss(
            parameter_value_dict["R (nm)"],
            parameter_value_dict["V_o_tank (m^3)"],
            parameter_value_dict["s_av (knots)"],
            r_f,
            e_l,
            L_l,
            rho_l,
            e_f,
            L_f,
            rho_f,
            parameter_value_dict["m_c (kg)"],
            parameter_value_dict["V_c (m^3)"],
            parameter_value_dict["f_p"],
            parameter_value_dict["f_port"],
            cargo_type="volume",
        )
        parameter_value_dict["cargo_loss_mass"] = calculate_cargo_loss(
            parameter_value_dict["R (nm)"],
            parameter_value_dict["V_o_tank (m^3)"],
            parameter_value_dict["s_av (knots)"],
            r_f,
            e_l,
            L_l,
            rho_l,
            e_f,
            L_f,
            rho_f,
            parameter_value_dict["m_c (kg)"],
            parameter_value_dict["V_c (m^3)"],
            parameter_value_dict["f_p"],
            parameter_value_dict["f_port"],
            cargo_type="mass",
        )

        data.append(parameter_value_dict)

    parameter_values_mc_df = pd.DataFrame(data)
    parameter_values_mc_df.to_csv(
        f"{top_dir}/tables/mc_parameter_values_{fuel}.csv", index=False
    )

    return parameter_values_mc_df


def plot_cargo_loss_vs_param_with_mc(
    fuel,
    parameter_values_df,
    parameter_values_mc_df,
    params_powers,
    param_name_title,
    param_title_short,
    show_median_line,
    x_lim=None,
):
    """
    Plots the cargo loss as a function of the given parameter, using values for both the defined vessels and MC values

    Parameters
    ----------
    fuel : str
        Name of the fuel to consider

    parameter_values_df : pd.DataFrame
        Dataframe containing parameters related to cargo loss for each vessel, along with maxima and minima

    parameter_values_mc_df : pd.DataFrame
        Dataframe containing parameters related to cargo loss for MC-generated events

    params_powers : dictionary
        Dictionary of the following form:
        {
            param1 (str): power1 (float),
            param2 (str): power2 (float)
        }

    where each param is the name of a model parameter as it's saved in parameter_values_df, and the corresponding power represents the power to raise the param to when multiplied with other params in the dictionary. Eg. the above dictionary would resolve to the following expression: (param1^power1) * (param2^power2).

    param_name_title : str
        Name of the parameter, as it will be shown in the plot

    param_title_short: str
        Short version of the parameter name

    show_median_line : bool
        Boolean parameter to determine whether to show the median line

    x_lim : list of floats
        If x_lim is not None, it should take the form [x_min, x_max]

    Returns
    -------
    None
    """
    # Make copies of the vessel and MC parameter value dfs
    parameter_values_df_cp = parameter_values_df.copy(deep=True)
    parameter_values_mc_df_cp = parameter_values_mc_df.copy(deep=True)

    # Add a new column to parameter_values_df_cp and parameter_values_mc_df_cp with the combined parameter products specified in params_powers
    parameter_values_df_cp["product"] = 1
    parameter_values_mc_df_cp["product"] = 1

    for param in params_powers:
        parameter_values_df_cp["product"] *= (
            parameter_values_df_cp[param] ** params_powers[param]
        )
        parameter_values_mc_df_cp["product"] *= (
            parameter_values_mc_df_cp[param] ** params_powers[param]
        )

    for capacity_type in ["mass", "volume"]:
        fig, ax = plt.subplots(figsize=(9, 6))
        #plt.title(f"{get_fuel_label(fuel)}: {param_title_short}", fontsize=24)

        if x_lim is not None:
            ax.set_xlim(x_lim[0], x_lim[1])
        ax.tick_params(axis="both", which="major", labelsize=20)

        # Use a qualitative colormap for distinct vessel colors
        vessels = [
            vessel
            for vessel in parameter_values_df_cp["Vessel"]
            if vessel not in {"Minimum", "Maximum"}
        ]
        colors = plt.cm.tab10(
            np.linspace(0, 1, len(vessels))
        )  # Use 'tab10' for categorical data
        vessel_color_map = dict(zip(vessels, colors))

        # Bin the MC points into 10 bins and calculate the average cargo loss in each bin
        num_bins = 10
        if x_lim is None:
            bins = np.linspace(
                parameter_values_mc_df_cp["product"].min(),
                parameter_values_mc_df_cp["product"].max(),
                num_bins + 1,
            )
        else:
            bins = np.linspace(max(min(parameter_values_mc_df_cp["product"]), x_lim[0]), x_lim[1], num_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Calculate bin centers

        # Calculate binned averages for volume and mass
        binned_avg = parameter_values_mc_df_cp.groupby(
            np.digitize(parameter_values_mc_df_cp["product"], bins) - 1
        )[f"cargo_loss_{capacity_type}"].median()

        # Plot on the first subplot
        ax.scatter(
            parameter_values_mc_df_cp["product"],
            parameter_values_mc_df_cp[f"cargo_loss_{capacity_type}"],
            label="MC",
            color="black",
            marker="o",
            s=3,
            alpha=0.3,
        )
        for vessel in vessels:
            vessel_data = parameter_values_df_cp[
                parameter_values_df_cp["Vessel"] == vessel
            ]
            ax.scatter(
                vessel_data["product"],
                vessel_data[f"cargo_loss_{capacity_type}"],
                label=vessel,
                color=vessel_color_map[vessel],
                marker="o",
                edgecolors="white",
                linewidth=1,
                s=300,
            )
        # Add the average cargo loss line
        if show_median_line:
            ax.plot(
                bin_centers,
                binned_avg[:num_bins],
                label="Median over MC",
                color="red",
                linestyle="-",
                linewidth=4,
            )
        ax.set_xlabel(param_name_title, fontsize=26)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.xaxis.offsetText.set_fontsize(22)
        if capacity_type == "volume":
            ax.set_ylabel(r"CL$_f^\text{volume}$", fontsize=28)
        else:
            ax.set_ylabel(r"CL$_f^\text{mass}$", fontsize=28)
        ymin, ymax = ax.get_ylim()
        cargo_loss_min = parameter_values_df_cp.loc[
            parameter_values_df_cp["Vessel"] == "Minimum", f"cargo_loss_{capacity_type}"
        ].iloc[0]
        cargo_loss_max = parameter_values_df_cp.loc[
            parameter_values_df_cp["Vessel"] == "Maximum", f"cargo_loss_{capacity_type}"
        ].iloc[0]
        ax.set_ylim(
            cargo_loss_min - 0.5 * (cargo_loss_max - cargo_loss_min),
            max(0.002, cargo_loss_max + 0.5 * (cargo_loss_max - cargo_loss_min))
        )
        param_name_save_plot = ""
        for param in params_powers:
            param_name_save_plot += "_" + param.split(" ")[0] + str(params_powers[param])
        plt.tight_layout()
        plt.savefig(f"plots/cargo_loss_{capacity_type}_vs{param_name_save_plot}_{fuel}.png", dpi=600)
        plt.savefig(f"plots/cargo_loss_{capacity_type}_vs{param_name_save_plot}_{fuel}.pdf")
        print(f"Plot saved to plots/cargo_loss_{capacity_type}_vs{param_name_save_plot}_{fuel}.png and .pdf")
        
        # Add a legend
        fig.set_size_inches(13, 6)
        fig.legend(loc="center left", bbox_to_anchor=(0.655, 0.56), fontsize=20)

        # Adjust layout
        plt.tight_layout(
            rect=[0, 0, 0.68, 1]
        )  # Leave space for the legend on the right
        plt.savefig(f"plots/cargo_loss_{capacity_type}_vs{param_name_save_plot}_{fuel}_with_legend.png", dpi=600)
        plt.savefig(f"plots/cargo_loss_{capacity_type}_vs{param_name_save_plot}_{fuel}_with_legend.pdf")
        print(f"Plot saved to plots/cargo_loss_{capacity_type}_vs{param_name_save_plot}_{fuel}_with_legend.png and .pdf")
        plt.close()
        
        # Create separate figure for the legend
        legend_fig, legend_ax = plt.subplots(figsize=(16, 2))
        legend_ax.axis("off")  # Hide axes

        # Collect all legend entries
        handles, labels = ax.get_legend_handles_labels()

        # Plot legend using as many columns as needed
        n_cols = min(len(handles), 4)  # Adjust as desired
        legend_fig.legend(
            handles,
            labels,
            loc="center",
            ncol=n_cols,
            fontsize=16,
            frameon=False,
        )
        legend_fig.tight_layout()
        legend_path_base = f"plots/cargo_loss_{capacity_type}_vs{param_name_save_plot}_{fuel}_legend"
        legend_fig.savefig(f"{legend_path_base}.png", dpi=600)
        legend_fig.savefig(f"{legend_path_base}.pdf")
        print(f"Legend saved to {legend_path_base}.png and .pdf")
        plt.close(legend_fig)
        

def plot_x_vs_y_with_mc(
    fuel,
    parameter_values_df,
    parameter_values_mc_df,
    x_powers,
    y_powers,
    x_name_title,
    y_name_title,
    x_title_short,
    y_title_short,
    show_best_fit_line,
    color_gradient_param=None,
    color_gradient_title=None,
    x_lim=None
):
    """
    Plots the cargo loss as a function of the given parameter, using values for both the defined vessels and MC values

    Parameters
    ----------
    fuel : str
        Name of the fuel to consider

    parameter_values_df : pd.DataFrame
        Dataframe containing parameters related to cargo loss for each vessel, along with maxima and minima

    parameter_values_mc_df : pd.DataFrame
        Dataframe containing parameters related to cargo loss for MC-generated events

    x_powers, y_powers : dictionaries
        Dictionary of the following form for the x and y parameters:
        {
            param1 (str): power1 (float),
            param2 (str): power2 (float)
        }

    where each param is the name of a model parameter as it's saved in parameter_values_df, and the corresponding power represents the power to raise the param to when multiplied with other params in the dictionary. Eg. the above dictionary would resolve to the following expression: (param1^power1) * (param2^power2).

    x_name_title, y_name_title : strs
        Names of the x and y parameters, as they will be shown in the plot

    x_title_short, y_title_short: str
        Short versions of the x and y parameter names

    show_median_line : bool
        Boolean parameter to determine whether to show the median line

    color_gradient_param : str
        Parameter whose value to use for defining a red-blue color gradient for the color of vessel markers

    color_gradient_title : str
        Name to put on the color gradient for the vessel markers
        
    x_lim : [float, float]
        x-axis plotting limits

    Returns
    -------
    None
    """

    # Make copies of the vessel and MC parameter value dfs
    parameter_values_df_cp = parameter_values_df.copy(deep=True)
    parameter_values_mc_df_cp = parameter_values_mc_df.copy(deep=True)

    # Add a new column to parameter_values_df_cp and parameter_values_mc_df_cp with the combined parameter products specified in params_powers
    parameter_values_df_cp["x_product"] = 1
    parameter_values_mc_df_cp["x_product"] = 1
    parameter_values_df_cp["y_product"] = 1
    parameter_values_mc_df_cp["y_product"] = 1

    for param in x_powers:
        parameter_values_df_cp["x_product"] *= (
            parameter_values_df_cp[param] ** x_powers[param]
        )
        parameter_values_mc_df_cp["x_product"] *= (
            parameter_values_mc_df_cp[param] ** x_powers[param]
        )

    for param in y_powers:
        parameter_values_df_cp["y_product"] *= (
            parameter_values_df_cp[param] ** y_powers[param]
        )
        parameter_values_mc_df_cp["y_product"] *= (
            parameter_values_mc_df_cp[param] ** y_powers[param]
        )

    fig, ax = plt.subplots(figsize=(12, 6))
#    ax.set_title(
#        f"{get_fuel_label(fuel)}: {x_title_short} vs. {y_title_short}", fontsize=24
#    )
    ax.tick_params(axis="both", which="major", labelsize=16)

    # Use a qualitative colormap for distinct vessel colors
    vessels = [
        vessel
        for vessel in parameter_values_df_cp["Vessel"]
        if vessel not in {"Minimum", "Maximum"}
    ]

    if color_gradient_param is None:
        colors = plt.cm.tab10(
            np.linspace(0, 1, len(vessels))
        )  # Use 'tab10' for categorical data
        vessel_color_map = dict(zip(vessels, colors))

    else:
        # Normalize the values of the color_gradient_param for the colormap
        norm = plt.Normalize(
            parameter_values_df_cp[color_gradient_param].min(),
            parameter_values_df_cp[color_gradient_param].max(),
        )
        cmap = plt.cm.coolwarm  # Use a red-blue colormap

        # Create a color map for vessels based on the color_gradient_param
        vessel_color_map = {
            vessel: cmap(
                norm(
                    parameter_values_df_cp.loc[
                        parameter_values_df_cp["Vessel"] == vessel, color_gradient_param
                    ].iloc[0]
                )
            )
            for vessel in vessels
        }

        # Add a colorbar for the gradient
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # ScalarMappable needs a dummy array
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02)
        cbar.set_label(color_gradient_title, fontsize=18)
        cbar.ax.tick_params(labelsize=16)

    # Plot the vessel and MC data as scatterplots
    ax.scatter(
        parameter_values_mc_df_cp["x_product"],
        parameter_values_mc_df_cp["y_product"],
        label="MC",
        color="black",
        marker="o",
        s=3,
        alpha=0.3,
    )
    for vessel in vessels:
        vessel_data = parameter_values_df_cp[parameter_values_df_cp["Vessel"] == vessel]
        ax.scatter(
            vessel_data["x_product"],
            vessel_data["y_product"],
            label=vessel,
            color=vessel_color_map[vessel],
            marker="o",
            edgecolors="white",
            linewidth=1,
            s=300,
        )

    # Add a best fit line
    if show_best_fit_line:
        # Use only the vessel data for regression
        vessel_data_x = parameter_values_df_cp["x_product"]
        vessel_data_y = parameter_values_df_cp["y_product"]

        # Fit a linear regression model to vessel data
        coefficients = np.polyfit(
            vessel_data_x, vessel_data_y, 1
        )  # Linear fit: degree=1
        best_fit_line = np.poly1d(coefficients)  # Create a polynomial function

        # Generate x values for the best fit line
        x_fit = np.linspace(vessel_data_x.min()-0.05*(vessel_data_x.max()-vessel_data_x.min()), vessel_data_x.max()+0.05*(vessel_data_x.max()-vessel_data_x.min()), 500)
        y_fit = best_fit_line(x_fit)

        # Plot the best fit line
        ax.plot(
            x_fit,
            y_fit,
            color="red",
            linestyle="--",
            linewidth=4,
            label="Best Fit Line (Vessels)",
        )

    ax.set_xlabel(x_name_title, fontsize=20)
    ax.set_ylabel(y_name_title, fontsize=20)
    if x_lim:
        ax.set_xlim(x_lim[0], x_lim[1])

    # Place the legend to the right of the plot
    lines_labels = [ax.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc="center right", bbox_to_anchor=(1, 0.5), fontsize=16)

    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0, 0, 0.68, 1])  # Leave space for the legend on the right
    param_name_save_plot = ""
    y_param_keys = list(y_powers.keys())
    param_name_save_plot += y_param_keys[0].split(" ")[0] + str(
        y_powers[y_param_keys[0]]
    )

    if len(y_param_keys) > 1:
        for param in y_powers[1:]:
            param_name_save_plot += "_" + param.split(" ")[0] + str(y_powers[param])

    param_name_save_plot += "_vs"
    for param in x_powers:
        param_name_save_plot += "_" + param.split(" ")[0] + str(x_powers[param])

    plt.savefig(f"plots/{param_name_save_plot}_{fuel}.png", dpi=300)
    plt.savefig(f"plots/{param_name_save_plot}_{fuel}.pdf")
    print(f"Plot saved to plots/{param_name_save_plot}_{fuel}.png and .pdf")
    plt.close()

def get_cargo_loss_parameter_extrema(parameter_values_df):
    """
    Gets the upper and lower extrema of tunable vessel design parameters used to evaluate the fractional cargo loss.

    Parameters
    ----------
    parameter_values_df : pd.DataFrame
        Dataframe containing parameters related to cargo loss for each vessel, along with maxima and minima

    Returns
    -------
    Rs : list of floats
        Extreme values of the vessel design range, in nautical miles

    s_avs : list of floats
        Extreme values of the average speed (knots)

    P_avs : list of floats
        Extreme values of the average main engine power, in MW (MJ/s)

    m_cs : float
        Extreme values of the mass cargo capacity of the vessel, in kg

    V_cs : float
        Extreme values of the volume cargo capacity of the vessel, in m^3
    
    f_ps : float
        Extreme values of the pilot tank to nominal tank size ratio
    """
    def get_bounds(parameter_str):
        param_min = parameter_values_df.loc[
            parameter_values_df["Vessel"] == "Minimum", parameter_str
        ].iloc[0]
        param_max = parameter_values_df.loc[
            parameter_values_df["Vessel"] == "Maximum", parameter_str
        ].iloc[0]
        return [param_min, param_max]

    Rs = get_bounds("R (nm)")
    s_avs = get_bounds("s_av (knots)")
    P_avs = get_bounds("P_av (MW)")
    m_cs = get_bounds("m_c (kg)")
    V_cs = get_bounds("V_c (m^3)")
    f_ps = get_bounds("f_p")
    f_ports = get_bounds("f_port")
    return Rs, s_avs, P_avs, m_cs, V_cs, f_ps, f_ports


def main():
    top_dir = get_top_dir()

    # Collect vessel design and fuel parameters for each vessel+fuel
    nominal_vessel_params_df = collect_nominal_vessel_design_params(top_dir)
    fuel_params_df = collect_fuel_params(top_dir)

    ################ Make plots of cargo loss vs. various parameters ################

    params_powers = {
        r"$f^\text{pilot}$": {"f_p": 1},
        r"$\alpha^\text{volume}$": {
            "V_o_tank (m^3)": 1,
            "V_c (m^3)": -1,
        },
        r"$\alpha^\text{mass}$": {
            "m_o_tank (kg)": 1,
            "m_c (kg)": -1,
        },
        "$N$ (days)": {"N_days": 1},
        r"$\frac{P^\text{av}}{m_o^\text{cargo}}$ (MJ / nautical mile / kg)": {
            "P_av (MW)": 1,
            "m_c (kg)": -1,
        },
    }

    param_titles_short = {
        r"$f^\text{pilot}$": {"f_p": 1},
        r"$\alpha^\text{volume}$": r"$\alpha^\text{volume}$",
        r"$\alpha^\text{mass}$": r"$\alpha^\text{mass}$",
        "$N$ (days)": r"$N$",
        r"$\frac{P^\text{av}}{m_o^\text{cargo}}$ (MJ / nautical mile / kg)": r"$\frac{P^\text{av}}{m_o^\text{cargo}}$ (MJ / nautical mile / kg)",
    }

    show_median_line = {
        r"$f^\text{pilot}$": True,
        r"$\alpha^\text{volume}$": False,
        r"$\alpha^\text{mass}$": False,
        "$N$ (days)": True,
        r"$\frac{P^\text{av}}{m_o^\text{cargo}}$ (MJ / nautical mile / kg)": True
    }
    
    alpha_mass_xlims = {
        "liquid_hydrogen": [-0.0025, 0.01],
        "lng": [-0.005, 0.08]
    }

    for fuel in ["methanol", "ammonia", "liquid_hydrogen", "compressed_hydrogen", "lng"]:
    
        x_lims = {
            r"$f^\text{pilot}$": None,
            r"$\alpha^\text{volume}$": [-0.05, 0.2],
            r"$\alpha^\text{mass}$": alpha_mass_xlims[fuel] if fuel in alpha_mass_xlims else None,
            "$N$ (days)": [0, 350],
            r"$\frac{P^\text{av}}{m_o^\text{cargo}}$ (MJ / nautical mile / kg)": [-0.3e-7, 4e-7]
        }
    
        parameter_values_df = get_parameter_values(
            nominal_vessel_params_df, fuel_params_df, fuel
        )
        
        parameter_values_mc_df = make_mc(
            nominal_vessel_params_df, fuel_params_df, fuel, 50000
        )

        plot_x_vs_y_with_mc(
            fuel,
            parameter_values_df,
            parameter_values_mc_df,
            {"V_o_tank (m^3)": 1},
            {"V_c (m^3)": 1},
            r"$V_o^\text{tank}$ (m$^3$)",
            r"$V_o^\text{cargo}$ (m$^3$)",
            r"$V_o^\text{tank}$",
            r"$V_o^\text{cargo}$",
            show_best_fit_line=True,
            color_gradient_param="cargo_loss_volume",
            color_gradient_title=r"$CL_f^\text{volume}$",
            x_lim = [0, 17000]
        )

        for param_title in params_powers:
            plot_cargo_loss_vs_param_with_mc(
                fuel,
                parameter_values_df,
                parameter_values_mc_df,
                params_powers[param_title],
                param_title,
                param_titles_short[param_title],
                show_median_line=show_median_line[param_title],
                x_lim=x_lims[param_title],
            )

if __name__ == "__main__":
    main()
