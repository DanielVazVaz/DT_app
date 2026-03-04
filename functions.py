import numpy as np 
import pandas as pd
from scipy.optimize import fsolve
import pyvis
import os


def create_graph_from_results(m, html_folder):
    """
    Creates a directed graph from distillation process results using pyvis.
    
    Parameters:
    m (object): Model with attributes:
        - K: List of column IDs.
        - y: Dict of column values.
        - FM: List of feed stream tuples (column, stream).
        - PM: List of product stream tuples (column, stream).
    
    Steps:
    1. Select columns with y >= 0.99.
    2. Create feed and product stream dictionaries.
    3. Generate column labels.
    4. Identify all involved streams.
    5. Create pyvis network graph with nodes and edges.
    6. Save graph as HTML.
    
    Returns:
    None
    """
    chosen_columns = [i for i in m.K if m.y[i].value >= 0.99]
    feed_streams = {k:[] for k in chosen_columns}
    product_streams = {k:[] for k in chosen_columns}
    for (k,i) in m.FM:
        if k in chosen_columns:
            feed_streams[k].append(i)
    for (k,i) in m.PM:
        if k in chosen_columns:
            product_streams[k].append(i)
    column_labels = {k: f"{k}\n---\n{product_streams[k]}" for k in product_streams}
    streams_involved = [j for i in feed_streams.values() for j in i] + [j for i in product_streams.values() for j in i]

    # We get the larger streams and the smalles one. This is the one with more letters in streams_involved
    largest_stream = max(streams_involved, key = lambda x: len(x))
    

    G = pyvis.network.Network(height="600px", width="100%", directed=True, notebook=False)
    G.add_nodes([i for i in  streams_involved], 
                color = ["#7e6e9942" if i==largest_stream else "#bedabfff" if i not in ["A", "B", "C", "D"] else "#86bcd1" for i in streams_involved],
                shape = ["database" if i==largest_stream else "ellipse" for i in streams_involved])
    G.add_nodes([i for i in column_labels], 
                label = [column_labels[i] for i in column_labels], 
                shape = ["box" for i in column_labels], 
                color = ["#edb2c4" for i in column_labels], 
                )
    # Edges from streams to columns
    G.add_edges([(feed_streams[k][0],k) for k in feed_streams])
    # Edges from columns to streams
    G.add_edges([(k,i) for k in product_streams for i in product_streams[k]])
    G.write_html(os.path.join(html_folder,"net.html"), notebook=False)


def get_sharp_splits_with_tuples(components):
    """
    Generate all possible sharp splits and tuple splits for a given list of components.
    A sharp split is a contiguous subset of the components list represented as a string.
    A tuple split is a pair of sharp splits that partition the list into two contiguous segments.
    Parameters:
    components (list of str): A list of components to generate splits from.
    Returns:
    tuple: A tuple containing two elements:
        - sharp_splits (list of str): A list of all possible sharp splits.
        - tuple_splits (list of tuple): A list of tuples, where each tuple contains two sharp splits
          that partition the list into two contiguous segments.
    """
    sharp_splits = []
    n = len(components)
    
    # First, collect all individual sharp splits as strings
    for start in range(n):
        for end in range(start + 1, n + 1):
            subset = ''.join(components[start:end])
            sharp_splits.append(subset)
    
    # Now, collect tuples of sharp splits that partition the list into two contiguous segments
    tuple_splits = []
    for split_point in range(1, n):  # The split point must be between 1 and n-1 to ensure valid partitions
        left_part = components[:split_point]
        right_part = components[split_point:]
        
        # Generate all possible contiguous sharp splits for the left and right parts
        left_splits = [''.join(left_part[start:]) for start in range(len(left_part))]
        right_splits = [''.join(right_part[:end]) for end in range(1, len(right_part) + 1)]
        
        # Combine each left split with each right split
        for left in left_splits:
            for right in right_splits:
                tuple_splits.append((left, right))
    
    return sharp_splits, tuple_splits

def get_sharp_splits(components):
    """
    Generate all possible contiguous sublists (sharp splits) from a list of components.
    Args:
        components (list): A list of components from which to generate sharp splits.
    Returns:
        list: A list of lists, where each sublist is a contiguous subset of the input components.
    """
    splits = []
    n = len(components)
    
    # Iterate through all possible starting points
    for start in range(n):
        # Iterate through all possible ending points
        for end in range(start + 1, n + 1):
            subset = components[start:end]
            splits.append(subset)
    
    return splits


def Antoine_eq(A,B,C,D,E,F,G,T):
    """
    Calculate the Antoine equation for vapor pressure.
    The Antoine equation is a mathematical expression used to calculate the vapor pressure of a pure substance as a function of temperature.
    Parameters:
    A (float): Antoine coefficient A
    B (float): Antoine coefficient B
    C (float): Antoine coefficient C
    D (float): Antoine coefficient D
    E (float): Antoine coefficient E
    F (float): Antoine coefficient F
    G (float): Antoine coefficient G
    T (float): Temperature at which to calculate the vapor pressure
    Returns:
    float: Calculated vapor pressure at temperature T
    """
    return A + B/(C + T) + D*T + E*np.log(T) + F*T**G

def fsolve_Antoine_eq(T, A,B,C,D,E,F,G,P):
    return Antoine_eq(A,B,C,D,E,F,G,T) - np.log(P)    

def fsolve_Antoine_mix(T, comps, comps_molar_frac, A_par, P = 1, position = "cond"):
    """The basis is that for the condenser: Ptotal = sum(i, Pvap(i,T)) = sum(i, xi*Pvap0(i,T)) for all components    
       While for the revolver: Ptotal*sum(zi/Pivap)=1"""    
    summation = 0
    ncomp = {"A": 0, "B": 1, "C": 2, "D":3, "E":4, "F":5, "G":6}  # To get the order automatically
    if position == "cond":
        for comp in comps:
            summation += comps_molar_frac[ncomp[comp]]*np.exp(Antoine_eq(*A_par[comp], T))
        return P - summation
    elif position == "reb":
        for comp in comps:
            summation += comps_molar_frac[ncomp[comp]]/np.exp(Antoine_eq(*A_par[comp], T))
        return 1 - P*summation
    
def dH_eq(A, B, C, Tc, T):    
    """
    Calculate the enthalpy of vaporization (dH) using the given equation.

    Parameters:
    A (float): Coefficient A in the equation.
    B (float): Coefficient B in the equation.
    C (float): Coefficient C in the equation.
    Tc (float): Critical temperature.
    T (float): Temperature at which the enthalpy change is calculated.

    Returns:
    float: The calculated enthalpy change (dH).
    """
    return A*(1-T/Tc)**B*np.exp(-C*T/Tc)

def dH_eq_mix(comps, comps_molar_frac, T, H_par):
    """
    Calculate the enthalpy change of a mixture at equilibrium.

    Parameters:
    comps (list of str): List of component names (e.g., ["A", "B", "C", "D"]).
    comps_molar_frac (list of float): List of molar fractions for each component.
    T (float): Temperature at which the enthalpy change is calculated.
    H_par (dict): Dictionary containing enthalpy parameters for each component.

    Returns:
    float: The enthalpy change of the mixture at equilibrium.
    """
    summation = 0
    ncomp = {"A": 0, "B": 1, "C": 2, "D":3, "E":4, "F":5, "G":6}  # To get the order automatically
    for comp in comps:
        summation += comps_molar_frac[ncomp[comp]]*dH_eq(*H_par[comp], T)
    # for ncomp, comp in enumerate(comps):
    #     summation += comps_molar_frac[ncomp]*dH_eq(*H_par[comp], T)
    return summation

def fsolve_Underwood(theta, alpha_list, molar_frac_list, q):
    """
    Solves the Underwood equation for a given theta.

    The Underwood equation is used in distillation calculations to find the 
    minimum reflux ratio and other parameters. This function calculates the 
    summation term in the Underwood equation and returns the result.

    Parameters:
    theta (float): The variable to solve for in the Underwood equation.
    alpha_list (list of float): List of relative volatilities of the components.
    molar_frac_list (list of float): List of molar fractions of the components.
    q (float): The feed quality, which is the fraction of the feed that is liquid.

    Returns:
    float: The result of the Underwood equation for the given theta.
    """
    summation = 0
    for n,i in enumerate(alpha_list):
        summation += molar_frac_list[n]*i/(i - theta)
    return 1 - q - summation

def Fenske_eq(xlkD, xhkD, xlkB, xhkB, alpha):
    """
    Calculate the minimum number of theoretical stages required for a distillation column using the Fenske equation.
    Parameters:
    xlkD (float): Mole fraction of the light key in the distillate.
    xhkD (float): Mole fraction of the heavy key in the distillate.
    xlkB (float): Mole fraction of the light key in the bottoms.
    xhkB (float): Mole fraction of the heavy key in the bottoms.
    alpha (float): Relative volatility of the light key to the heavy key.
    Returns:
    float: Minimum number of theoretical stages.
    """
    xlkD = max(xlkD, 1e-5)
    xhkD = max(xhkD, 1e-5)
    xlkB = max(xlkB, 1e-5)
    xhkB = max(xhkB, 1e-5)

    return np.log((xlkD/xhkD)*(xhkB/xlkB))/np.log(alpha)

def diameter_approximation(D, RR, vapor_velocity = 3):
    """
    Approximate the diameter of a distillation column.

    Parameters:
    D (float): Distillate flow rate in kmol/h.
    RR (float): Reflux ratio.
    vapor_velocity (float, optional): Vapor velocity in m/s. Default is 3 m/s.

    Returns:
    float: Estimated diameter of the distillation column in meters.
    """
    Vflow = D*(1+RR) # kmol/h
    Vflow = Vflow*1000*22.4/1000/3600 # m3/s
    A = Vflow/vapor_velocity # m2
    Diameter = np.sqrt(4*A/np.pi)
    return Diameter

def condenser_area(Q, Tcond, U = 900, Tcoolant = 298, dTcoolant = 5):
    """
    Calculate the area of a condenser required for a given heat duty.

    Parameters:
    Q (float): Heat duty in watts (W).
    Tcond (float): Condensation temperature in Kelvin (K).
    U (float, optional): Overall heat transfer coefficient in W/m²K. Default is 900 W/m²K.
    Tcoolant (float, optional): Inlet temperature of the coolant in Kelvin (K). Default is 298 K.
    dTcoolant (float, optional): Temperature rise of the coolant in Kelvin (K). Default is 5 K.

    Returns:
    float: Required condenser area in square meters (m²).
    """
    # Assumption of U in W/m2K
    dT1 = Tcond - Tcoolant
    dT2 = Tcond - Tcoolant - dTcoolant
    LMTD = (dT1 - dT2)/np.log(dT1/dT2)
    A = Q/(U*LMTD)
    return A

def reboiler_area(Q, Treb, U = 900, Tsteam = 523.15):
    """
    Calculate the reboiler area required for a given heat duty.

    Parameters:
    Q (float): Heat duty in watts (W).
    Treb (float): Reboiler temperature in Kelvin (K).
    U (float, optional): Overall heat transfer coefficient in W/m²K. Default is 900 W/m²K.
    Tsteam (float, optional): Steam temperature in Kelvin (K). Default is 523.15 K.

    Returns:
    float: Required reboiler area in square meters (m²).
    """
    # Since neither the steam nor the liquid change T, but only phase, the LMTD is the same as the temperature difference
    # Assumption of U in W/m2K
    LMTD = Tsteam - Treb
    A = Q/(U*LMTD)
    return A

def calculate_vessel_cost(D, H, W = 0.05, material_density = 7850):
    """
    Calculate the cost of a cylindrical vessel based on its dimensions and material density.
    Parameters:
    D (float): Diameter of the vessel in meters.
    H (float): Height of the vessel in meters.
    W (float, optional): Wall thickness of the vessel in meters. Default is 0.05 meters.
    material_density (float, optional): Density of the material in kg/m^3. Default is 7850 kg/m^3.
    Returns:
    float: The estimated cost of the vessel.
    """
    V = np.pi*D**2/4*H - np.pi*(D-2*W)**2/4*(H-2*W)
    M = V*material_density
    Cost = 11600+34*(M**0.85)
    return Cost 

def calculate_tray_cost(N, D):
    """
    Calculate the cost of trays in a distillation column.

    Parameters:
    N (int): Number of trays.
    D (float): Diameter of the column.

    Returns:
    float: Total cost of the trays.
    """
    return N*(130+440*D**1.8)

def calculate_exchanger_cost(A):
    """
    Calculate the cost of a heat exchanger based on its area.

    This function estimates the cost of a heat exchanger using a formula
    that takes into account the area of the heat exchanger.

    Parameters:
    A (float): The area of the heat exchanger in square meters.

    Returns:
    float: The estimated cost of the heat exchanger in dollars.
    """
    return 28000+54*(A**1.2)




def create_data(pure_comp_set, x_initial, F_initial, A_par, H_par, TP, save_excel=False):
    """
    Generates data for distillation column design based on given parameters.
    Parameters:
    pure_comp_set (list): List of pure components in the mixture.
    x_initial (array-like): Initial mole fractions of the components.
    F_initial (float): Initial feed flow rate.
    A_par (dict): Antoine equation parameters for each component.
    H_par (dict): Enthalpy parameters for each component.
    TP (float): Operating pressure of the distillation column.
    save_excel (bool, optional): If True, saves the generated data to an Excel file. Default is False.
    Returns:
    dict: A dictionary containing two DataFrames:
        - "data1": DataFrame with detailed distillation column data.
        - "data2": DataFrame with flow distribution data.
    """
    sharp_splits, separation_set = get_sharp_splits_with_tuples(pure_comp_set)
    sharp_splits_no_feed = [i for i in sharp_splits if len(i) < len(pure_comp_set)] 
    Fx_initial = x_initial*F_initial
    column_set =  ["k" + str(i+1) for i in range(len(separation_set))]
    df = pd.DataFrame()
    df["Columns"] =  column_set
    df["Separations"] = separation_set
    df["F"]   = 0
    df["F"]   = df["F"].astype(object)
    df["x_f"] = 0
    df["x_f"] = df["x_f"].astype(object)
    df["F_heads"] = 0
    df["F_heads"] = df["F_heads"].astype(object)
    df["x_heads"] = 0
    df["x_heads"] = df["x_heads"].astype(object)
    df["x_bottoms"] = 0
    df["x_bottoms"] = df["x_bottoms"].astype(object)
    for column in df.index:
        present_elements_heads   = [i for i in pure_comp_set if i in df.loc[column, "Separations"][0]]
        present_elements_bottoms = [i for i in pure_comp_set if i in df.loc[column, "Separations"][1]]
        list_f          = []
        list_heads      = []
        list_bottoms    = []
        for n,comp in enumerate(pure_comp_set):
            if comp in present_elements_heads or comp in present_elements_bottoms:
                list_f.append(Fx_initial[n])
            else:
                list_f.append(0) 
            if comp in present_elements_heads:
                list_heads.append(Fx_initial[n])
            else:
                list_heads.append(0)
            if comp in present_elements_bottoms:
                list_bottoms.append(Fx_initial[n])
            else:
                list_bottoms.append(0)
        x_f = np.array(list_f)
        x_f /= x_f.sum()
        x_head = np.array(list_heads)
        x_head /= x_head.sum()
        x_bottoms = np.array(list_bottoms)
        x_bottoms /= x_bottoms.sum()
        df.at[column, "F"]   = list_f
        df.at[column, "x_f"] = x_f
        df.at[column, "F_heads"] = list_heads
        df.at[column, "x_heads"] = x_head
        df.at[column, "x_bottoms"] = x_bottoms
        df.loc[column, "D [kmol/h]"] = sum(list_heads)
        df.loc[column, "B [kmol/h]"] = sum(list_f) - sum(list_heads)
        
    for row in df.index:
        df.loc[row, "Tcond [K]"] = fsolve(fsolve_Antoine_mix, args = ([i for i in df.loc[row, "Separations"][0]],
                                                                    df.loc[row, "x_heads"],A_par, TP, "cond"), x0 = 348)
        df.loc[row, "Treb [K]"] = fsolve(fsolve_Antoine_mix, args = ([i for i in df.loc[row, "Separations"][1]],
                                                                    df.loc[row, "x_bottoms"],A_par, TP, "reb"), x0 = 380)
        df.loc[row, "dHvap_head [MJ/kmol]"] =  dH_eq_mix(comps = [i for i in df.loc[row, "Separations"][0]], comps_molar_frac = df.loc[row, "x_heads"], T = df.loc[row, "Tcond [K]"], H_par=H_par)
        df.loc[row, "dHvap_bots [MJ/kmol]"] =  dH_eq_mix(comps = [i for i in df.loc[row, "Separations"][1]], comps_molar_frac = df.loc[row, "x_bottoms"], T = df.loc[row, "Treb [K]"], H_par=H_par)

    df["alpha [i/HK]"] = 0
    df["alpha [i/HK]"] = df["alpha [i/HK]"].astype(object)
    for row in df.index:
        # Alpha [A/HK, B/HK, C/HK, D/HK]
        HK = df.loc[row, "Separations"][1][0]
        alpha_list = []
        for comp in pure_comp_set:
            Pi_cond = np.exp(Antoine_eq(*A_par[comp], df.loc[row, "Tcond [K]"]))
            Pi_reb  = np.exp(Antoine_eq(*A_par[comp], df.loc[row, "Treb [K]"]))
            Pj_cond = np.exp(Antoine_eq(*A_par[HK], df.loc[row, "Tcond [K]"]))
            Pj_reb  = np.exp(Antoine_eq(*A_par[HK], df.loc[row, "Treb [K]"]))
            alpha_i_j = np.sqrt(Pi_cond*Pi_reb/Pj_cond/Pj_reb)
            alpha_list.append(alpha_i_j)
        df.at[row, "alpha [i/HK]"] = alpha_list
        index_HK = pure_comp_set.index(HK)
        index_LK = index_HK - 1
        df.loc[row, "alpha [LK/HK]"] = alpha_list[index_LK]
        df.loc[row, "Nmin"] = Fenske_eq(df.loc[row, "x_heads"][index_LK], 
                                        df.loc[row, "x_heads"][index_HK], 
                                        df.loc[row, "x_bottoms"][index_LK], 
                                        df.loc[row, "x_bottoms"][index_HK], 
                                        df.loc[row, "alpha [LK/HK]"])
        df.loc[row, "N"] = np.ceil((df.loc[row, "Nmin"]*2))
        df.loc[row, "H [m]"] = np.ceil((df.loc[row, "Nmin"]*2))*1 # Assuming 1 m per stage
        df.loc[row, "theta"] = fsolve(fsolve_Underwood, args = (df.loc[row, "alpha [i/HK]"], df.loc[row, "x_f"], 1), x0 = 1.01)
        df.loc[row, "RR"]  = 1.2*-fsolve_Underwood(theta = df.loc[row, "theta"], alpha_list = df.loc[row, "alpha [i/HK]"], molar_frac_list= df.loc[row, "x_heads"], q = 0)
        df.loc[row, "Diameter [m]"] = diameter_approximation(D = df.loc[row, "D [kmol/h]"], RR = df.loc[row, "RR"], vapor_velocity = 1.5)
        df.loc[row, "Qc [MJ/h]"] = np.array(df.loc[row, "F_heads"]).sum()*(1 + df.loc[row, "RR"])*df.loc[row, "dHvap_head [MJ/kmol]"]
        df.loc[row, "Qr [MJ/h]"] = np.array(df.loc[row, "F_heads"]).sum()*(1 + df.loc[row, "RR"])*df.loc[row, "dHvap_bots [MJ/kmol]"]
        df.loc[row, "rQc [GJ/kmol]"] = df.loc[row, "Qc [MJ/h]"]/1000/np.array(df.loc[row, "F"]).sum()
        df.loc[row, "rQr [GJ/kmol]"] = df.loc[row, "Qr [MJ/h]"]/1000/np.array(df.loc[row, "F"]).sum()
        df.loc[row, "DTC [K]"] = df.loc[row, "Treb [K]"] - df.loc[row, "Tcond [K]"]

        # Condenser and reboiler area calculations
        df.loc[row, "Acond [m2]"] = condenser_area(Q = df.loc[row, "Qc [MJ/h]"]*1e6/3600, Tcond = df.loc[row, "Tcond [K]"], U = 900, Tcoolant = 298, dTcoolant = 5)
        df.loc[row, "Areb [m2]"] = reboiler_area(Q = df.loc[row, "Qr [MJ/h]"]*1e6/3600, Treb = df.loc[row, "Treb [K]"], U = 900, Tsteam = 523.15)

        # Vessel and tray cost calculations
        df.loc[row, "Cost_vessel [k$]"] = calculate_vessel_cost(D = df.loc[row, "Diameter [m]"], H = df.loc[row, "H [m]"], W = 0.05, material_density = 7850)/1000
        df.loc[row, "Cost_tray [k$]"] = calculate_tray_cost(N = df.loc[row, "N"], D = df.loc[row, "Diameter [m]"])/1000

        # Exchanger cost calculations
        df.loc[row, "Cost_condenser [k$]"] = calculate_exchanger_cost(A = df.loc[row, "Acond [m2]"])/1000
        df.loc[row, "Cost_reboiler [k$]"] = calculate_exchanger_cost(A = df.loc[row, "Areb [m2]"])/1000

        # We consider that the vessel and tray are fixed costs and the exchangers are variable costs. We consider annualization of 5 years
        df.loc[row, "Fixed cost [k$/y]"] = (df.loc[row, "Cost_vessel [k$]"] + df.loc[row, "Cost_tray [k$]"])/5
        df.loc[row, "Variable cost [k$·h/kmol·y]"] = (df.loc[row, "Cost_condenser [k$]"] + df.loc[row, "Cost_reboiler [k$]"])/5/np.array(df.loc[row, "F"]).sum()

        # Second dataframe:
        df2 = pd.DataFrame(index =column_set, columns = sharp_splits_no_feed)
        Fi   = F_initial*x_initial 
        ncomp = {pure_comp_set[i]: i for i in range(len(pure_comp_set))}
        for row in df2.index:
            D_comp = df[df["Columns"] == row]["Separations"].values[0][0]
            B_comp = df[df["Columns"] == row]["Separations"].values[0][1]
            list_comp = D_comp + B_comp
            total_flow = np.array([Fi[ncomp[n]] for n in list_comp]).sum()
            for column in df2.columns:
                if D_comp == column:
                    D_flow = np.array([Fi[ncomp[n]] for n in D_comp]).sum()
                    df2.loc[row, column] = D_flow/total_flow
                elif B_comp == column:
                    B_flow = np.array([Fi[ncomp[n]] for n in B_comp]).sum()
                    df2.loc[row, column] = B_flow/total_flow
                else:
                    df2.loc[row, column] = 0
    if save_excel:
        with pd.ExcelWriter(f"data_{TP}bar.xlsx") as writer:
            df.to_excel(writer, sheet_name="data1")
            df2.to_excel(writer, sheet_name="data2")
    return {"data1": df, "data2": df2}


if __name__ == "__main__":
    A_par = {"A": (67.2281, -5420.3, 0, 0, -8.8253, 9.6171E-06, 2),
         "B": (93.1371, -6995.5, 0, 0, -12.702, 1.2381E-05, 2),
         "C": (76.3161, -6996.4, 0, 0, -9.8802, 7.2099E-06, 2),
         "D": (84.5711, -7900.2, 0, 0, -11.003, 7.1802E-06, 2)}
    
    H_par = {"A": (37.01, 0.4121, -0.1238, 469.6),
         "B": (43.85, 0.397, -0.039, 507.4),
         "C": (53.66, 0.2831, 0.2831, 540.2),
         "D": (58.46, 0.3324, 0.1834, 568.8)}
    
    column_set     = ["k" + str(i+1) for i in range(10)]
    pure_comp_set  = ["A", "B", "C", "D"]
    #separation_set = [("A", "BCD"), ("AB", "CD"), ("ABC", "D"), ("B", "CD"), ("BC", "D"), ("A", "BC"), ("AB", "C"), ("C", "D"), ("B", "C"), ("A", "B")]
    _, separation_set = get_sharp_splits_with_tuples(pure_comp_set)
    x_initial      = np.array([0.10, 0.30, 0.40, 0.20])
    #x_initial      = np.array([0.10, 0.30, 0.60])
    F_initial      = 120
    TP = 1
    data = create_data(pure_comp_set, x_initial, F_initial, A_par, H_par, TP, save_excel = True)