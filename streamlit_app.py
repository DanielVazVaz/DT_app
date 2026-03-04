import numpy as np 
from pyomo_model import create_pyomo_model, solve_pyomo_model
from functions import create_data, Antoine_eq, dH_eq, create_graph_from_results
import pandas as pd
import os
import streamlit as st
import streamlit.components.v1 as components
abecedary = "ABCDEFGHIJKLM"
FOLDER_HTML = "networks"


st.set_page_config(
    page_title="Sharp-distillation optimization",
    page_icon="🌍",
    layout="wide",
) 


with st.expander("🗒️ About", expanded=False):
    st.write("""
    This is a web app to optimize sharp distillation columns.  
    The app is built using Python, Streamlit, and Pyomo. 
    """)




st.markdown("# Component information")

col1, col2 = st.columns(2)
n_components_selector = col1.selectbox("Number of components", [2, 3, 4, 5, 6, 7], 
                                    index=2, placeholder="Select number of components...")
col2.write(f"The components, from more to less volatile, are:  \n  :red[{', '.join(abecedary[:n_components_selector])}]")

tab1, tab2= st.tabs(["Antoine parameters", "dHvap parameters"])
with tab1:
    st.markdown("### Antoine parameters")
    st.markdown("$\log(P^{vap}) = A - \\frac{B}{T + C} + DT + E\log(T) + FT^G \qquad P^{vap}[bar]\quad T[K]$")
    antoine_params_list = "ABCDEFG"
    antoine_df = pd.DataFrame(index = [i for i in abecedary[:n_components_selector]], columns = [i for i in antoine_params_list])
    default_data =  [[67.2281,	-5420.3,	0,	0,	-8.8253,	9.6171E-6,	2],
                    [93.1371,	-6995.5,	0,	0,	-12.702,	1.2381E-5,	2],
                    [76.3161,	-6996.4,	0,	0,	-9.8802,	7.2099E-6,	2],
                    [84.5711,	-7900.2,	0,	0,	-11.003,	7.1802E-6,	2]]

    antoine_df = pd.DataFrame(index = [i for i in abecedary[:n_components_selector]], columns = [i for i in antoine_params_list])
    # WE fill the initial data regarding the number of components
    try:
        for i in range(n_components_selector):
            antoine_df.loc[abecedary[i],:] = default_data[i]
    except IndexError:
        pass
    antoine_df.index.name = "Chemical"
    st_antoine = st.data_editor(antoine_df, width = 30000, num_rows = "fixed")
    antoine_pvap_df = pd.DataFrame(index = [i for i in abecedary[:n_components_selector]], columns = ["Pvap (25 ºC)"])
    for i in range(n_components_selector):
        try:
            pvap_25 = Antoine_eq(*[float(st_antoine.loc[abecedary[i],j]) for j in antoine_params_list], 25 + 273)
            pvap_25 = np.exp(pvap_25)
        except TypeError:
            pvap_25 = np.nan
        except ValueError:
            pvap_25 = np.nan   

        antoine_pvap_df.loc[abecedary[i], "Pvap (25 ºC)"] = pvap_25
    
    st.table(antoine_pvap_df.style.map(lambda cell: 'color: red' if np.isnan(cell) else 'color:green'))
    if all(antoine_pvap_df["Pvap (25 ºC)"].notna()):
        st.markdown("✔️ :green[All parameters are filled]")
    
with tab2:
    st.markdown("### $\Delta H_{vap}$ parameters")
    st.markdown(r"$\Delta H_{vap} = A(1-\frac{T}{T_c})^B \exp(\frac{-CT}{T_c}) \qquad \Delta H_{vap}[kJ/mol]\quad T[K]$")
    dvap_param_list = ["A", "B", "C", "Tc"]
    dvap_df = pd.DataFrame(index = [i for i in abecedary[:n_components_selector]], columns = [i for i in dvap_param_list])
    default_data = [[37.01,	0.4121,	-0.1238,	469.6],
                    [43.85,	0.397,	-0.039,	507.4],
                    [53.66,	0.2831,	0.2831,	540.2],
                    [58.46,	0.3324,	0.1834,	568.8]]
        # WE fill the initial data regarding the number of components
    try:
        for i in range(n_components_selector):
            dvap_df.loc[abecedary[i],:] = default_data[i]
    except IndexError:
        pass
    dvap_df.index.name = "Chemical"
    st_dvap = st.data_editor(dvap_df, width = 30000, num_rows = "fixed")
    dvap_25_df = pd.DataFrame(index = [i for i in abecedary[:n_components_selector]], columns = ["dHvap (25 ºC)"])
    for i in range(n_components_selector):
        try:
            dVap_25 = dH_eq(*[float(st_dvap.loc[abecedary[i],j]) for j in dvap_param_list], 25 + 273)
        except TypeError:
            dVap_25 = np.nan
        except ValueError:
            dVap_25 = np.nan
        dvap_25_df.loc[abecedary[i], "dHvap (25 ºC)"] = dVap_25
    st.table(dvap_25_df.style.map(lambda cell: 'color: red' if np.isnan(cell) else 'color:green'))
    if all(dvap_25_df["dHvap (25 ºC)"].notna()):
        st.markdown("✔️ :green[All parameters are filled]")


with st.form("Data"):
    # Every form must have a submit button.
    st.markdown("# Feeding information")
    st.markdown("### Feed flowrate")
    st.write("Enter the total flowrate of the feed stream [kmol/h]")
    F_initial_text = st.text_input("Feed flowrate [kmol/h]: ", value=120)
    st.write("Enter the initial composition of the feed stream as a numpy array, e.g.. [0.2, 0.3, 0.5]")
    x_initial_text = st.text_input("Feed composition: ", value = "[0.10, 0.30, 0.40, 0.20]")
    st.write("Enter the total pressure of the system [bar]")
    TP_text = st.text_input("Total pressure [bar]: ", value = 1)



    submitted = st.form_submit_button("Submit")
    if submitted:
        pure_comp_set  = abecedary[:n_components_selector]
        #x_initial      = np.array([0.10, 0.30, 0.40, 0.2])
        #F_initial      = 120
        #TP = 5
        
        F_initial = float(F_initial_text)
        x_initial = np.asarray(np.matrix(x_initial_text)).flatten()
        TP = float(TP_text)
        
        A_par = {i: tuple(st_antoine.loc[i,:].astype(float).to_list()) for i in abecedary[:n_components_selector]}
        H_par = {i: tuple(st_dvap.loc[i,:].astype(float).to_list()) for i in abecedary[:n_components_selector]}
        results = create_data(pure_comp_set, x_initial,F_initial, A_par, H_par, TP, save_excel = False)
        #print(results["data1"])

        # Dataframe of calculated data
        st.markdown("### Calculated data")
        st.write(results["data1"])
        m = create_pyomo_model(results, pure_comp_set, F_initial, 8000, 0.15/1000, 3.9/1000)
        sol = solve_pyomo_model(m, solver = "glpk")

        # Results
        st.markdown("### Results")
        st.write(f"The total annual cost is: {m.z.value:.2f} k$/year")

        ## Graph
        #Get the current path of the app 
        current_path = os.path.dirname(os.path.abspath(__file__))
        # set the current working directory to the current path of the app
        os.chdir(current_path)
        print("CURRENT PATH IS: ", current_path)
        create_graph_from_results(m, html_folder = FOLDER_HTML)
        print("HTML file created in: ", os.path.join(FOLDER_HTML,"net.html"))
        HtmlFile = open(os.path.join(FOLDER_HTML,"net.html"), 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, height = 600)