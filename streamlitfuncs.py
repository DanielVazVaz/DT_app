import pandas as pd
import numpy as np
from pyomo_model import create_pyomo_model, solve_pyomo_model
from functions import create_data, get_sharp_splits_with_tuples, Antoine_eq, dH_eq
import pyvis 
import networkx as nx


def color_survived(val):
    color = 'red' if pd(val) else 'yellow' if val==1 else 'green'
    return f'background-color: {color}'

if __name__ == "__main__":
    pure_comp_set  = ["A", "B", "C", "D"]
    x_initial      = np.array([0.10, 0.30, 0.40, 0.20])
    F_initial      = 120

    AH = 8000 #[hours per year]
    CUC = 0.15/1000 #[k$/GJ]
    HUC = 3.9/1000  #[k$/GJ]

    A_par = {"A": (67.2281, -5420.3, 0, 0, -8.8253, 9.6171E-06, 2),
            "B": (93.1371, -6995.5, 0, 0, -12.702, 1.2381E-05, 2),
            "C": (76.3161, -6996.4, 0, 0, -9.8802, 7.2099E-06, 2),
            "D": (84.5711, -7900.2, 0, 0, -11.003, 7.1802E-06, 2)}

    H_par = {"A": (37.01, 0.4121, -0.1238, 469.6),
            "B": (43.85, 0.397, -0.039, 507.4),
            "C": (53.66, 0.2831, 0.2831, 540.2),
            "D": (58.46, 0.3324, 0.1834, 568.8)}

    #? ====== SOLVER ======
    TP = 1
    data = create_data(pure_comp_set, x_initial, F_initial, A_par, H_par,
                    TP, save_excel = False)
    m = create_pyomo_model(data, pure_comp_set, F_initial, AH, CUC, HUC)
    results = solve_pyomo_model(m)

    # g = pyvis.network.Network(notebook = True)
    # nxg = nx.complete_graph(10)
    # g.from_nx(nxg)
    # g.show("example.html", notebook = False)


