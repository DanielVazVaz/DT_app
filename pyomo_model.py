import pyomo.environ as pyo
from functions import get_sharp_splits_with_tuples



def create_pyomo_model(data, pure_comp_set, 
                       F_initial, AH, CUC, HUC):
    #! PREVIOUS CALCULATIONS
    sharp_splits, _ = get_sharp_splits_with_tuples(pure_comp_set)
    thermo_data = data["data1"]
    split_data  = data["data2"]


    feed_columns = []
    for c in thermo_data.index:
        if "".join(thermo_data.loc[c, "Separations"]) == "".join(pure_comp_set):
            feed_columns.append(thermo_data.loc[c, "Columns"])

    #! MODEL
    m = pyo.ConcreteModel()

    #? ====== SETS ====== 
    m.K = pyo.Set(initialize = thermo_data["Columns"].values)
    m.M = pyo.Set(initialize = sharp_splits)
    m.IN = pyo.Set(within= m.M, initialize = [i for i in m.M if len(i)==1])
    m.IM = pyo.Set(within=m.M, initialize = [i for i in sharp_splits if len(i)>1 and len(i) < len(pure_comp_set)])
    m.FS = pyo.Set(within=m.K, initialize = feed_columns)

    m.MM = pyo.SetOf(m.M)
    # Feeding to each column
    dic_FM = {}
    for column in m.K:
        dic_FM[column] = "".join(thermo_data[thermo_data["Columns"] == column]["Separations"].values[0]) 
    m.FM = pyo.Set(within=m.K*m.M, 
                initialize = [(i, dic_FM[i]) for i in dic_FM])
    # Products of each column
    dic_PM = []
    for column in m.K:
        dic_PM.append((column, thermo_data[thermo_data["Columns"] == column]["Separations"].values[0][0]))
        dic_PM.append((column, thermo_data[thermo_data["Columns"] == column]["Separations"].values[0][1]))
    m.PM = pyo.Set(within=m.K*m.M, initialize = dic_PM)
    #? ====== PARAMETERS ======
    m.FT = pyo.Param(initialize = F_initial, 
                    doc = "Total flowrate [kmol/h]")
    m.RQC = pyo.Param(m.K, 
                    initialize = {thermo_data.loc[i, "Columns"]:thermo_data.loc[i, "rQc [GJ/kmol]"] for i in thermo_data.index},
                    doc = "Relative heat of condenser [GJ/kmol]")
    m.RQR = pyo.Param(m.K, 
                    initialize = {thermo_data.loc[i, "Columns"]:thermo_data.loc[i, "rQr [GJ/kmol]"] for i in thermo_data.index},
                    doc = "Relative heat of reboiler [GJ/kmol]")
    m.FC = pyo.Param(m.K,
                    #initialize = {thermo_data[thermo_data["Separations"] == i]["Columns"].values[0]:FC_sep[i] for i in FC_sep},
                    initialize = {thermo_data.loc[i,"Columns"]:thermo_data.loc[i,"Fixed cost [k$/y]"]  for i in thermo_data.index},
                    doc = "Fixed investment cost [k$/y]")

    m.VC = pyo.Param(m.K,
                    #initialize = {thermo_data[thermo_data["Separations"] == i]["Columns"].values[0]:VC_sep[i] for i in VC_sep},
                    initialize = {thermo_data.loc[i,"Columns"]:thermo_data.loc[i,"Variable cost [k$·h/kmol·y]"]  for i in thermo_data.index},
                    doc = "Variable investment cost [k$·h/kmol]")

    m.SF = pyo.Param(m.K, m.M,
                    initialize = {(i,j): split_data.loc[i, j] for i in split_data.index for j in split_data.columns},
                    doc = "Split factor")

    m.AH = pyo.Param(initialize = AH, doc = "Hours operation year [h/y]") #[hours per year]
    m.CUC = pyo.Param(initialize = CUC, doc = "Cost of cooling utility [k$/GJ]") #[k$/GJ]
    m.HUC = pyo.Param(initialize = HUC, doc = "Cost of heating utility [k$/GJ]")  #[k$/GJ]

    #? ====== VARIABLES ======
    m.f = pyo.Var(m.K, domain = pyo.NonNegativeReals, doc = "Feed flowrate to column k [kmol/h]")
    m.ccolumn = pyo.Var(m.K, domain = pyo.NonNegativeReals, doc = "Cost of column k [k$/y]")
    m.y = pyo.Var(m.K, domain = pyo.Binary, doc = "Binary variable if column k exists")
    m.z = pyo.Var(doc = "Objective function")

    #? ====== CONSTRAINTS ======
    def of_eq(m):
        """Objective function equation
        """
        column_cost     = sum(m.ccolumn[k] for k in m.K)
        utilities_cost  = m.AH*sum(m.f[k]*(m.RQC[k]*m.CUC + m.RQR[k]*m.HUC) for k in m.K)
        return m.z == column_cost + utilities_cost
    m.obj = pyo.Constraint(rule = of_eq)

    def eq01(m):
        """Mass balance feed columns
        """
        return sum(m.f[k] for k in m.FS) == m.FT
    m.eq1 = pyo.Constraint(rule = eq01)

    def eq02(m, M):
        """Mass balance other columns
        """
        return sum(m.f[k] for k in m.K if (k,M) in m.FM) == sum(m.f[k]*m.SF[k,M] for k in m.K if (k,M) in m.PM)
    m.eq2 = pyo.Constraint(m.IM, rule = eq02)

    def eq03(m, K):
        """Convex hull for the existance of a column
        """
        return m.f[K] <= m.FT*m.y[K]
    m.eq3 = pyo.Constraint(m.K, rule = eq03)

    def eq04(m, K):
        """ Binary variable constraint convex hull
        """
        return m.ccolumn[K] == m.FC[K]*m.y[K] + m.VC[K]*m.f[K]
    m.eq4 = pyo.Constraint(m.K, rule = eq04)

    #? ====== OBJECTIVE FUNCTION ======
    m.objective_function = pyo.Objective(expr = m.z, sense = pyo.minimize)

    return m

def solve_pyomo_model(m, solver = "gams", gams_path = r"C:\Main_Folder\GAMS\46\gams.exe" ):
    if solver.upper() == "GAMS":
        pyo.pyomo.common.Executable("gams").set_path(gams_path)
        solver = pyo.SolverFactory("gams")
        results = solver.solve(m, tee = True)
    else:
        solver = pyo.SolverFactory(solver)
        results = solver.solve(m, tee = True)
    return results






