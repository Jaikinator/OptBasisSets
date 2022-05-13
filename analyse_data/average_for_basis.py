"""
Evaluate the result.csv
"""

import os
import time

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from warnings import warn

pd.set_option("display.max_rows", 999)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

path = "results.csv"


df = pd.read_csv(path,index_col=0).reset_index(drop=True)
print(df.columns.values)
molec_old = set(df["molecule"])


df["optb-initial"] = df["opt_energy"] - df["initial_energy"]
df["reference-initial"] = df["ref_energy"] - df["initial_energy"]
df["rel. improvement %"] = 100/ df["reference-initial"] * df["optb-initial"]

indx = df[df["optb-initial"] > 120].index
df.drop(indx, inplace=True)
# indx2 = df[df["optb-initial"] > -1e-5].index
# df.drop(indx2, inplace=True)
indx3 = df[df["best_i"] <+ 10].index
df.drop(indx3, inplace=True)
indx4 = df[df["rel. improvement %"] > 120].index
df.drop(indx4, inplace=True)
# indx5 = df[df["rel. improvement %"] < 0].index
# df.drop(indx5, inplace=True)

df.reset_index(drop=True, inplace=True)





basis_multi_index = pd.MultiIndex.from_frame(df[["basis","ref. basis"]])
basis_var = set(basis_multi_index)

imp_arr = []
mol_arr = []
basis_var_arr = []

for basismut in basis_var:
    ind = df[basis_multi_index == basismut].index
    df_mol = df["molecule"][ind]
    for mol in molec_old:
        index_mol_per_basis_mutation = df_mol[df_mol == mol].index
        imp = df["rel. improvement %"][index_mol_per_basis_mutation].max()
        imp_arr.append(imp)
        basis_var_arr.append(basismut)
        mol_arr.append(mol)

df_basis_variation_average = pd.DataFrame({"molecule": mol_arr,"basis variation":basis_var_arr,"improvement %": imp_arr})
# df_basis_variation_average.to_csv("evaluation/basis_variation_max_per_mol.csv")
# print(df_basis_variation_average)
df = df_basis_variation_average
basis_var = set(df["basis variation"])

mean_imp_arr = []
basis_var_arr = []

for var in basis_var:
    ind_val = df[df["basis variation"] == var].index
    mean_val = df["improvement %"][ind_val].mean()
    mean_imp_arr.append(mean_val)
    basis_var_arr.append(var)

df_mean_basis_var = pd.DataFrame({"basis variation": basis_var_arr,"mean improvement %": mean_imp_arr})
df_mean_basis_var["basis variation"] = df_mean_basis_var["basis variation"].astype(str)
df_mean_basis_var.to_csv("evaluation/mean_basis_var.csv")

print(df_mean_basis_var)

fig = px.bar(df_mean_basis_var, x="basis variation", y="mean improvement %",
             title="Mean improvement per basis variation")
fig.update_layout(font=dict(size=20))

fig.write_html("evaluation/mean_basis_var_not_filtert.html")