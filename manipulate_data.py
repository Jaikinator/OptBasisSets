"""
Evaluate the result.csv
"""

import os
import pandas as pd
import numpy as np
from warnings import warn

pd.set_option("display.max_rows", 999)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

path = "results.csv"


df = pd.read_csv(path,index_col=0).reset_index(drop=True)
molec_old = set(df["molecule"])



df["optb-initial"] = df["opt_energy"] - df["initial_energy"]
df["reference-initial"] = df["ref_energy"] - df["initial_energy"]
df["rel. improvement %"] = 100/ df["reference-initial"] * df["optb-initial"]

indx = df[df["optb-initial"] > 0].index
df.drop(indx, inplace=True)
indx2 = df[df["optb-initial"] > -1e-5].index
df.drop(indx2, inplace=True)
indx3 = df[df["best_i"] <+ 10].index
df.drop(indx3, inplace=True)
indx4 = df[df["rel. improvement %"] > 100].index
df.drop(indx4, inplace=True)
indx5 = df[df["rel. improvement %"] < 0].index
df.drop(indx5, inplace=True)

df.reset_index(drop=True, inplace=True)
# reorder columns of df
df = df[['molecule', 'basis', 'ref. basis','learning rate', 'maxiter', 'method', 'f_rtol', 'best_f', 'best_df',
           'best_i', 'opt_energy', 'initial_energy', 'ref_energy', 'optb-initial',
           'reference-initial', 'rel. improvement %']]

df.to_csv("evaluation/results_filtered.csv", index=False)

molec = set(df["molecule"])

print("molecule difference: ", len(molec_old - molec),  molec_old - molec)

if not os.path.exists("evaluation"):
    os.mkdir("evaluation")

if os.path.exists("plots"):
    os.rename("plots", "evaluation/plots")  # move plots folder to evaluation folder

#remove results_filtert.csv to evaluation folder


with open("evaluation/molecules_dropped.txt", "w") as f:
    f.write("molecule dropped because the results where to bad): "
            + str(len(molec_old - molec)) + "\n")
    for i in molec_old - molec:
        f.write(i + "\n")

improvement = []
meanimprov = []
index2arr = []


for mol in molec:
    indx = df[df["molecule"] == mol].index
    indx2 = df[df["rel. improvement %"] > 0].index
    indx2 = indx2.append(df[df["rel. improvement %"] > 100].index)

    df.loc[indx2]["rel. improvement %"] = np.nan

    improvement.append([mol, np.max(df.loc[indx]["rel. improvement %"]),
                        np.mean(df.loc[indx]["rel. improvement %"]),
                        df[df["rel. improvement %"] == np.max(df.loc[indx]["rel. improvement %"])].index.values])

improvement = np.asanyarray(improvement)

df2 = pd.DataFrame()
df2["mol"] = improvement[:,0]
df2["best rel. improvement %"] = [float(improvement[i,1]) for i in range(len(improvement[:,1]))]
df2["mean rel. improvement %"] = [float(improvement[i,2]) for i in range(len(improvement[:,2]))]
df2["index best rel. improvement %"] = [int(improvement[i,3][0]) for i in range(len(improvement[:,2]))]


df_best = pd.DataFrame(df.iloc[[int(improvement[i,3][0]) for i in range(len(improvement[:,2]))]])
df_best.reset_index(drop=False, inplace=True)
dfdf= pd.concat([df2, df_best], axis=1)

index_eq = ((dfdf["index best rel. improvement %"] == dfdf["index"])).all()

if index_eq:
    dfdf.drop(["index best rel. improvement %", "index", "molecule"], axis=1, inplace=True)
else:
    warn("index best rel. improvement % and index are not the same")

dfdf.sort_values("best rel. improvement %", ignore_index = True ,inplace= True)
dfdf.rename(columns={"mol": "molecule"}, inplace=True)

#reorder columns of dfdf
dfdf = dfdf[['molecule', 'basis', 'ref. basis', 'best rel. improvement %', 'mean rel. improvement %',
             'opt_energy', 'initial_energy', 'ref_energy', 'optb-initial', 'reference-initial', 'method',
             'learning rate', 'maxiter', 'f_rtol', 'best_f', 'best_df', 'best_i',]]

dfdf.to_csv("evaluation/best_res.csv", na_rep='NaN',index=False)
