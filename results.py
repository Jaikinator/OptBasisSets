"""
Evaluate the result.csv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


pd.set_option("display.max_rows", 999)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

path = "results.csv"

df = pd.read_csv(path,index_col=0).dropna().reset_index(drop=True)

df["optb-sto3g"] = df["STO-3G_opt_energy"] - df["STO-3G_energy"]
df["cc-pvtz-sto3g"] = df["cc-pvtz_energy"] - df["STO-3G_energy"]
# ax = df.plot.bar(x='molecule', y='abs. energy diff (sto3g and opt)', rot=0)
indx = df[df["optb-sto3g"] > 0].index
df.drop(indx, inplace=True)

# fig, ax = plt.subplots(1,1)
# ax.scatter(df.index,abs(df["optb-sto3g"]), label = "optb-sto3g")
# ax.scatter(df.index,abs(df["cc-pvtz-sto3g"]), label ="cc-pvtz-sto3g", marker = "s")
# ax.set_ylabel("energy diff.")
# ax.set_xticks(df.index)
# ax.set_xticklabels(df["molecule"], rotation='vertical', fontsize=12)
# plt.legend()

print(df.columns.values)

# fig, ax = plt.subplots(1,1)
# ax.scatter(df.index,abs(df['STO-3G_energy']), label = 'STO-3G_energy')
# ax.scatter(df.index,abs(df["cc-pvtz_energy"]), label ="cc-pvtz_energy")
# ax.scatter(df.index,abs(df['STO-3G_opt_energy']), label ='STO-3G_opt_energy')
# ax.set_ylabel("energy")
# ax.set_xticks(df.index)
# ax.set_xticklabels(df["molecule"], rotation='vertical', fontsize=12)
# plt.legend()
#

plt.show()

molec = set(df["molecule"])
if not os.path.exists("plots"):
    os.mkdir("plots")
for mol in molec:
    indx = df[df["molecule"] == mol].index
    xval = df.loc[indx]["learning rate"]
    yval_sto3g = df.loc[indx]['STO-3G_energy']
    yval_sto3g_opt = df.loc[indx]["STO-3G_opt_energy"]
    yval_cc_pvtz = df.loc[indx]["cc-pvtz_energy"]

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.suptitle(mol)
    ax1.scatter(xval, yval_sto3g, label = "STO-3G")
    ax1.scatter(xval, yval_cc_pvtz, label="cc-pvtz")
    ax1.scatter(xval, yval_sto3g_opt, label="STO-3G_opt")
    ax1.set_xscale("log")
    ax1.set_xlabel("learning rate")
    ax1.set_ylabel("energy")
    ax1.grid(True, which="both", ls="-")
    ax1.grid()

    ax2.scatter(xval, yval_sto3g, label = "STO-3G")
    ax2.scatter(xval, yval_sto3g_opt, label="STO-3G_opt")
    ax2.set_xscale("log")
    ax2.set_xlabel("learning rate")
    ax2.grid(True, which="both", ls="-")
    fig.tight_layout()
    fig.legend()


    fig.savefig(f"plots/{mol}.png", dpi = 500)
    plt.close(fig)

    print(f"plots/{mol}.png done")