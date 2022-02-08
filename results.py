"""
Evaluate the result.csv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option("display.max_rows", 999)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

path = "results.csv"


df2 = pd.read_csv(path,index_col=0).reset_index(drop=True)

df2["optb-sto3g"] = df2["STO-3G_opt_energy"] - df2["STO-3G_energy"]
df2["cc-pvtz-sto3g"] = df2["cc-pvtz_energy"] - df2["STO-3G_energy"]
df2["improvement in %"] = (100 / df2["STO-3G_energy"] * df2["STO-3G_opt_energy"]) - 100
molec = set(df2["molecule"])
improvement = []
meanimprov = []
index2arr = []
indx2 = df2[df2["improvement in %"] < 0].index
df2.loc[indx2, "improvement in %"] = np.nan

for mol in molec:
    indx = df2[df2["molecule"] == mol].index
    indx2 = df2[df2["improvement in %"] > 0].index
    df2.loc[indx2]["improvement in %"] = np.nan

    # print(indx2)
    improvement.append([mol, np.max(df2.loc[indx]["improvement in %"]), np.mean(df2.loc[indx]["improvement in %"])])

improvement = np.asanyarray(improvement)

df3 = pd.DataFrame()
df3["mol"] = improvement[:,0]
df3["best improvement %"] = [float(improvement[i,1]) for i in range(len(improvement[:,1]))]
df3["mean improvement %"] = [float(improvement[i,2]) for i in range(len(improvement[:,2]))]
# index_mal_wieder = df3[np.isnan(df3["best improvement %"]) == True].index
# df3 = df3.fillna(0)
df3 = df3.sort_values("best improvement %", ignore_index = True)
print(df3.columns.values, df2.columns.values)
dfdf = pd.concat([df2,df3], axis= 1)

#print(dfdf)
dfdf.to_csv("plots/results_new.csv", na_rep='NaN',index=False, )



#
# df = pd.read_csv(path,index_col=0).dropna().reset_index(drop=True)
#
# df["optb-sto3g"] = df["STO-3G_opt_energy"] - df["STO-3G_energy"]
# df["cc-pvtz-sto3g"] = df["cc-pvtz_energy"] - df["STO-3G_energy"]
# # ax = df.plot.bar(x='molecule', y='abs. energy diff (sto3g and opt)', rot=0)
# indx = df[df["optb-sto3g"] > 0].index
# df.drop(indx, inplace=True)
# indx2 = df[df["optb-sto3g"] > -1e-3].index
# df.drop(indx2, inplace=True)
#
# #relativ improvement
#
# df["improvement in %"] = (100 / df["STO-3G_energy"] * df["STO-3G_opt_energy"]) - 100
#
# #best config for each molec
# molec = set(df["molecule"])
# improvement = []
# index2arr = []
# for mol in molec:
#     indx = df[df["molecule"] == mol].index
#     indx2 = df[df["improvement in %"] == max(df.loc[indx]["improvement in %"])].index
#     index2arr.append(indx2.tolist()[0])
#     improvement.append([mol, max(df.loc[indx]["improvement in %"])])
#
#
# improvement = np.array(improvement)
# fig, ax = plt.subplots(1,1)
#
# ax.plot(improvement[:,0], improvement[:,1])
# ax.set_ylabel("improvement in %")
# # ax.set_xticks([])
# #ax.set_xticklabels(df.loc[index2arr]["molecule"], rotation='vertical', fontsize=12)
#
# # plt.legend()
# plt.show()


# print(df.columns.values)
def create_plots():
    # fig, ax = plt.subplots(1,1)
    # ax.scatter(df.index,abs(df['STO-3G_energy']), label = 'STO-3G_energy')
    # ax.scatter(df.index,abs(df["cc-pvtz_energy"]), label ="cc-pvtz_energy")
    # ax.scatter(df.index,abs(df['STO-3G_opt_energy']), label ='STO-3G_opt_energy')
    # ax.set_ylabel("energy")
    # ax.set_xticks(df.index)
    # ax.set_xticklabels(df["molecule"], rotation='vertical', fontsize=12)
    # plt.legend()

    molec = set(df["molecule"])
    if not os.path.exists("plots"):
        os.mkdir("plots")
    for mol in molec:
        indx = df[df["molecule"] == mol].index
        xval = df.loc[indx]["learning rate"]
        yval_sto3g = df.loc[indx]['STO-3G_energy']
        yval_sto3g_opt = df.loc[indx]["STO-3G_opt_energy"]
        yval_cc_pvtz = df.loc[indx]["cc-pvtz_energy"]
        f_rtol = df.loc[indx]["f_rtol"]

        fig, (ax1, ax2) = plt.subplots(1, 2)

        fig.suptitle(mol)
        ax1.scatter(xval, yval_sto3g, label = "STO-3G",c = "red")
        ax1.scatter(xval, yval_cc_pvtz, label="cc-pvtz", c = "green")
        ax1.scatter(xval, yval_sto3g_opt, label="STO-3G_opt" ,c = "blue")
        for i in indx:
            ax1.annotate(f_rtol[i],(xval[i], yval_sto3g_opt[i]))
        ax1.set_xscale("log")
        ax1.set_xlabel("learning rate")
        ax1.set_ylabel("energy")
        ax1.set_xticks(xval ,xval)
        ax1.grid(True, which="both", ls="-")
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                   fancybox=True, shadow=True, ncol=1, prop={'size': 8})

        ax2.scatter(xval, yval_sto3g, label = "STO-3G",c = "red")
        ax2.scatter(xval, yval_sto3g_opt, label="STO-3G_opt",c = "blue")
        for i in indx:
            ax2.annotate(f_rtol[i],(xval[i], yval_sto3g_opt[i]))
        ax2.set_xticks(xval, xval)
        ax2.set_xscale("log")
        ax2.set_xlabel("learning rate")
        ax2.grid(True, which="both", ls="-")
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                   fancybox=True, shadow=True, ncol=1, prop={'size': 8})
        fig.tight_layout()

        fig.savefig(f"plots/{mol}.png", dpi = 100)
        plt.close(fig)

        print(f"plots/{mol}.png done")

