"""
create interactive scatter plot of molecules depending on the best optimization
"""
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import plotly.express as px
from optb import loadatomstruc

#load best result from evalutation/best_res.csv
df = pd.read_csv("evaluation/best_res.csv")

df['number of atoms'] = np.ones(len(df))

for i in range(len(df['molecule'])):
    molecule = loadatomstruc(df['molecule'][i])
    df['number of atoms'][i] = len(molecule.atomstruc)

print(df.columns.values)
#make a scatter plot using 'best rel. improvement %', 'mean rel. improvement %' using px
fig = px.scatter(df, x="mean rel. improvement %", y="best rel. improvement %",
                 hover_name="molecule", hover_data=["best rel. improvement %", "mean rel. improvement %",'opt_energy',
                                                    'initial_energy', 'ref_energy', 'basis', 'ref. basis']
                 ,color='method', size='number of atoms')

fig.show()




# for i in range(len(df)):
#     plt.scatter(df.loc[i]["best rel. improvement %"], df.loc[i]["mean rel. improvement %"], label=df.loc[i]["molecule"])
#     plt.annotate(df.loc[i]["molecule"], xy=(df.loc[i]["best rel. improvement %"], df.loc[i]["mean rel. improvement %"]))
#
# plt.xlabel("best rel. improvement %")
# plt.ylabel("mean rel. improvement %")
# plt.show()

