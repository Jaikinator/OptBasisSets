"""
create interactive scatter plot of molecules depending on the best optimization
"""
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from optb import loadatomstruc

#load best result from evalutation/best_res.csv
df = pd.read_csv("evaluation/best_res.csv")

df['number of atoms'] = np.ones(len(df))

for i in range(len(df['molecule'])):
    molecule = loadatomstruc(df['molecule'][i])
    df['number of atoms'][i] = len(molecule.atomstruc)

print(df.columns.values)
#make a scatter plot using 'best rel. improvement %', 'mean rel. improvement %' using px


fig1 = px.scatter(df, x="mean rel. improvement %", y="best rel. improvement %",
                 hover_name="molecule", hover_data= df.columns.values
                 ,color='method', size='number of atoms', marginal_x="histogram", marginal_y="histogram")




fig2 = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='aqua',
                align='left'),
    cells=dict(values=[df[col] for col in df.columns],
               fill_color='LawnGreen',
               font=dict(color='black'),
               align='left'))
])



# for i in range(len(df)):
#     plt.scatter(df.loc[i]["best rel. improvement %"], df.loc[i]["mean rel. improvement %"], label=df.loc[i]["molecule"])
#     plt.annotate(df.loc[i]["molecule"], xy=(df.loc[i]["best rel. improvement %"], df.loc[i]["mean rel. improvement %"]))
#
# plt.xlabel("best rel. improvement %")
# plt.ylabel("mean rel. improvement %")
# plt.show()

import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig1),
    dcc.Graph(figure=fig2)
])

if __name__ == '__main__':
    app.run_server(debug=True,host='0.0.0.0', port=1337)

