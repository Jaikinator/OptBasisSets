"""
Evaluate the result.csv
"""

import os
import time

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from optb import loadatomstruc
from optb.data.avdata import elg2 as g2, elw417 as w417
import warnings
import plotly.offline as py


pd.set_option("display.max_rows", 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def filter_entry(df, **kwargs):
    """
    Filter the results.
    :param df: dataframe with the results
    :param kwargs: keyword arguments.
    :return: pd.DataFrame.
    """
    df = df.copy()
    for key, value in kwargs.items():
        df = df[df[key] == value]

    return df

def normalize_basis_variation(df):
    """
    Normalize the basis variation str.
    :param df: dataframe with the results
    :return: pd.DataFrame.
    """
    df = df.copy()

    for i in range(len(df)):
       df.loc[i,"basis_variation"] = str(df.loc[i,"basis_variation"]).replace("(", "").replace(")", "").replace(" ", "").replace("'", "").replace(",",", ")

    return df

def calculate_impact(df):
    """
    Calculate the impact of the results.
    """
    # create basis variation column
    basis_multi_index = pd.MultiIndex.from_frame(df[["basis", "ref. basis"]]).to_numpy()
    df["basis_variation"] = basis_multi_index
    df = normalize_basis_variation(df) # normalize the basis variation str.
    # create nuber of atoms and db column
    number_of_atoms = np.ones(len(df))
    name_db = []
    for i in range(len(df['molecule'])):
        molecule = loadatomstruc(df['molecule'][i])
        number_of_atoms[i] = molecule.natoms
        name_db.append(molecule.db)
    df['number_of_atoms'] = number_of_atoms
    df["database"] = name_db

    # create energy per atom column
    df["optb_energy [hartree/atom]"] = df["opt_energy"]/ df["number_of_atoms"]
    df['initial_energy [hartree/atom]'] = df["initial_energy"]/ df["number_of_atoms"]
    df["ref_energy [hartree/atom]"] = df["ref_energy"]/ df["number_of_atoms"]

    # create impact column
    df["ref-opb [hartree/atom]"] = df["ref_energy [hartree/atom]"] - df["optb_energy [hartree/atom]"]
    df["ref-init [hartree/atom]"] = df["ref_energy [hartree/atom]"] - df["initial_energy [hartree/atom]"]
    df.reset_index(drop=True, inplace=True)

    return df

def drop_small_best_i(df, threshold = 10):
    """
    Drop results where the optimizer does not do enough steps.
    """
    bvlist_bevor = set(df["basis_variation"])
    df_len = len(df)
    ind = df[df['best_i'] < threshold].index
    df.drop(ind, inplace=True)
    df.reset_index(drop=True, inplace=True)
    bvlist_after = set(df["basis_variation"])
    if len(bvlist_bevor) != len(bvlist_after):
        warnings.warn("basis variation {} has been dropped since best_i < {}.".format(bvlist_bevor - bvlist_after,threshold))

    print(f"\n\tDropped {df_len - len(df)} results.\n")
    return df

def remove_SCF_not_converged(df, threshold = 100, folder = 'data', comment_scfnotconverged:str = "", save = True):
    """
    Remove the SCF not converged results.
    """
    warnings.warn("This remove_SCF_not_converged function is not working properly for specific use-cases.")
    _df = df.copy()

    if "basis_variation" not in _df.columns:
        _df = calculate_impact(_df)
        warnings.warn("Basis variation not in dataframe. Calculated.", RuntimeWarning)

    for basisv in set(df["basis_variation"]):

        ind = _df[_df["basis_variation"] == basisv].index
        _df_mol = _df.copy().iloc[ind]

        molec_old = set(_df_mol["molecule"])
        # drop result where the reference scf calculation is not converged
        ind_not_converged = _df_mol[abs(_df_mol["ref_energy"]/ _df_mol["initial_energy"]) > threshold].index

        _df_mol.drop(ind_not_converged, inplace=True) # just to check which molecules are dropped
        _df_mol.reset_index(drop=True, inplace=True)

        _df.drop(ind_not_converged, inplace=True)  # actual drop the results in the main dataframe
        _df.reset_index(drop=True, inplace=True)

        molec_drop_scf_not_conv = set(_df_mol["molecule"])

        # drop results where the optimized basis set is not converged in the scf calculation

        ind = _df[_df["basis_variation"] == basisv].index
        _df_mol_2 = _df.copy().iloc[ind]

        ind_not_converged_optb = _df_mol_2[abs(_df_mol_2["opt_energy"]/ _df_mol_2["ref_energy"]) > threshold].index

        _df_mol_2.drop(ind_not_converged_optb, inplace=True) # just to check which molecules are dropped because of optb energy too high
        _df_mol_2.reset_index(drop=True, inplace=True)

        molec_drop_scf_optb_not_conv = set(_df_mol_2["molecule"])

        _df.drop(ind_not_converged_optb, inplace=True) # actual drop the results in the main dataframe
        _df.reset_index(drop=True, inplace=True)

        print(f"For basis variation: {basisv}\n\t in total removed {len(molec_old - molec_drop_scf_optb_not_conv)} molecules.")
        print(f"\t removed because the scf of a given basis wasn't converged "
              f"{len(molec_old - molec_drop_scf_not_conv)} molecules.")
        print(f"\t removed because the scf of a optimised basis wasn't converged "
              f"{len(molec_drop_scf_not_conv - molec_drop_scf_optb_not_conv)} molecules.")

        if len(molec_old - molec_drop_scf_optb_not_conv) >  0 and save:
            basisvstr = str(basisv).replace("(", "").replace(")", "").replace(" ", "").replace("'", "")
            print(f"\n\tThe molecules that are not converged for {basisvstr} are "
                  f"written in {folder}/molecules_dropped_{basisvstr}_{comment_scfnotconverged}.txt\n")

            with open(f"{folder}/molecules_dropped_{basisvstr}_{comment_scfnotconverged}.txt", "w") as f:
                f.write(f"molecule of the basis variation {basisvstr} dropped because the results "
                        f"of pyscf are not converged for large basis): "
                        + str(len(molec_old - molec_drop_scf_not_conv)) + "\n")

                for i in molec_old - molec_drop_scf_not_conv:
                    f.write(i + "\n")

                f.write(f"molecule of the basis variation {basisvstr} dropped because the results "
                        f"of pyscf are not converged for optimised basis set): "
                        + str(len(molec_drop_scf_not_conv - molec_drop_scf_optb_not_conv)) + "\n")

                for i in molec_drop_scf_not_conv - molec_drop_scf_optb_not_conv:
                    f.write(i + "\n")

    return _df


def select_db(df,db):
    """
    Choose the database to use.
    :param df: dataframe with the results
    :param db: str - the name of the database to use.
    :return: pd.DataFrame - molecule from the choosen db.
    """

    if db == "g2":
        _df= df[df['database'] == "g2"]
        _df.reset_index(drop=True, inplace=True)
        return _df
    elif db == "w417":
        _df = df[df['database'] == "w417"]
        _df.reset_index(drop=True, inplace=True)
        return _df
    elif db == "all":
        return df
    else:
        raise ValueError("The database must be local or remote")

def drop_wrong_learning(df):
    """
    Drop the results where the learning is not correct and the result where diverged.
    """

    drop_to_big = df[df["ref-opb [hartree/atom]"] < df["ref-init [hartree/atom]"]].index
    print("len df old", len(df))

    df = df.drop(drop_to_big)
    df.reset_index(drop=True, inplace=True)
    print("len df new", len(df))
    return df

def get_best_res_mol(df, db ="w417", folder = "data", comment_scfnotconverged:str = ""):
    """
    Get the best result for each molecule and basis variation.
    """
    _df = df.copy()
    _df = calculate_impact(_df)
    _df = select_db(_df, db)
    _df = drop_small_best_i(_df, threshold= 10 )
    _df = remove_SCF_not_converged(_df, threshold=10 , folder=folder , comment_scfnotconverged=comment_scfnotconverged)
    _df = drop_wrong_learning(_df)

    out_df = pd.DataFrame(columns = _df.columns.values)


    for basisv in set(_df["basis_variation"]):
        ind = _df[_df["basis_variation"] == basisv].index
        df_mol = _df["molecule"][ind]

        for num,mol in enumerate(set(df_mol)):

            index_mol_per_basis_mutation = df_mol[df_mol == mol].index

            imp = _df["ref-opb [hartree/atom]"][index_mol_per_basis_mutation].idxmin()

            out_df.loc[len(out_df)] = _df.iloc[imp]

    return out_df

def get_average_impact_bv(df ,abs = True):
    mean_imp_arr = []
    mean_total_arr = []
    basis_var_arr = []

    for var in set(df["basis_variation"]):
        ind_val = df[df["basis_variation"] == var].index
        mean_val = df["ref-opb [hartree/atom]"][ind_val].mean()
        mean_total_avg = df["ref-init [hartree/atom]"][ind_val].mean()
        mean_imp_arr.append(mean_val)
        mean_total_arr.append(mean_total_avg)
        basis_var_arr.append(var)

    df_mean_basis_var = pd.DataFrame({"basis_variation": basis_var_arr,
                                      "mean energy difference [hartree/atom]": mean_imp_arr ,
                                      "ref - init energy difference [hartree/atom]":mean_total_arr})
    df_mean_basis_var["basis_variation"] = df_mean_basis_var["basis_variation"].astype(str)

    # calculate the absolute difference for each basis variation

    df_mean_basis_var["absolute difference [hartree/atom]"] = \
        df_mean_basis_var["ref - init energy difference [hartree/atom]"] - df_mean_basis_var["mean energy difference [hartree/atom]"]

    if abs:

        df_mean_basis_var["mean energy difference [hartree/atom]"] =  \
            df_mean_basis_var["mean energy difference [hartree/atom]"].abs()

        df_mean_basis_var["ref - init energy difference [hartree/atom]"] =  \
            df_mean_basis_var["ref - init energy difference [hartree/atom]"].abs()

        df_mean_basis_var["absolute difference [hartree/atom]"] =  \
            df_mean_basis_var["absolute difference [hartree/atom]"].abs()

    return df_mean_basis_var

def reorder_df(df1, df2):
    """
    Reorder the dataframe df1 according to the dataframe df2.
    :param df1: dataframe with the results
    :param df2: dataframe with the results
    :param key: str - the key to use to reorder the dataframe.
    :return: pd.DataFrame.
    """
    df1 = df1.copy()
    df2 = df2.copy()

    diff = [item for item in df2.columns.values if item not in df1.columns.values]

    for i in diff:
        df1[i] = np.nan

    df1 = df1[df2.columns.values]
    std_headers = ['molecule', 'file number', 'initial_energy', 'ref_energy', 'opt_energy', 'learning rate', 'maxiter',
                   'miniter', 'method', 'best_f', 'best_df', 'best_dcnorm', 'best_i', 'max_i', 'f_rtol', 'basis',
                   'ref. basis', 'basis_variation', 'number_of_atoms', 'database', 'optb_energy [hartree/atom]',
                   'initial_energy [hartree/atom]', 'ref_energy [hartree/atom]', 'ref-opb [hartree/atom]', 'ref-init [hartree/atom]']

    if len(df1.columns.values) == len(df2.columns.values)\
        and len(df1.columns.values) == len(std_headers)\
        and len(df2.columns.values) == len(std_headers):

        if all(df1.columns.values == std_headers) and all(df2.columns.values == std_headers):
            new_order = ['molecule','number_of_atoms', 'database', 'file number', 'basis', 'ref. basis', 'basis_variation',

                         'learning rate', 'maxiter','miniter', 'method', 'best_f', 'best_df', 'best_dcnorm',
                         'best_i', 'max_i', 'f_rtol',

                         'initial_energy', 'ref_energy', 'opt_energy',
                         'initial_energy [hartree/atom]', 'ref_energy [hartree/atom]','optb_energy [hartree/atom]',
                         'ref-opb [hartree/atom]', 'ref-init [hartree/atom]']
            df1 = df1[new_order]
            df2 = df2[new_order]

            return df1, df2

    else:
        warnings.warn("The dataframe headers are not the standard ones. Only df1 will be changed.")
        return df1



####
#standardize the dataframe
####

std_order = ['molecule','number_of_atoms', 'database', 'file number', 'basis', 'ref. basis', 'basis_variation',
             'learning rate', 'maxiter','miniter', 'method', 'best_f', 'best_df', 'best_dcnorm',
             'best_i', 'max_i', 'f_rtol',
             'initial_energy', 'ref_energy', 'opt_energy',
             'initial_energy [hartree/atom]', 'ref_energy [hartree/atom]','optb_energy [hartree/atom]',
             'ref-opb [hartree/atom]', 'ref-init [hartree/atom]']

#
#
# #####
# #data one
# #####
#
# path_one = "data/results_old.csv"
#
# df_one = pd.read_csv(path_one,index_col=0).reset_index(drop=True)
#
# #####
# #data two
# #####
#
# path_two = "data/results.csv"
#
# df_two = pd.read_csv(path_two,index_col=0).reset_index(drop=True)
#
# #############################
#
# df_one = reorder_df(df_one, df_two)
# df_one_impact = calculate_impact(df_one)
#
#
# df_one_best = get_best_res_mol(df_one, db = "w417", folder = "data", comment_scfnotconverged="old")
# df_one_best = df_one_best[std_order]
# df_one_best.to_csv("data/best_results_old_w417.csv", index=False)
#
# df_one_best = get_best_res_mol(df_one, db = "g2", folder = "data", comment_scfnotconverged="old")
# df_one_best = df_one_best[std_order]
# df_one_best.sort_values(by=["basis_variation", "number_of_atoms"], inplace=True)
# df_one_best.to_csv("data/best_results_old_g2.csv", index=False)
#
#
# df_two_impact = calculate_impact(df_two)
#
# df_one_impact ,df_two_impact = reorder_df(df_one_impact,df_two_impact)
#
# df_one_impact.sort_values(by=["basis_variation", "number_of_atoms"], inplace=True)
# df_two_impact.sort_values(by=["basis_variation", "number_of_atoms"], inplace=True)
#
# df_two_impact.to_csv("data/results_new_impact.csv", index=False)
# df_one_impact.to_csv("data/results_old_impact.csv", index=False)
#
# df_two_best = get_best_res_mol(df_two, db = "w417", folder = "data", comment_scfnotconverged="new")
# df_two_best = df_two_best[std_order]
# df_two_best.sort_values(by=["basis_variation", "number_of_atoms"], inplace=True)
# df_two_best.to_csv("data/best_results_new_w417.csv", index=False)


#############################
# read the best results to calculate the average improvement in the energy
#############################

####
# for all results merge the dataframes
####

df_old = pd.read_csv("data/best_results_old_w417.csv")
df_new = pd.read_csv("data/best_results_new_w417.csv")

df_full = pd.concat([df_new, df_old]).reset_index(drop=True)
df_full_mean = get_average_impact_bv(df_full)

####
# for single molecules
####

df_old_imp = pd.read_csv("data/results_old_impact.csv")
df_new_imp = pd.read_csv("data/results_new_impact.csv")

df_imp_full = pd.concat([df_new_imp, df_old_imp]).reset_index(drop=True)


# fig = px.bar(df, x="basis_variation", y="mean energy difference [hartree/atom]",color="ref - init energy difference [hartree/atom]",barmode='group',
#              title="Average impact of basis variation")
# fig.update_layout(
#     #xaxis_showticklabels=False,
#     title_text="Average impact of basis variation",
#     xaxis_title="Basis variation",
#     yaxis_title="Average energy difference [Hartree/atom]",
#     font=dict(
#         family="Courier New, monospace",
#         size=18,
#         color="#000066"
#     )
# )





def make_plots_basis_var(df):
    df = normalize_basis_variation(df_full_mean)
    print(df)
    df.sort_values(by=["absolute difference [hartree/atom]"], ascending=False, inplace=True)
    fig = go.Figure(data=[
        go.Bar(name='mean energy difference between the reference and the optimised basis', x=df["basis_variation"],
               y=df["mean energy difference [hartree/atom]"]),
        go.Bar(name='mean energy difference between two basis sets', x=df["basis_variation"],
               y=df["ref - init energy difference [hartree/atom]"])
    ])
    # Change the bar mode
    fig.update_layout(barmode='group',
                      title_text="Average impact of optimisation",
                      xaxis_title="Basis variation",
                      yaxis_title="Energy difference [Hartree/atom]",
                      font=dict(
                          family="Courier New, monospace",
                          size=18,
                          color="#000066"
                      ),
                      legend=dict(
                          x=1.0,
                          xanchor="right",
                          y=1.0,
                          traceorder="normal",
                          font=dict(
                              family="Courier New, monospace",
                              size=18,
                              color="#000066"
                          ),
                          bgcolor="LightSteelBlue",
                          bordercolor="Black",
                          borderwidth=2
                      )
                      )
    # fig.write_html("data/avg_energy_bv.html")
    fig.show()

    #############################
    # absolute difference in the energy
    #############################

    fig = px.bar(df, x="basis_variation", y="absolute difference [hartree/atom]",
                 title="Absolute difference in the energy")

    fig.update_layout(
        # xaxis_showticklabels=False,
        title_text="Absolute difference in the energy",
        xaxis_title="Basis variation",
        yaxis_title="Absolute difference [Hartree/atom]",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#000066"
        )
    )
    fig.show()

def plots_molecule(df_mol, molecule, **kwargs):

    df = filter_entry(df_mol, molecule = molecule, **kwargs)

    df = drop_wrong_learning(df)

    df = drop_small_best_i(df, threshold=10)

    df = remove_SCF_not_converged(df, threshold=10)

    df = normalize_basis_variation(df)
    blist = list(set(df["basis_variation"]))
    print(blist)
    fig = make_subplots(rows=2, cols=2,  # subplotting grid hardcoded
                        shared_xaxes = False,
                        shared_yaxes= True,
                        horizontal_spacing=0.01,
                        vertical_spacing=0.1,
                        subplot_titles=blist,
                        specs=[[{"type": "scatter", "l" : 0.01}, {"type": "scatter"}],
                                 [{"type": "scatter", "l" : 0.01}, {"type": "scatter"}]],
                        x_title="Learning rate",
                        y_title="Energy difference [Hartree/atom]"
                        )
    fig.update_annotations(font_size=18)


    row = 1
    col = 1
    for num,b in enumerate(blist):
        df_b = df.loc[df["basis_variation"] == b]

        if num == 0:
            show_legend = True
        else:
            show_legend = False

        if num%2 == 0:
            grid = {"row" : row , "col" : col}
            col += 1
        elif num%2 == 1:
            grid = {"row" : row , "col" : col}
            col = 1
            row += 1

        fig.append_trace(
            go.Scatter(mode="lines", x=df_b["learning rate"], y=df_b["initial_energy [hartree/atom]"],
                       name="initial energy", marker=dict(color="blue", size=20), line = dict(color="blue", width=2),
                       legendgroup="initial energy", showlegend=show_legend,connectgaps=True),
            **grid)
        fig.append_trace(
            go.Scatter(mode="lines", x=df_b["learning rate"], y=df_b["ref_energy [hartree/atom]"],
                       name="reference energy", marker=dict(color="red", size=20), line=dict(color="red", width=2),
                       legendgroup="reference energy", showlegend=show_legend),
            **grid)
        fig.append_trace(
            go.Scatter(mode="markers+text", x=df_b["learning rate"], y=df_b["optb_energy [hartree/atom]"],
                       name="optimised energy", marker=dict(color="green", size=20),
                       legendgroup="optimised energy", showlegend=show_legend),
            **grid)

        fig.update_xaxes(type= "log",exponentformat = "power" ,
                        showline=True, linewidth=1, linecolor='black',mirror=True,**grid)

        fig.update_yaxes(title=None, tickfont = {"size": 15, "color": "#000066" , "family": "Courier New, monospace"},
                         showline=True, linewidth=1, linecolor='black',mirror=True, **grid)


    fig.update_layout(
            title_text="Energy of molecule {}".format(molecule),

            legend=dict(
                x=1.0,
                xanchor="right",
                y=1.0,
                traceorder="normal"),
            font=dict(
                family="Droid Sans",
                size=18,
                color="#000066"
            )
        )
    #fig.show()

def scatter_all_molecules(df_mol, **kwargs):

    fig1 = px.scatter(df_mol, x="ref-opb [hartree/atom]", y="best_f",
                      hover_name="molecule", hover_data=df_mol.columns.values
                      , color='method', size='number_of_atoms', marginal_x="histogram", marginal_y="histogram")
    fig1.update_layout(
        title_text="Energy difference between reference and optimised basis set",
        xaxis_title="Energy difference [Hartree/atom]",
        # yaxis_title="Number of atoms",
        font=dict(
            family="Droid Sans",
            size=18,
            color="#000066"
        ),
        legend=dict(
            x=1.0,
            xanchor="right",
            y=1.0,
            traceorder="normal",
            font=dict(
                family="Droid Sans",
                size=18,
                color="#000066"
            ),
            bgcolor="LightSteelBlue",
            bordercolor="Black",
            borderwidth=2
        )
    )
    fig1.show()



# plots_molecule(df_imp_full, "h2" , database = "w417" ,method = "adam")
print(sorted(list(set(normalize_basis_variation(df_imp_full)["basis_variation"]))))
















































































#### check again what I done here
# df_new = df_full[df_full["basis_variation"] == "('STO-3G', '3-21G')"]
# df_new["diff of diff [hartree/atom]"] = df_new["ref-opb [hartree/atom]"] - df_new["ref-init [hartree/atom]"]
# print(df_new)
# fig1 = px.histogram(df_new, x="diff of diff [hartree/atom]", nbins=50)
# fig1.show()


#plotly bar plot with two y axis

# fig1 = px.scatter(df_new, x="ref-opb [hartree/atom]", y="ref-init [hartree/atom]",
#                  hover_name="molecule", hover_data= df_new.columns.values
#                  ,color='method', size='number_of_atoms', marginal_x="histogram", marginal_y="histogram")
# fig1.show()










# df = get_average_impact_bv(df)
#
# fig = px.bar(df, x="basis_variation", y="mean energy difference [hartree/atom]",color="ref - init energy difference [hartree/atom]",barmode='group',
#              title="Average impact of basis variation")
# fig.update_layout(
#     #xaxis_showticklabels=False,
#     title_text="Average impact of basis variation",
#     xaxis_title="Basis variation",
#     yaxis_title="Average energy difference [Hartree/atom]",
#     font=dict(
#         family="Courier New, monospace",
#         size=18,
#         color="#000066"
#     )
# )
# print(df)
#
# fig = go.Figure(data=[
#     go.Bar(name='mean energy difference between the reference and the optimised basis', x=df["basis_variation"], y=df["mean energy difference [hartree/atom]"]),
#     go.Bar(name='mean energy difference between two basis sets', x=df["basis_variation"], y=df["ref - init energy difference [hartree/atom]"])
# ])
# # Change the bar mode
# fig.update_layout(barmode='group',
#                   title_text="Average impact of optimisation",
#                   xaxis_title="Basis variation",
#                   yaxis_title="Energy difference [Hartree/atom]",
#                   font=dict(
#                       family="Courier New, monospace",
#                       size=18,
#                       color="#000066"
#                   )
#                   )
# fig.write_html("evaluation/avg_energy_bv.html")