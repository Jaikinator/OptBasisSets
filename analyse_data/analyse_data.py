"""
Evaluate the result.csv
"""

import os
import time

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from optb import loadatomstruc
from optb.data.avdata import elg2 as g2, elw417 as w417
import warnings


pd.set_option("display.max_rows", 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def calculate_impact(df):
    """
    Calculate the impact of the results.
    """
    # create basis variation column
    basis_multi_index = pd.MultiIndex.from_frame(df[["basis", "ref. basis"]]).to_numpy()
    df["basis_variation"] = basis_multi_index

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
    df["optb_energy [hartree /atom]"] = df["opt_energy"] / df["number_of_atoms"]
    df['initial_energy [hartree/atom]'] = df["initial_energy"] / df["number_of_atoms"]
    df["ref_energy [hartree/atom]"] = df["ref_energy"] / df["number_of_atoms"]

    # create impact column
    df["ref-opb [hartree /atom]"] = df["ref_energy [hartree/atom]"] - df["optb_energy [hartree /atom]"]
    df["ref-init [hartree /atom]"] = df["ref_energy [hartree/atom]"] - df["initial_energy [hartree/atom]"]
    df.reset_index(drop=True, inplace=True)

    return df

def drop_small_best_i(df, threshold = 10):
    """
    Drop results where the optimizer does not do enough steps.
    """
    df_len = len(df)
    ind = df[df['best_i'] < threshold].index
    df.drop(ind, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"\n\tDropped {df_len - len(df)} results.\n")
    return df

def remove_SCF_not_converged(df, threshold = 100):
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
        ind_not_converged = _df_mol[abs(_df_mol["ref_energy"] / _df_mol["initial_energy"]) > threshold].index

        _df_mol.drop(ind_not_converged, inplace=True) # just to check which molecules are dropped
        _df_mol.reset_index(drop=True, inplace=True)

        _df.drop(ind_not_converged, inplace=True)  # actual drop the results in the main dataframe
        _df.reset_index(drop=True, inplace=True)

        molec_drop_scf_not_conv = set(_df_mol["molecule"])

        # drop results where the optimized basis set is not converged in the scf calculation

        ind = _df[_df["basis_variation"] == basisv].index
        _df_mol_2 = _df.copy().iloc[ind]

        ind_not_converged_optb = _df_mol_2[abs(_df_mol_2["opt_energy"] / _df_mol_2["ref_energy"]) > threshold].index

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

        if len(molec_old - molec_drop_scf_optb_not_conv) > 0:
            print(f"\n\tThe molecules that are not converged for {basisv} are "
                  f"written in evaluation/molecules_dropped_{basisv}.txt\n")
            with open(f"evaluation/molecules_dropped_{basisv}.txt", "w") as f:
                f.write(f"molecule of the basis variation {basisv} dropped because the results "
                        f"of pyscf are not converged for large basis): "
                        + str(len(molec_old - molec_drop_scf_not_conv)) + "\n")

                for i in molec_old - molec_drop_scf_not_conv:
                    f.write(i + "\n")

                f.write(f"molecule of the basis variation {basisv} dropped because the results "
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
    else:
        raise ValueError("The database must be local or remote")

def drop_wrong_learning(df):
    drop_to_big = df[df["ref-opb [hartree /atom]"] < df["ref-init [hartree /atom]"]].index
    print("len df old", len(df))

    df = df.drop(drop_to_big)
    df.reset_index(drop=True, inplace=True)
    print("len df new", len(df))
    return df
def get_best_res_mol(df, db ="w417"):
    """
    Get the best result for each molecule and basis variation.
    """
    _df = df.copy()
    _df = calculate_impact(_df)
    _df = select_db(_df, db)
    _df = drop_small_best_i(_df, threshold= 10)
    _df = remove_SCF_not_converged(_df, threshold=10)
    _df = drop_wrong_learning(_df)

    out_df = pd.DataFrame(columns = _df.columns.values)

    for basisv in set(_df["basis_variation"]):
        ind = _df[_df["basis_variation"] == basisv].index
        df_mol = _df["molecule"][ind]

        for mol in set(df_mol):

            index_mol_per_basis_mutation = df_mol[df_mol == mol].index
            imp = _df["ref-opb [hartree /atom]"][index_mol_per_basis_mutation].idxmin()
            out_df = out_df.append(_df.iloc[imp],ignore_index=True)

    return out_df

def get_average_impact_bv(df):
    mean_imp_arr = []
    mean_total_arr = []
    basis_var_arr = []

    for var in set(df["basis_variation"]):
        ind_val = df[df["basis_variation"] == var].index
        mean_val = df["ref-opb [hartree /atom]"][ind_val].mean()
        mean_total_avg = df["ref-init [hartree /atom]"][ind_val].mean()
        mean_imp_arr.append(mean_val)
        mean_total_arr.append(mean_total_avg)
        basis_var_arr.append(var)

    df_mean_basis_var = pd.DataFrame({"basis_variation": basis_var_arr,
                                      "mean energy difference [hartree/atom]": mean_imp_arr ,
                                      "ref - init energy difference [hartree/atom]":mean_total_arr})
    df_mean_basis_var["basis_variation"] = df_mean_basis_var["basis_variation"].astype(str)

    return df_mean_basis_var



# path = "/nfs/data-013/jaikinator/PycharmProjects/OptBasisSets/results.csv"
# df = pd.read_csv(path,index_col=0).reset_index(drop=True)
#
# df_best = get_best_res_mol(df, db = "w417")
# df_best.to_csv("evaluation/best_results_w417.csv", index=False)



path = "/nfs/data-013/jaikinator/PycharmProjects/OptBasisSets/analyse_data/evaluation/best_results_w417.csv"
df = pd.read_csv(path,index_col=0).reset_index(drop=True)
df_mean = get_average_impact_bv(df)

df_mean["mean energy difference [hartree/atom]"] = df_mean["mean energy difference [hartree/atom]"].abs()
df_mean["ref - init energy difference [hartree/atom]"] = df_mean["ref - init energy difference [hartree/atom]"].abs()

df_new = df[df["basis_variation"] == "('STO-3G', 'cc-pvtz')"]
df_new["diff of diff [hartree/atom]"] = df_new["ref-opb [hartree /atom]"] - df_new["ref-init [hartree /atom]"]
print(df_new)
fig1 = px.histogram(df_new, x="diff of diff [hartree/atom]", nbins=50)
fig1.show()
#plotly bar plot with two y axis

# fig1 = px.scatter(df_new, x="ref-opb [hartree /atom]", y="ref-init [hartree /atom]",
#                  hover_name="molecule", hover_data= df.columns.values
#                  ,color='method', size='number_of_atoms', marginal_x="histogram", marginal_y="histogram")
# fig1.show()



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