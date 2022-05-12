"""
Evaluate the result.csv
"""

import os
import time

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from optb import loadatomstruc
from optb.data.avdata import elg2 as g2, elw417 as w417
import warnings
import logging as log

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

def remove_SCF_not_converged(df, threshold = 10):
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

        ind_not_converged = _df_mol[abs(_df_mol["ref_energy"] / _df_mol["initial_energy"]) > threshold].index

        if str(basisv) == "('aug-cc-pVDZ', 'aug-pc-2')":
            print("Hello", basisv)
            test = _df_mol[_df_mol["molecule"] == "CH3OCH3"]
            print(test )
            print(test["ref_energy"] / test["initial_energy"])
        _df_mol.drop(ind_not_converged, inplace=True) #just to check which molecules are dropped
        _df_mol.reset_index(drop=True, inplace=True)

        _df.drop(ind_not_converged, inplace=True) # actual drop the results in the main dataframe
        _df.reset_index(drop=True, inplace=True)

        molec_new = set(_df_mol["molecule"])
        print(f"\n\t For basis variation: {basisv}\n\t Removed {len(molec_old - molec_new)} molecules.")


        if len(molec_old - molec_new) > 0:
            print(f"\n\tThe molecules that are not converged for {basisv} are written in evaluation/molecules_dropped_{basisv}.txt\n")
            with open(f"evaluation/molecules_dropped_{basisv}.txt", "w") as f:
                f.write(f"molecule of the basis variation {basisv} dropped because the results of pyscf are not converged for large basis): "
                        + str(len(molec_old - molec_new)) + "\n")
                for i in molec_old - molec_new:
                    f.write(i + "\n")
    return _df


def get_best_res_mol(df):
    """
    Get the best result for each molecule and basis variation.
    """
    _df = df.copy()
    _df = calculate_impact(_df)
    _df = drop_small_best_i(_df, threshold=10)
    _df = remove_SCF_not_converged(_df, threshold=10)
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
    basis_var_arr = []
    for var in set(df["basis_variation"]):
        ind_val = df[df["basis_variation"] == var].index
        if var == "('aug-cc-pVDZ', 'aug-pc-2')":
            print(df[df["molecule"] == "CH3OCH3" ])

        mean_val = df["ref-opb [hartree /atom]"][ind_val].mean()
        mean_imp_arr.append(mean_val)
        basis_var_arr.append(var)

    df_mean_basis_var = pd.DataFrame({"basis_variation": basis_var_arr,"mean energy difference": mean_imp_arr})
    df_mean_basis_var["basis_variation"] = df_mean_basis_var["basis_variation"].astype(str)

    return df_mean_basis_var

def filter_good_results(df):
    """
    Filter the good results.
    """
    _df = df.copy()

    _df = drop_small_best_i(_df)

    indx = _df[_df["optb-initial"] > 0].index
    _df.drop(indx, inplace=True)

    indx3 = _df[_df["best_i"] < + 10].index
    _df.drop(indx3, inplace=True)
    indx4 = _df[_df["rel. improvement %"] > 100].index
    _df.drop(indx4, inplace=True)
    indx5 = _df[_df["rel. improvement %"] < 0].index
    _df.drop(indx5, inplace=True)

    return _df

def filter_bad_results(df):
    """
    Filter the bad results.
    """

    _df = df.copy()


    indx = _df[_df["optb-initial"] < 120].index
    _df.drop(indx, inplace=True)
    # indx2 = df[df["optb-initial"] < 1e-5].index
    # df.drop(indx2, inplace=True)
    indx4 = _df[_df["rel. improvement %"] < 120].index
    _df.drop(indx4, inplace=True)
    # indx5 = df[df["rel. improvement %"] > 0].index
    # df.drop(indx5, inplace=True)
    _df.reset_index(drop=True, inplace=True)
    return _df


path = "/nfs/data-013/jaikinator/PycharmProjects/OptBasisSets/results.csv"
df = pd.read_csv(path,index_col=0).reset_index(drop=True)

# df = calculate_impact(df)
# df = drop_small_best_i(df, threshold=10)

df_best = get_best_res_mol(df)
df_best.to_csv("evaluation/best_results.csv", index=False)



# path = "/nfs/data-013/jaikinator/PycharmProjects/OptBasisSets/analyse_data/evaluation/best_results.csv"
# df = pd.read_csv(path,index_col=0).reset_index(drop=True)
# df = get_average_impact_bv(df)
# print(df)

#
# mean_imp_arr = []
# basis_var_arr = []
#
# for var in basis_var:
#     ind_val = df[df["basis variation"] == var].index
#     mean_val = df["improvement %"][ind_val].mean()
#     mean_imp_arr.append(mean_val)
#     basis_var_arr.append(var)
#
# df_mean_basis_var = pd.DataFrame({"basis variation": basis_var_arr,"mean improvement %": mean_imp_arr})
# df_mean_basis_var["basis variation"] = df_mean_basis_var["basis variation"].astype(str)
# df_mean_basis_var.to_csv("evaluation/mean_basis_var.csv")
#
# print(df_mean_basis_var)
#
# fig = px.bar(df_mean_basis_var, x="basis variation", y="mean improvement %",
#              title="Mean improvement per basis variation")
# fig.update_layout(font=dict(size=20))
#
# fig.write_html("evaluation/mean_basis_var_not_filtert.html")