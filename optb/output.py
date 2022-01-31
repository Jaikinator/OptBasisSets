"""
configure output of a minimization
"""
import os ,shutil
import json

from time import asctime

from optb import *

import pandas as pd

def conf_output(basis,refbasis, atomstruc, step, rtol, outf = None, comments:str = "", overwrite = False, **kwargs):
    """
    configure writer and path to save other data
    :param basis: name of basis which should be optimized
    :param refbasis: reference basis
    :param atomstruc: Name of molecule
    :param step: learning rate
    :param rtol: relative tolerance of the function
    :param outf: output folder path
    :param comments: comments
    :param overwrite: overwrite old data
    :param kwargs:
    :return: writer path for Tensorboard and path for other data
    """

    if type(atomstruc) is not str:
        atomstruc = str(atomstruc)

    if outf is None:
        outf = os.path.dirname(os.path.realpath(__file__))
        _outf  = os.path.join(outf, "output")
        if not os.path.exists(_outf):
            os.mkdir(_outf)
        outf = _outf
    elif "output" not in outf:
        _outf = os.path.join(outf, "output")
        if not os.path.exists(_outf):
            os.mkdir(_outf)
        outf = _outf
    else:
        pass

    outb_dir = f"{outf}/{basis}_{refbasis}"
    outdir = os.path.join(outb_dir, atomstruc)
    if not os.path.exists(outb_dir):
        os.mkdir(outb_dir)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if os.path.exists(outdir) and overwrite:
        text = f"folder removed: {atomstruc}"
        warnings.warn(text)
        shutil.rmtree(outdir)

    writerpath = f"{outdir}/TB_{atomstruc}_lr{step}_f_rtol{rtol}_{asctime()}"

    print("save output to: ", outdir)
    return writerpath, outdir

def save_output(outdir, b1, b1_energy,b2, b2_energy,optbasis, optbasis_energy, atomstruc, lr , maxiter,method = "Adam", f_rtol =1e-8 ,optkwargs : dict = {}):
    """
    save data after minimization
    :param outdir: folder where data will be stored
    :param b1: name of basis which should be optimized
    :param b1_energy: energy of the dft calc using b1 as basis
    :param b2: name of the reference basis
    :param b2_energy: energy of the dft calc using b2 as basis
    :param optbasis: dict of the optimized basis set
    :param optbasis_energy: energy of the dft calc using optbasis as basis
    :param atomstruc: atomstructure
    :return: None
    """
    if type(atomstruc) is not str:
        atomstruc = str(atomstruc)
    print(f"total energy scf with {b1} as initial basis:\n",
          b1_energy)
    print(f"total energy scf with {b2} as reference basis:\n",
          b2_energy)
    print("energy after optimization of basis as basis set:\n",
          optbasis_energy)

    energy_out = {f"{b1}_energy" : b1_energy,
                  f"{b2}_energy": b2_energy,
                  f"{b1}_opt_energy": optbasis_energy}


    mini_dict = {"learning rate": lr,
                 "maxiter": maxiter,
                 "method": method,
                 "f_rtol" : f_rtol,
                 **optkwargs}  # minimizer kwargs

    df = pd.DataFrame(energy_out, index=[0])

    df_mini = pd.DataFrame(mini_dict, index=[0])


    it = 1
    exist = True

    while exist: # if you run multipile calc take care to not overwrite the old output files
        exist = os.path.exists(f"{outdir}/{atomstruc}_opt_{b1}_energy_00{it}.csv") \
                or os.path.exists(f"{outdir}/{atomstruc}_opt_{b1}_energy_0{it}.csv")\
                or os.path.exists(f"{outdir}/{atomstruc}_opt_{b1}_energy_{it}.csv")
        if exist:
            it += 1

        else:

            if it < 10:
                df_outp = f'{outdir}/{atomstruc}_opt_{b1}_energy_00{it}.csv'
                df_outp_mini = f'{outdir}/{atomstruc}_learning_settings_00{it}.csv'
                jsonbasisf = f'{outdir}/{atomstruc}_opt_{b1}_basis_00{it}.json'
            elif it >= 10 and it < 100:
                df_outp = f'{outdir}/{atomstruc}_opt_{b1}_energy_0{it}.csv'
                df_outp_mini = f'{outdir}/{atomstruc}_learning_settings_0{it}.csv'
                jsonbasisf = f'{outdir}/{atomstruc}_opt_{b1}_basis_0{it}.json'
            else:
                df_outp = f'{outdir}/{atomstruc}_opt_{b1}_energy_0{it}.csv'
                df_outp_mini = f'{outdir}/{atomstruc}_learning_settings_0{it}.csv'
                jsonbasisf = f'{outdir}/{atomstruc}_opt_{b1}_basis_{it}.json'

            df.to_csv(df_outp, na_rep='NaN',index=False)
            df_mini.to_csv(df_outp_mini, na_rep='NaN', index=False)

            with open(jsonbasisf, 'w', encoding='utf-8') as file:
                json.dump(optbasis, file, ensure_ascii=False, indent=4)





"""
merge the resulting output files for better Legibility
"""



def merge_mol(mol_path):
    """
    reads all results from a given molecule and wrap them up to a single Dataframe
    """

    raw_data = [dir for dir in os.listdir(mol_path) if not os.path.isdir(os.path.join(mol_path, dir))]

    moldf = pd.DataFrame()
    lsettings = []
    energys = []

    for data in raw_data:
        if not ".json" in data or not "results.csv" in data:
            if "learning_settings" in data:
                lsettings += [data]
            if "energy" in data:
                energys += [data]
    i = 1
    while i <= len(lsettings):
        for datasettings, dataenergys in zip(lsettings, energys):
            if str(i) in datasettings:
                df_lr = pd.read_csv(os.path.join(mol_path, datasettings))
                df_energy = pd.read_csv(os.path.join(mol_path, dataenergys))
                df = pd.concat([df_lr, df_energy], axis=1)
                moldf = moldf.append(df, ignore_index=True)
                i += 1

    return moldf


def merge_data(path=False, basis_dir=False, mol_dir=False, save=3):
    """
    merges all output files of every output (of type .csv)
    """
    if os.path.exists("output") and not path:
        path = os.path.abspath("output")

    if not basis_dir:
        basis_dir = [dir for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]
    else:
        basis_path = os.path.join(path, basis_dir)

    if not mol_dir:
        allbasisdf = pd.DataFrame()

        for folder in basis_dir:
            basis_path = os.path.join(path, folder)
            mol_dir = [dir for dir in os.listdir(basis_path) if os.path.isdir(os.path.join(basis_path, dir))]

            allmoldf = pd.DataFrame()

            for mol in mol_dir:
                mol_path = os.path.join(basis_path, mol)
                moldf = merge_mol(mol_path)
                if save >= 3:
                    moldf.to_csv(f"{mol_path}/results.csv", na_rep='NaN')

                molname = [mol for _ in range(len(moldf.index))]
                moldf["molecule"] = molname
                allmoldf = allmoldf.append(moldf, ignore_index=False)

            if save >= 2:
                allmoldf.to_csv(f"{basis_path}/results.csv", na_rep='NaN')

            basis12 = folder.split("_")
            basis1 = [basis12[0] for _ in range(len(allmoldf.index))]
            basis_ref = [basis12[1] for _ in range(len(allmoldf.index))]

            allmoldf["basis"] = basis1
            allmoldf["ref. basis"] = basis_ref

            allbasisdf = allbasisdf.append(allmoldf, ignore_index=False)

        if save >= 1:
            allmoldf.to_csv(f"{path}/results.csv", na_rep='NaN')

    else:
        # will do nothing.
        pass
        # mol_path = os.path.join(basis_path, mol_dir)
        # moldf = merge_mol(mol_path)


