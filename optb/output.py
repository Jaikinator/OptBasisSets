"""
configure output of a minimization
"""
import os ,shutil
import json
import warnings
from time import asctime
import pandas as pd

import optb


def conf_output(basis,refbasis, atomstruc, outf = None, overwrite = False, **kwargs):
    """
    configure writer and path to save other data
    :param basis: name of basis which should be optimized
    :param refbasis: reference basis
    :param atomstruc: Name of molecule
    :param step: learning rate
    :param rtol: relative tolerance of the function
    :param outf: output folder path
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

    writerpath = conf_writer_path(outdir)

    print("save output to: ", outdir)
    return writerpath, outdir

def conf_writer_path(path):
    """
    configure writer path for tensorboard
    :param path: path to output folder
    :return: writer path
    """
    exist = True
    it = 1
    while exist:  # if you run multipile calc take care to not overwrite the old output files

        exist = os.path.exists(f"{path}/TB_00{it}") \
                or os.path.exists(f"{path}/TB_0{it}") \
                or os.path.exists(f"{path}/TB_{it}")
        if exist:
            it += 1
        else:
            if it < 10:
               writerpath = f"{path}/TB_00{it}"
            elif it >= 10 and it < 100:
               writerpath = f"{path}/TB_0{it}.out"
            else:
               writerpath = f"{path}/TB_{it}"

    return writerpath

def conf_output_old(basis,refbasis, atomstruc,step, rtol, outf = None, comments:str = "", overwrite = False, **kwargs):
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

def save_output_old (outdir, b1, b1_energy,b2, b2_energy,optbasis, optbasis_energy, atomstruc, lr ,
                maxiter, method , f_rtol , packer, misc: dict = {} ,optkwargs : dict = {},):
    """
    odler version of save_output which is used for the old version of the code
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

    if "best_x" in misc.keys():
        best_x = optb.bconv(misc["best_x"],packer)
        misc.pop('best_x', None)
    else:
        best_x = optbasis
    if "best_f" in misc.keys():
        misc["best_f"] = float(misc["best_f"])

    mini_dict = {"learning rate": lr,
                 "maxiter": maxiter,
                 "method": method,
                 "f_rtol" : f_rtol,
                 **misc,
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
                if optbasis != best_x:
                    jsonbasisf_best = f'{outdir}/{atomstruc}_opt_{b1}_best_basis_00{it}.json'
                else :
                    jsonbasisf_best = False
            elif it >= 10 and it < 100:
                df_outp = f'{outdir}/{atomstruc}_opt_{b1}_energy_0{it}.csv'
                df_outp_mini = f'{outdir}/{atomstruc}_learning_settings_0{it}.csv'
                jsonbasisf = f'{outdir}/{atomstruc}_opt_{b1}_basis_0{it}.json'
                if optbasis != best_x:
                    jsonbasisf_best = f'{outdir}/{atomstruc}_opt_{b1}_best_basis_0{it}.json'
                else :
                    jsonbasisf_best = False
            else:
                df_outp = f'{outdir}/{atomstruc}_opt_{b1}_energy_0{it}.csv'
                df_outp_mini = f'{outdir}/{atomstruc}_learning_settings_0{it}.csv'
                jsonbasisf = f'{outdir}/{atomstruc}_opt_{b1}_basis_{it}.json'
                if optbasis != best_x:
                    jsonbasisf_best = f'{outdir}/{atomstruc}_opt_{b1}_best_basis_{it}.json'
                else :
                    jsonbasisf_best = False
            df.to_csv(df_outp, na_rep='NaN',index=False)
            df_mini.to_csv(df_outp_mini, na_rep='NaN', index=False)

            with open(jsonbasisf, 'w', encoding='utf-8') as file:
                json.dump(optbasis, file, ensure_ascii=False, indent=4)

            if jsonbasisf_best:
                with open(jsonbasisf_best, 'w', encoding='utf-8') as file:
                        json.dump(best_x, file, ensure_ascii=False, indent=4)


def save_output(outdir, b1, b1_energy,b2, b2_energy,optbasis, optbasis_energy, atomstruc, lr ,
                maxiter, miniter, method , packer, misc: dict = {} ,optkwargs : dict = {},):
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
    :param lr: learning rate
    :param maxiter: max number of iterations
    :param miniter: min number of iterations
    :param method: method used for optimization
    :param f_rtol: tolerance for the energy
    :param packer: packer used
    :param molout: scf output
    :param misc: dict of misc settings
    :param optkwargs: dict of optimization settings
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


    if "best_x" in misc.keys():
        best_x = optb.bconv(misc["best_x"],packer)
        misc.pop('best_x', None)
    else:
        best_x = optbasis
    if "best_f" in misc.keys():
        misc["best_f"] = float(misc["best_f"])



    it = 1
    exist = True

    #create file names:

    while exist: # if you run multipile calc take care to not overwrite the old output files
        exist = os.path.exists(f"{outdir}/{atomstruc}_res_00{it}.csv") \
                or os.path.exists(f"{outdir}/{atomstruc}_res_0{it}.csv")\
                or os.path.exists(f"{outdir}/{atomstruc}_res_{it}.csv")
        if exist:
            it += 1

        else:

            if it < 10:
                df_outp = f'{outdir}/{atomstruc}_res_00{it}.csv'
                jsonbasisf = f'{outdir}/{atomstruc}_opt_{b1}_basis_00{it}.json'
                if optbasis != best_x:
                    jsonbasisf_best = f'{outdir}/{atomstruc}_opt_{b1}_best_basis_00{it}.json'
                else :
                    jsonbasisf_best = False
            elif it >= 10 and it < 100:

                df_outp = f'{outdir}/{atomstruc}_res_0{it}.csv'
                jsonbasisf = f'{outdir}/{atomstruc}_opt_{b1}_basis_0{it}.json'
                if optbasis != best_x:
                    jsonbasisf_best = f'{outdir}/{atomstruc}_opt_{b1}_best_basis_0{it}.json'
                else :
                    jsonbasisf_best = False
            else:
                df_outp = f'{outdir}/{atomstruc}_res_{it}.csv'
                jsonbasisf = f'{outdir}/{atomstruc}_opt_{b1}_basis_{it}.json'
                if optbasis != best_x:
                    jsonbasisf_best = f'{outdir}/{atomstruc}_opt_{b1}_best_basis_{it}.json'
                else :
                    jsonbasisf_best = False

            mini_dict = {"molecule": atomstruc,
                         "file number": it,
                         f"{b1}_energy": b1_energy,
                         f"{b2}_energy": b2_energy,
                         f"{b1}_opt_energy": optbasis_energy,
                         "learning rate": lr,
                         "maxiter": maxiter,
                         "miniter": miniter,
                         "method": method,
                         **misc,
                         **optkwargs}  # minimizer kwargs

            df = pd.DataFrame(mini_dict, index=[0])

            df.to_csv(df_outp, na_rep='NaN', index=False)

            with open(jsonbasisf, 'w', encoding='utf-8') as file:
                json.dump(optbasis, file, ensure_ascii=False, indent=4)

            if jsonbasisf_best:
                with open(jsonbasisf_best, 'w', encoding='utf-8') as file:
                        json.dump(best_x, file, ensure_ascii=False, indent=4)


"""
merge the resulting output files for better Legibility
"""

def merge_mol_old(mol_path):
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
    :param path: folder where data is stored
    :param basis_dir: specify basis set if u want
    :param mol_dir: specify molecule in basis set if u want
    :param save: specify tree deepness where result files will be created
                    1 equals to upper tree ==> just one result file in path. where everything is in
                    2 equals to every basis folder ==> one result file each basis folder and 1
                    3 every molecule in every basis gets one result file. +2 +1
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
                all_avail_files = [dir for dir in os.listdir(mol_path) if not os.path.isdir(os.path.join(mol_path, dir))]
                # search for keywords in avail_files
                old_fname = lambda x: "learning_settings" in x or "energy" in x # search for keywords in avail_files

                if True in list(map(old_fname,all_avail_files)): # if keywords are found fall back to old method
                    moldf = merge_mol_old(mol_path)
                    warnings.warn("learning_settings or energy in avail data fall back"
                                  " to older version results.csv might be wrong.",FutureWarning)
                else:
                    moldf = pd.DataFrame()
                    for file in all_avail_files:
                        if not ".json" in file or not "results.csv" in file:
                            if "results.csv" in file: # get sure result file is not added twice
                                pass
                            elif "res" in file:
                                _data_df = pd.read_csv(os.path.join(mol_path, file))
                                moldf = moldf.append(_data_df, ignore_index=True)


                if save >= 3:
                    moldf.to_csv(f"{mol_path}/results.csv", na_rep='NaN')

                if not "molecule" in moldf.columns.values:
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

            allmoldf = allmoldf.rename(columns={f"{basis12[0]}_energy": "initial_energy",
                                                f"{basis12[1]}_energy": "ref_energy",
                                                f"{basis12[0]}_opt_energy" : "opt_energy"})

            allbasisdf = allbasisdf.append(allmoldf, ignore_index=False)

        if save >= 1:
            allbasisdf.to_csv(f"{path}/results.csv", na_rep='NaN')

    else:
        # will do nothing.
        pass
        # mol_path = os.path.join(basis_path, mol_dir)
        # moldf = merge_mol(mol_path)


