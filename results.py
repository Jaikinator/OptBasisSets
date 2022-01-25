import os
import pandas as pd

def merge_mol(mol_path):
        """
        reads all results from a given molecule and wrap them up to a single Dataframe
        """

        raw_data = [dir for dir in os.listdir(mol_path) if not os.path.isdir(os.path.join(mol_path, dir))]

        moldf = pd.DataFrame()
        lsettings = []
        energys = []

        for data in raw_data:
            if not ".json" in data or not "results.csv" in data :
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
                    df = pd.concat([df_lr, df_energy], axis= 1)
                    moldf = moldf.append(df, ignore_index=True)
                    i += 1

        return moldf

def merge_data(path = False, basis_dir = False , mol_dir = False , save = 3 ):
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
                allmoldf = allmoldf.append(moldf, ignore_index= False)

            if save >= 2:
                allmoldf.to_csv(f"{basis_path}/results.csv", na_rep='NaN')

            basis12 = folder.split("_")
            basis1 = [basis12[0] for _ in range(len(allmoldf.index))]
            basis_ref = [basis12[1] for _ in range(len(allmoldf.index))]

            allmoldf["basis"] = basis1
            allmoldf["ref. basis"] = basis_ref

            allbasisdf = allbasisdf.append(allmoldf, ignore_index=False)

        if save >=1:
            allmoldf.to_csv(f"{path}/results.csv", na_rep='NaN')

    else:
        # will do nothing.
        pass
        # mol_path = os.path.join(basis_path, mol_dir)
        # moldf = merge_mol(mol_path)



merge_data()