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

    writerpath = f"{outdir}/TB_lr{step}_f_rtol{rtol}_{asctime()}"

    return writerpath, outdir

def save_output(outdir, b1, b1_energy,b2, b2_energy,optbasis, optbasis_energy):
    """
    save data after minimization
    :param outdir: folder where data will be stored
    :param b1: name of basis which should be optimized
    :param b1_energy: energy of the dft calc using b1 as basis
    :param b2: name of the reference basis
    :param b2_energy: energy of the dft calc using b2 as basis
    :param optbasis: dict of the optimized basis set
    :param optbasis_energy: energy of the dft calc using optbasis as basis
    :return: None
    """
    print(f"total energy scf with {b1} as initial basis:\n",
          b1_energy)
    print(f"total energy scf with {b2} as reference basis:\n",
          b2_energy)
    print("energy after optimization of basis as basis set:\n",
          optbasis_energy)

    energy_out = {f"{b1}_energy" : b1_energy,
                  f"{b2}_energy": b2_energy,
                  f"{b1}_opt_energy": optbasis_energy}

    df = pd.DataFrame(energy_out, index=[0])
    df_outp = f'{outdir}/opt_{b1}_energy.csv'
    df.to_csv(df_outp)

    jsonbasisf = f'{outdir}/opt_{b1}_basis.json'

    with open(jsonbasisf, 'w', encoding='utf-8') as file:
        json.dump(optbasis, file, ensure_ascii=False, indent=4)

