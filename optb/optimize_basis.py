import warnings

import torch
import xitorch.optimize

from torch.utils.tensorboard import SummaryWriter
from optb.output import *
from numpy import any , isnan , nan


def scf_dft_energy(basis, atomstruc, atomstrucstr):
    """
    runs a dft calculation using pyscf.
    :return: total energy of the system.

    """

    for val in basis.values():
        for element in val:
            for orb in element:
                if type(orb) is list:
                    if any(isnan(orb)):
                        warnings.warn("Your Basis includes NaN Values no energy calculated")
                        return nan

    mol = gto.Mole()
    mol.atom = atomstruc
    mol.unit = 'Bohr'  # in Angstrom
    mol.verbose = 6
    if atomstrucstr is not None:
        if os.path.exists("./output"):
            mol.output = f'./output/scf_optB_{atomstrucstr}.out'  # try to save it in outputfolder
        else:
            mol.output = f'scf_optB_{atomstrucstr}.out'  # save in home folder
    mol.symmetry = False
    mol.basis = basis

    try:
        mol.spin = 0
        mol.build()
    except:
        mol.spin = 1
        mol.build()

    mf = scf.RKS(mol)
    mf.xc = 'B3LYP'
    mf.kernel()
    return mf.energy_tot()

def optimize_basis(basis: str, basis_ref : str, atomstruc : Union[str, list],step: list[float] , maxiter = 100000000,
                   method: str = "Adam", diverge = -1.0 ,maxdivattempts = 50 , output_path = None ,  minimize_kwargs: dict = {}, out_kwargs : dict = {}):
    """
    function to optimize basis functions.
    :param basis: basis which should be optimized
    :param basis_ref: reference basis
    :param atomstruc: Molecule Structure
    :param step: learning rate
    :param maxiter: maximal learning steps
    :param method: method to minimize
    :param diverge: for xitorch minimizer to check if the learning diverges
    :param minimize_kwargs: kwargs for the xitorch minimizer
    :param out_kwargs: kwargs to configure the output save functions
    :return: optimized basis
    """

    if minimize_kwargs["f_rtol"]:
        f_rtol = minimize_kwargs["f_rtol"]
        minimize_kwargs.pop("f_rtol")
        if type(f_rtol) is list and len(f_rtol) == 1:
            f_rtol = f_rtol[0]
    else:
        f_rtol = 1e-8


    if type(step) is list and type(f_rtol) is float:
        f_rtol_arr = []
        for i in range(len(step)):
            f_rtol_arr.append(f_rtol)
        f_rtol = f_rtol_arr
    elif type(step) is list and len(step) != len(f_rtol):
        msg = f"There are {len(step)} steps given but more than one or not an equal number of f_rtol as step. " \
              f"f_rtol has to be one or equal amount as steps."
        raise ValueError(msg)

    if type(atomstruc) is list:
        check_atomstruclist_of_str = all(isinstance(elem, str) for elem in atomstruc) #chek if list of atomstruc is list of molecule names
    else:
        check_atomstruclist_of_str = False

    if type(step) is list and check_atomstruclist_of_str:

        if len(atomstruc) == len(step):

            for i in range(len(atomstruc)):

                print(f"\n start optimization of {basis} Basis for the Molecule {atomstruc[i]}")

                bsys1, bsys2, func_dict = Mole_minimizer(basis, basis_ref, atomstruc[i])

                writerpath, outpath = conf_output(basis, basis_ref, atomstruc[i], step[i], f_rtol,
                                                  outf= output_path, **out_kwargs)
                writer = SummaryWriter(writerpath)

                min_bparams = xitorch.optimize.minimize(projection,
                                                        func_dict["bparams"],
                                                        (func_dict["bpacker"],
                                                         func_dict["ref_basis"],
                                                         func_dict["atomstruc_dqc"],
                                                         func_dict["atomstruc"],
                                                         func_dict["coeffM"],
                                                         func_dict["occ_scf"],
                                                         func_dict["num_gauss"],),
                                                        step=step[i],
                                                        method=method,
                                                        maxiter=maxiter,
                                                        verbose=True,
                                                        writer=writer,
                                                        diverge = diverge,
                                                        maxdivattempts = maxdivattempts,
                                                        f_rtol = f_rtol[i]
                                                        ,**minimize_kwargs)

                bsys1.get_SCF(atomstrucstr = atomstruc[i])
                energy_small_basis = bsys1.SCF.get_tot_energy
                energy_ref_basis = bsys2.SCF.get_tot_energy

                optbasis = bconv(min_bparams, func_dict["bpacker"])
                optbasis_energy = scf_dft_energy(optbasis, bsys1.atomstruc, atomstruc[i])

                save_output(outpath, basis, energy_small_basis, basis_ref, energy_ref_basis, optbasis,
                            optbasis_energy,
                            atomstruc[i], step[i], maxiter, method,f_rtol, optkwargs=minimize_kwargs)
            return optbasis

        else:
            for atom in atomstruc:
                bsys1, bsys2, func_dict = Mole_minimizer(basis, basis_ref, atom)

                for s in range(len(step)):

                    print(f"\n start optimization of {basis} Basis for the Molecule {atom}, with {len(bsys2.atomstruc)}")

                    writerpath, outpath = conf_output(basis, basis_ref, atom, step[s], f_rtol,
                                                      outf=output_path, **out_kwargs)
                    writer = SummaryWriter(writerpath)

                    min_bparams = xitorch.optimize.minimize(projection,
                                                            func_dict["bparams"],
                                                            (func_dict["bpacker"],
                                                             func_dict["ref_basis"],
                                                             func_dict["atomstruc_dqc"],
                                                             func_dict["atomstruc"],
                                                             func_dict["coeffM"],
                                                             func_dict["occ_scf"],
                                                             func_dict["num_gauss"],),
                                                            step=step[s],
                                                            method=method,
                                                            maxiter=maxiter,
                                                            verbose=True,
                                                            writer=writer,
                                                            diverge=diverge,
                                                            maxdivattempts=maxdivattempts,
                                                            f_rtol = f_rtol[s],
                                                            **minimize_kwargs)

                    bsys1.get_SCF(atomstrucstr=atom)
                    energy_small_basis = bsys1.SCF.get_tot_energy
                    energy_ref_basis = bsys2.SCF.get_tot_energy

                    optbasis = bconv(min_bparams, func_dict["bpacker"])
                    optbasis_energy = scf_dft_energy(optbasis, bsys1.atomstruc, atom)

                    save_output(outpath, basis, energy_small_basis, basis_ref, energy_ref_basis, optbasis,
                                optbasis_energy,
                                atom, step[s], maxiter, method ,f_rtol[s], optkwargs=minimize_kwargs)

            return optbasis

    elif check_atomstruclist_of_str and type(step) is float:

        for i in range(len(atomstruc)):

            print(f"\n start optimization of {basis} Basis for the Molecule {atomstruc[i]}")

            bsys1, bsys2, func_dict = Mole_minimizer(basis, basis_ref, atomstruc[i])

            writerpath, outpath = conf_output(basis, basis_ref, atomstruc[i], step, f_rtol,
                                              outf=output_path, **out_kwargs)
            writer = SummaryWriter(writerpath)

            min_bparams = xitorch.optimize.minimize(projection,
                                                    func_dict["bparams"],
                                                    (func_dict["bpacker"],
                                                     func_dict["ref_basis"],
                                                     func_dict["atomstruc_dqc"],
                                                     func_dict["atomstruc"],
                                                     func_dict["coeffM"],
                                                     func_dict["occ_scf"],
                                                     func_dict["num_gauss"],),
                                                    step=step,
                                                    method=method,
                                                    maxiter=maxiter,
                                                    verbose=True,
                                                    writer=writer,
                                                    diverge = diverge,
                                                    maxdivattempts=maxdivattempts,
                                                    f_rtol = f_rtol,
                                                    **minimize_kwargs)

            bsys1.get_SCF(atomstrucstr=atomstruc[i])
            energy_small_basis = bsys1.SCF.get_tot_energy
            energy_ref_basis = bsys2.SCF.get_tot_energy

            optbasis = bconv(min_bparams, func_dict["bpacker"])
            optbasis_energy = scf_dft_energy(optbasis, bsys1.atomstruc, atomstruc[i])

            save_output(outpath, basis, energy_small_basis, basis_ref, energy_ref_basis, optbasis,
                        optbasis_energy,
                        atomstruc[i], step, maxiter, method, f_rtol, optkwargs = minimize_kwargs)
        return optbasis

    elif type(atomstruc) is str and type(step) is list:

        for i in range(len(step)):

            print(f"\n start optimization of {basis} Basis for the Molecule {atomstruc}")

            bsys1, bsys2, func_dict = Mole_minimizer(basis, basis_ref, atomstruc)

            writerpath, outpath = conf_output(basis, basis_ref, atomstruc, step[i], f_rtol,
                                              outf=output_path, **out_kwargs)
            writer = SummaryWriter(writerpath)

            min_bparams = xitorch.optimize.minimize(projection,
                                                    func_dict["bparams"],
                                                    (func_dict["bpacker"],
                                                     func_dict["ref_basis"],
                                                     func_dict["atomstruc_dqc"],
                                                     func_dict["atomstruc"],
                                                     func_dict["coeffM"],
                                                     func_dict["occ_scf"],
                                                     func_dict["num_gauss"],),
                                                    step=step[i],
                                                    method=method,
                                                    maxiter=maxiter,
                                                    verbose=True,
                                                    writer=writer,
                                                    diverge=diverge,
                                                    maxdivattempts=maxdivattempts,
                                                    f_rtol = f_rtol[i],
                                                    **minimize_kwargs)

            bsys1.get_SCF(atomstrucstr=atomstruc)
            energy_small_basis = bsys1.SCF.get_tot_energy
            energy_ref_basis = bsys2.SCF.get_tot_energy

            optbasis = bconv(min_bparams, func_dict["bpacker"])
            optbasis_energy = scf_dft_energy(optbasis, bsys1.atomstruc, atomstruc)

            save_output(outpath, basis, energy_small_basis, basis_ref, energy_ref_basis, optbasis,
                        optbasis_energy,
                        atomstruc, step[i], maxiter, method,f_rtol = f_rtol[i], optkwargs=minimize_kwargs)
        return optbasis

    elif type(atomstruc) is str and type(step) is float:

        print(f"\n start optimization of {basis} Basis for the Molecule {atomstruc}")

        bsys1, bsys2, func_dict = Mole_minimizer(basis, basis_ref, atomstruc)

        writerpath, outpath = conf_output(basis, basis_ref, atomstruc, step, f_rtol,
                                                  outf= output_path, **out_kwargs)
        writer = SummaryWriter(writerpath)

        min_bparams = xitorch.optimize.minimize(projection,
                                                func_dict["bparams"],
                                                (func_dict["bpacker"],
                                                 func_dict["ref_basis"],
                                                 func_dict["atomstruc_dqc"],
                                                 func_dict["atomstruc"],
                                                 func_dict["coeffM"],
                                                 func_dict["occ_scf"],
                                                 func_dict["num_gauss"],),
                                                step=step,
                                                method=method,
                                                maxiter=maxiter,
                                                verbose=True,
                                                writer=writer,
                                                diverge=diverge,
                                                maxdivattempts=maxdivattempts,
                                                f_rtol= f_rtol,
                                                **minimize_kwargs)

        bsys1.get_SCF(atomstrucstr=atomstruc)
        energy_small_basis = bsys1.SCF.get_tot_energy
        energy_ref_basis = bsys2.SCF.get_tot_energy

        optbasis = bconv(min_bparams, func_dict["bpacker"])
        optbasis_energy = scf_dft_energy(optbasis, bsys1.atomstruc, atomstruc)

        save_output(outpath, basis, energy_small_basis, basis_ref, energy_ref_basis, optbasis, optbasis_energy,
                    atomstruc, step, maxiter, method,f_rtol = f_rtol, optkwargs=minimize_kwargs)

        return optbasis

    elif type(atomstruc) is list and not check_atomstruclist_of_str:
        msg = f"optimize_basis does not support list Atomtype of atomstruc (e.g. [element , [x, y, z]])" \
              f" yet pls do it manually."#
        raise msg

    else:
       raise "Something went wrong"
