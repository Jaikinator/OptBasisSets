import warnings

import torch
from  pyscf import gto, scf
import xitorch.optimize
from typing import Union
from torch.utils.tensorboard import SummaryWriter
from optb.output import *
from numpy import any , isnan , nan
from optb.projection import projection
from optb.Mole import Mole_minimizer
from optb.basis_converter import bconv
from optb.AtomsDB import loadatomstruc


def scf_dft_energy(basis :dict , atomstruc : list, atomstrucstr = None, outpath : str = "./"):
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
            # try to save it in outputfolder
            savep = f'./output/scf_optB_{atomstrucstr}.out'

            exist = os.path.exists(savep)
            if exist:
                it = 1
                while exist:  # if you run multipile calc take care to not overwrite the old output files

                    exist = os.path.exists(f"{outpath}/scf_optB_{atomstrucstr}_00{it}.out") \
                            or os.path.exists(f"{outpath}/scf_optB_{atomstrucstr}_0{it}.out")\
                            or os.path.exists(f"{outpath}/scf_optB_{atomstrucstr}_{it}.out")
                    if exist:
                        it += 1
                    else:
                        if it < 10:
                             mol.output = f"{outpath}/scf_optB_{atomstrucstr}_00{it}.out"
                        elif it >= 10 and it < 100:
                            mol.output = f"{outpath}/scf_optB_{atomstrucstr}_0{it}.out"
                        else:
                            mol.output = f"{outpath}/scf_optB_{atomstrucstr}_{it}.out"

            else:
                mol.output = savep
        else:
            mol.output = f'scf_optB_{atomstrucstr}.out'  # save in home folder
    else:
        mol.output = "scf.out"

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

def optimize_basis_old(basis: str,
                   basis_ref : str,
                   atomstruc : Union[str, list],
                   step: list[float],
                   maxiter : int = 100000000,
                   miniter : int = 1 ,
                   method: str = "Adam",
                   diverge = -1.0 ,
                   maxdivattempts = 50,
                   output_path = None,
                   get_misc = True,
                   mol_kwargs: dict = {},
                   minimize_kwargs: dict = {},
                   out_kwargs : dict = {}):
    """
    function to optimize basis functions.
    :param basis: basis which should be optimized
    :param basis_ref: reference basis
    :param atomstruc: Molecule Structure
    :param step: learning rate
    :param maxiter: maximal learning steps
    :param miniter: minimal learning steps (to avoid local minima)
    :param method: method to minimize
    :param diverge: for xitorch minimizer to check if the learning diverges
    :param maxdivattempts: attemps for divergens break up
    :param output_path: path where output data will be saved
    :param get_misc: adds Miscellaneous params to savefiles
    :param mol_kwargs: kwargs for the Mole Class called by Mole_minimizer
    :param minimize_kwargs: kwargs for the xitorch minimizer
    :param out_kwargs: kwargs to configure the output save functions
    :return: optimized basis
    """

    if "f_rtol" in minimize_kwargs:
        f_rtol = minimize_kwargs["f_rtol"]
        minimize_kwargs.pop("f_rtol")
        if type(f_rtol) is list and len(f_rtol) == 1:
            f_rtol = f_rtol[0]
    else:
        f_rtol = 0.0


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

                bsys1, bsys2, func_dict = Mole_minimizer(basis, basis_ref, atomstruc[i],**mol_kwargs)

                writerpath, outpath = conf_output_old(basis, basis_ref, atomstruc[i], step[i], f_rtol,
                                                  outf= output_path, **out_kwargs)
                writer = SummaryWriter(writerpath)

                min_bparams, misc = xitorch.optimize.minimize(projection,
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
                                                        miniter=miniter,
                                                        verbose=True,
                                                        writer=writer,
                                                        diverge = diverge,
                                                        maxdivattempts = maxdivattempts,
                                                        get_misc = get_misc,
                                                        f_rtol = f_rtol[i],
                                                        **minimize_kwargs)
                print("line 170")
                bsys1.get_SCF(atomstrucstr = atomstruc[i])
                energy_small_basis = bsys1.SCF.get_tot_energy
                energy_ref_basis = bsys2.SCF.get_tot_energy

                optbasis = bconv(min_bparams, func_dict["bpacker"])
                optbasis_energy = scf_dft_energy(optbasis, bsys1.atomstruc, atomstruc[i], outpath)

                save_output(outpath, basis, energy_small_basis, basis_ref, energy_ref_basis, optbasis,optbasis_energy,
                            atomstruc[i], step[i], maxiter, miniter , method,f_rtol,func_dict["bpacker"], optkwargs=minimize_kwargs)
            return optbasis

        else:
            for atom in atomstruc:
                bsys1, bsys2, func_dict = Mole_minimizer(basis, basis_ref, atom, **mol_kwargs )
                bsys1.get_SCF(atomstrucstr=atom)
                for s in range(len(step)):

                    print(f"\n start optimization of {basis} Basis for the Molecule {atom} and learning-rate {step[s]}.")

                    writerpath, outpath = conf_output_old(basis, basis_ref, atom, step[s], f_rtol,
                                                      outf=output_path, **out_kwargs)
                    writer = SummaryWriter(writerpath)

                    min_bparams, misc = xitorch.optimize.minimize(projection,
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
                                                            miniter=miniter,
                                                            verbose=True,
                                                            writer=writer,
                                                            diverge=diverge,
                                                            maxdivattempts=maxdivattempts,
                                                            get_misc=get_misc,
                                                            f_rtol = f_rtol[s],
                                                            **minimize_kwargs)

                    print("line 215")
                    energy_small_basis = bsys1.SCF.get_tot_energy
                    energy_ref_basis = bsys2.SCF.get_tot_energy

                    optbasis = bconv(min_bparams, func_dict["bpacker"])
                    optbasis_energy = scf_dft_energy(optbasis, bsys1.atomstruc, atom, outpath)

                    save_output(outpath, basis, energy_small_basis, basis_ref, energy_ref_basis, optbasis,
                                optbasis_energy,
                                atom, step[s], maxiter, miniter, method ,f_rtol[s],func_dict["bpacker"]
                                ,misc = misc, optkwargs = minimize_kwargs)

            return optbasis

    elif check_atomstruclist_of_str and type(step) is float:

        for i in range(len(atomstruc)):

            print(f"\n start optimization of {basis} Basis for the Molecule {atomstruc[i]}")

            bsys1, bsys2, func_dict = Mole_minimizer(basis, basis_ref, atomstruc[i], **mol_kwargs)

            writerpath, outpath = conf_output_old(basis, basis_ref, atomstruc[i], step, f_rtol,
                                              outf=output_path, **out_kwargs)
            writer = SummaryWriter(writerpath)

            min_bparams, misc = xitorch.optimize.minimize(projection,
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
                                                    miniter=miniter,
                                                    verbose=True,
                                                    writer=writer,
                                                    diverge = diverge,
                                                    maxdivattempts=maxdivattempts,
                                                    get_misc=get_misc,
                                                    f_rtol = f_rtol,
                                                    **minimize_kwargs)
            print("line 261")
            bsys1.get_SCF(atomstrucstr= atomstruc[i])
            energy_small_basis = bsys1.SCF.get_tot_energy
            energy_ref_basis = bsys2.SCF.get_tot_energy

            optbasis = bconv(min_bparams, func_dict["bpacker"])
            optbasis_energy = scf_dft_energy(optbasis, bsys1.atomstruc, atomstruc[i], outpath)

            save_output(outpath, basis, energy_small_basis, basis_ref, energy_ref_basis, optbasis,
                        optbasis_energy,
                        atomstruc[i], step, maxiter, miniter, method, f_rtol,func_dict["bpacker"],
                        misc = misc, optkwargs = minimize_kwargs)
        return optbasis

    elif type(atomstruc) is str and type(step) is list:

        for i in range(len(step)):

            print(f"\n start optimization of {basis} Basis for the Molecule {atomstruc} and learning-rate {step[i]}")

            bsys1, bsys2, func_dict = Mole_minimizer(basis, basis_ref, atomstruc,**mol_kwargs)

            writerpath, outpath = conf_output_old(basis, basis_ref, atomstruc, step[i], f_rtol,
                                              outf=output_path, **out_kwargs)
            writer = SummaryWriter(writerpath)

            min_bparams, misc = xitorch.optimize.minimize(projection,
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
                                                    miniter=miniter,
                                                    verbose=True,
                                                    writer=writer,
                                                    diverge=diverge,
                                                    maxdivattempts=maxdivattempts,
                                                    get_misc=get_misc,
                                                    f_rtol = f_rtol[i],
                                                    **minimize_kwargs)
            print("line 307")
            bsys1.get_SCF(atomstrucstr=atomstruc)
            energy_small_basis = bsys1.SCF.get_tot_energy
            energy_ref_basis = bsys2.SCF.get_tot_energy

            optbasis = bconv(min_bparams, func_dict["bpacker"])
            optbasis_energy = scf_dft_energy(optbasis, bsys1.atomstruc, atomstruc, outpath)

            save_output(outpath, basis, energy_small_basis, basis_ref, energy_ref_basis, optbasis,
                        optbasis_energy,
                        atomstruc, step[i], maxiter, miniter, method, f_rtol = f_rtol[i], packer=func_dict["bpacker"],
                        misc = misc, optkwargs=minimize_kwargs)
        return optbasis

    elif type(atomstruc) is str and type(step) is float:

        print(f"\n start optimization of {basis} Basis for the Molecule {atomstruc} and learning-rate {step}")

        bsys1, bsys2, func_dict = Mole_minimizer(basis, basis_ref, atomstruc, **mol_kwargs)

        writerpath, outpath = conf_output_old(basis, basis_ref, atomstruc, step, f_rtol,
                                                  outf= output_path, **out_kwargs)
        writer = SummaryWriter(writerpath)

        min_bparams, misc = xitorch.optimize.minimize(projection,
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
                                                miniter=miniter,
                                                verbose=True,
                                                writer=writer,
                                                diverge=diverge,
                                                maxdivattempts=maxdivattempts,
                                                get_misc=get_misc,
                                                f_rtol= f_rtol,
                                                **minimize_kwargs)
        print("line 351")
        print(min_bparams)
        print(misc)
        bsys1.get_SCF(atomstrucstr=atomstruc)
        energy_small_basis = bsys1.SCF.get_tot_energy
        energy_ref_basis = bsys2.SCF.get_tot_energy

        optbasis = bconv(min_bparams, func_dict["bpacker"])
        optbasis_energy = scf_dft_energy(optbasis, bsys1.atomstruc, atomstruc, outpath)
        print(method)
        save_output(outpath, basis, energy_small_basis, basis_ref, energy_ref_basis, optbasis, optbasis_energy,
                    atomstruc, step, maxiter,miniter =  miniter, method = method,
                    f_rtol = f_rtol, packer=func_dict["bpacker"],misc = misc, optkwargs=minimize_kwargs)

        return optbasis

    elif type(atomstruc) is list and not check_atomstruclist_of_str:
        msg = f"optimize_basis does not support list Atomtype of atomstruc (e.g. [element , [x, y, z]])" \
              f" yet pls do it manually."
        raise msg

    else:
       raise "Something went wrong"


class OPTBASIS:

    def __init__(self,basis: str,
                   basis_ref : str,
                   molecule : Union[str, list],
                   step:  float,
                   maxiter : int = 100000000,
                   miniter : int = 1 ,
                   method: str = "Adam",
                   diverge = -1.0 ,
                   maxdivattempts = 50,
                   output_path = None,
                   get_misc = True,
                   mol_kwargs: dict = {},
                   minimize_kwargs: dict = {},
                   out_kwargs : dict = {}):
        """
            Class to optimize basis functions. It uses the xitorch.optimize.minimize function.
            be aware that the basis functions are optimized in the small basis set.
            :param basis: str basis which should be optimized
            :param basis_ref: str reference basis
            :param molecule: Molecule Name or list of molecule names
            :param step: floats with the lr size for the optimization
            :param maxiter: int maximum number of iterations
            :param miniter: int minimum number of iterations
            :param method: str method for optimization
            :param diverge: float diverge value
            :param maxdivattempts: int maximum number of divergence attempts
            :param output_path: str path to output folder
            :param get_misc: bool if True the misc dictionary is returned
            :param mol_kwargs: dict kwargs for Molecule class
            :param minimize_kwargs: dict kwargs for minimize function
            :param out_kwargs: dict kwargs for save_output function
            :return:
        """
        self.basis = basis
        self.basis_ref = basis_ref
        self.molecule = molecule
        self.step = step
        self.maxiter = maxiter
        self.miniter = miniter
        self.method = method
        self.diverge = diverge
        self.maxdivattempts = maxdivattempts
        self.output_path = output_path
        self.get_misc = get_misc
        self.mol_kwargs = mol_kwargs
        self.minimize_kwargs = minimize_kwargs
        self.out_kwargs = out_kwargs

        self.atomstruc = loadatomstruc(self.molecule, **self.mol_kwargs).atomstruc
        self.opt_bparams = torch.Tensor
        self.misc_params = dict

        if not output_path == None:
            writerp, self.outpath = self._setup_output_path()
            self.writer = SummaryWriter(writerp)
        else:
            self.writer = None
            self.outpath = None

        if isinstance(self.step, list):
            warnings.warn('Attention your input takes multiple learning rates opt_bparams as well as misc_params will be '
                          'overwritten')


    def select_optimal_optimizer(self):
        print("selecting optimal optimizer")
        if isinstance(self.molecule, str) and isinstance(self.step, float):
            print("optimizing basis for single molecule")
            return self.optimize_basis_single_molecule

        elif isinstance(self.molecule, list) and isinstance(self.step, float):
            check_atomstruclist_of_str = all(
                    isinstance(elem, str) for elem in self.molecule)  # chek if list of atomstruc is list of molecule names
            if check_atomstruclist_of_str:
                print("optimizing basis for list of molecules")
                return self.optimize_basis_list_of_molecules
            else:
                print("optimizing basis for a molecule but input is not a"
                      " list of molecule names instead a list of atomstruc")
                return self.optimize_basis_single_molecule

        elif isinstance(self.molecule, str) and isinstance(self.step, list):
            print("optimizing basis for single molecule but input is a list of steps")
            return self.optimize_basis_single_molecule_list_of_steps
        elif isinstance(self.molecule, list) and isinstance(self.step, list):
            print("optimizing basis for list of molecules but input is a list of steps")
            return self.optimize_basis_list_of_molecules_list_of_steps
        else:
            raise ValueError("input is not valid")

    def _convert_basis(self, opt_bparams, bpacker):
        """
        convert Torch.tensor to scf  basis
        :return: list
        """
        opt_basis = bconv(opt_bparams, bpacker)
        return opt_basis

    def calc_optbasis_energy(self, basis, outpath: str = "./"):
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
        mol.atom = self.atomstruc
        mol.unit = 'Bohr'  # in Angstrom
        mol.verbose = 6
        if self.molecule is not None:
            if os.path.exists("./output"):
                # try to save it in outputfolder
                savep = f'./output/scf_optB_{self.molecule}.out'

                exist = os.path.exists(savep)
                if exist:
                    it = 1
                    while exist:  # if you run multipile calc take care to not overwrite the old output files

                        exist = os.path.exists(f"{outpath}/scf_optB_{self.molecule}_00{it}.out") \
                                or os.path.exists(f"{outpath}/scf_optB_{self.molecule}_0{it}.out") \
                                or os.path.exists(f"{outpath}/scf_optB_{self.molecule}_{it}.out")
                        if exist:
                            it += 1
                        else:
                            if it < 10:
                                mol.output = f"{outpath}/scf_optB_{self.molecule}_00{it}.out"
                            elif it >= 10 and it < 100:
                                mol.output = f"{outpath}/scf_optB_{self.molecule}_0{it}.out"
                            else:
                                mol.output = f"{outpath}/scf_optB_{self.molecule}_{it}.out"

                else:
                    mol.output = savep
            else:
                mol.output = f'scf_optB_{self.molecule}.out'  # save in home folder
        else:
            mol.output = "scf.out"

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

    def _setup_output_path(self):
        """
        setup output path and writer path for tensorboard and output files
        :return: str path to output folder and str path to tensorboard folder
        """
        return conf_output(self.basis, self.basis_ref, self.molecule, outf=self.output_path, **self.out_kwargs)

    def _get_mol_setup(self):
        """
        setup molecule object and dictionary of kwargs for minimized function
        :return: Molecule object of each basis and dict of args for function that gets minimized

        """
        bsys1, bsys2, func_dict = Mole_minimizer(self.basis, self.basis_ref, self.molecule, **self.mol_kwargs)
        bsys1.get_SCF(atomstrucstr=self.molecule)
        energy_small_basis = bsys1.SCF.get_tot_energy
        energy_ref_basis = bsys2.SCF.get_tot_energy

        return energy_small_basis, energy_ref_basis, func_dict

    def _save_out(self, energy_small_basis, energy_ref_basis, func_dict, *args,**kwargs):

        optbasis = bconv(self.opt_bparams, func_dict["bpacker"])
        optbasis_energy = self.calc_optbasis_energy(optbasis, self.outpath)

        _dict = {"outdir" : self.outpath,
                    "b1": self.basis,
                   "b1_energy" : energy_small_basis,
                   "b2": self.basis_ref,
                   "b2_energy" : energy_ref_basis,
                   "optbasis" : optbasis,
                   "optbasis_energy" : optbasis_energy,
                   "atomstruc" :self.molecule,
                   "lr" : self.step,
                   "maxiter" : self.maxiter,
                   "miniter" : self.miniter,
                   "method" : self.method,
                   "packer" : func_dict["bpacker"],
                   "misc" : self.misc,
                   "optkwargs": self.minimize_kwargs}

        for val in kwargs.keys():
            if val in _dict.keys():
                _dict[val] = kwargs[val]

        if kwargs["step"]:
            _dict["lr"] = kwargs["step"]


        save_output(**_dict)
        return self


    def run(self, func_dict, *args,**kwargs):
        """
        run the optimization
        :return: torch.Tensor of optimized basis and dict of misc values
        """
        print("running optimization")
        min_dict = {"step" : self.step,
                    "method" : self.method,
                    "maxiter" : self.maxiter,
                    "miniter" :self.miniter,
                    "verbose" : True,
                    "writer" : self.writer,
                    "diverge" : self.diverge,
                    "maxdivattempts" : self.maxdivattempts,
                    "get_misc":self.get_misc,
                    **self.minimize_kwargs}

        for val in kwargs.keys():
            if val in min_dict.keys():
                min_dict[val] = kwargs[val]
            # realy simple but it just works u know? ;)
            # btw hab die Patente A, B, C und die 6 ich fahr die großen Pödde.

        min_bparams =  xitorch.optimize.minimize(projection,
                                        func_dict["bparams"],
                                        (func_dict["bpacker"],
                                         func_dict["ref_basis"],
                                         func_dict["atomstruc_dqc"],
                                         func_dict["atomstruc"],
                                         func_dict["coeffM"],
                                         func_dict["occ_scf"],
                                         func_dict["num_gauss"],),
                                        **min_dict)

        if self.get_misc:
            self.opt_bparams = min_bparams[0]
            self.misc = min_bparams[1]
            return min_bparams[0], min_bparams[1]
        else:
            self.opt_bparams = min_bparams
            return min_bparams


    def optimize_basis(self, save_out=True):

        energy_small_basis, energy_ref_basis, func_dict = self._get_mol_setup()


        if isinstance(self.step, list):
            for i in self.step:
                self.run(func_dict, step = i)
                if save_out:
                    self._save_out(energy_small_basis, energy_ref_basis, func_dict, step = i)

        else:
            self.run(func_dict, step = self.step)
            if save_out:
                self._save_out(energy_small_basis, energy_ref_basis, func_dict, step = self.step)

        return self



    def __repr__(self):
        return f"OPTBASIS(basis={self.basis}, basis_ref={self.basis_ref}, molecule={self.molecule}, step={self.step}," \
               f" maxiter={self.maxiter}, miniter={self.miniter}, method={self.method}, diverge={self.diverge}," \
               f" maxdivattempts={self.maxdivattempts}, output_path={self.output_path}, get_misc={self.get_misc}," \
               f" mol_kwargs={self.mol_kwargs}, minimize_kwargs={self.minimize_kwargs}, out_kwargs={self.out_kwargs})"


def optimize_basis(basis: str,
                   basis_ref : str,
                   atomstruc : Union[str, list],
                   step:  float,
                   maxiter : int = 100000000,
                   miniter : int = 1 ,
                   method: str = "Adam",
                   diverge = -1.0 ,
                   maxdivattempts = 50,
                   output_path = None,
                   get_misc = True,
                   mol_kwargs: dict = {},
                   minimize_kwargs: dict = {},
                   out_kwargs : dict = {}):
    """
    function to optimize basis functions. It uses the xitorch.optimize.minimize function.
    be aware that the basis functions are optimized in the small basis set.
    :param basis: str basis which should be optimized
    :param basis_ref: str reference basis
    :param atomstruc: Molecule Structure
    :param step: floats with the lr size for the optimization
    :param maxiter: int maximum number of iterations
    :param miniter: int minimum number of iterations
    :param method: str method for optimization
    :param diverge: float diverge value
    :param maxdivattempts: int maximum number of divergence attempts
    :param output_path: str path to output folder
    :param get_misc: bool if True the misc dictionary is returned
    :param mol_kwargs: dict kwargs for Molecule class
    :param minimize_kwargs: dict kwargs for minimize function
    :param out_kwargs: dict kwargs for save_output function
    :return:
    """

    print(f"\n start optimization of {basis} Basis for the Molecule {atomstruc} and learning-rate {step}")

    bsys1, bsys2, func_dict = Mole_minimizer(basis, basis_ref, atomstruc, **mol_kwargs)

    writerpath, outpath = conf_output(basis, basis_ref, atomstruc,outf=output_path, **out_kwargs)

    writer = SummaryWriter(writerpath)



    if get_misc:
        min_bparams, misc = xitorch.optimize.minimize(projection,
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
                                                      miniter=miniter,
                                                      verbose=True,
                                                      writer=writer,
                                                      diverge=diverge,
                                                      maxdivattempts=maxdivattempts,
                                                      get_misc=get_misc,
                                                      **minimize_kwargs)
    else:
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
                                                      miniter=miniter,
                                                      verbose=True,
                                                      writer=writer,
                                                      diverge=diverge,
                                                      maxdivattempts=maxdivattempts,
                                                      get_misc=get_misc,
                                                      **minimize_kwargs)

    print(min_bparams)
    bsys1.get_SCF(atomstrucstr=atomstruc)
    energy_small_basis = bsys1.SCF.get_tot_energy
    energy_ref_basis = bsys2.SCF.get_tot_energy

    optbasis = bconv(min_bparams, func_dict["bpacker"])
    optbasis_energy = scf_dft_energy(optbasis, bsys1.atomstruc, atomstruc, outpath)

    save_output(outpath, basis, energy_small_basis, basis_ref, energy_ref_basis, optbasis, optbasis_energy,
                atomstruc, step, maxiter, miniter=miniter, method=method,
                packer=func_dict["bpacker"], misc=misc, optkwargs=minimize_kwargs)


def opt_basis_mult_steps(basis: str,
                   basis_ref : str,
                   atomstruc : Union[str, list],
                   step:  float,
                   maxiter : int = 100000000,
                   miniter : int = 1 ,
                   method: str = "Adam",
                   diverge = -1.0 ,
                   maxdivattempts = 50,
                   output_path = None,
                   get_misc = True,
                   mol_kwargs: dict = {},
                   minimize_kwargs: dict = {},
                   out_kwargs : dict = {}):
    """
    Optimize basis for multiple learning rates.
    Be aware that this function done in optimization for each learning rate in serial.
    """
    for i in step:
        optimize_basis(basis, basis_ref, atomstruc, i, maxiter, miniter, method,
                       diverge, maxdivattempts, output_path, get_misc, mol_kwargs, minimize_kwargs, out_kwargs)


def mult_atomstruc(basis: str,
                   basis_ref : str,
                   atomstruc : Union[str, list],
                   step:  float,
                   maxiter : int = 100000000,
                   miniter : int = 1 ,
                   method: str = "Adam",
                   diverge = -1.0 ,
                   maxdivattempts = 50,
                   output_path = None,
                   get_misc = True,
                   mol_kwargs: dict = {},
                   minimize_kwargs: dict = {},
                   out_kwargs : dict = {}):
    """
    Optimize basis for multiple atom structures
    Be aware that this function done in optimization for each atom structure in serial.
    """
    for i in atomstruc:
        optimize_basis(basis, basis_ref, i, step, maxiter, miniter, method,
                       diverge, maxdivattempts, output_path, get_misc, mol_kwargs, minimize_kwargs, out_kwargs)










