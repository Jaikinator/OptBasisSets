"""
File that stores class Obj to define a Molecular System
"""

from pyscf import gto, scf
import dqc
import torch
import xitorch as xt
from dqc.utils.datastruct import AtomCGTOBasis
import dqc.hamilton.intor as intor
from dqc.api.parser import parse_moldesc
import warnings  # for warnings
import os
import basis_set_exchange as bse  # basest set exchange library
from optb.data.w417db import *
from optb.data.avdata import *

# get atom pos:
from ase.build import molecule

from optb.params_periodic_system import el_dict  # contains dict with all numbers and Symbols of the periodic table


def _get_element_arr(atomstruc):
    """
    create array with all elements in the optb
    """
    elements_arr = [atomstruc[i][0] for i in range(len(atomstruc))]
    for i in range(len(elements_arr)):
        if type(elements_arr[i]) is str:
            elements_arr[i] = el_dict[elements_arr[i]]
    return elements_arr


class MoleSCF:
    def __init__(self, basis: str, atomstruc: list, elementsarr=None):
        """
        MoleSCF provides all relevant data for the basis optimization that refers to pyscf
        A mol type Object will be created as well as a restricted kohn sham
        :param basis: str (like: STO-3G)
        :param atomstruc: list with all elements on their specific positions. (example below)
                            [['H', [0.5, 0.0, 0.0]],
                             ['H',  [-0.5, 0.0, 0.0]],
                             ['He', [0.0, 0.5, 0.0]]]
        :param elementsarr: provides information witch elements are in the system eg. H takes element number 1
        """

        self.basis = basis  # just str of basis
        self.atomstruc = atomstruc

        if elementsarr == None:
            self.elements = _get_element_arr(self.atomstruc)
        else:
            if type(elementsarr) is list:
                self.elements = elementsarr
            else:
                warnings.warn("elementsarr type is not list or None")

        self.basis = basis  # just str of basis
        self.atomstruc = atomstruc

        self.molbasis = None
        self.xc = "B3LYP"
        self.mol = self._create_Mol()
        self.dft = self.dft_calc()

    def _get_molbasis_fparser(self):
        """
        function to override scf basis function with data from:
        https://www.basissetexchange.org/
        If the file doesn't exist it will be downloaded in the NWChem format.
        :return: mol.basis object
        """
        print(os.path.dirname(os.path.realpath(__file__)))
        folderpath = os.path.realpath("data/NWChemBasis")

        if os.path.exists(folderpath) == False:  # check if NWChemBasis or data folder exist. if not it will be created
            warnings.warn("No NWChemBasis or data folder exist it will be created.")
            os.mkdir(os.path.abspath(folderpath))
            fullpath = os.path.abspath(folderpath)
            print(f"NWChemBasis folder is created to {fullpath}")

        bdict = {}

        for i in range(len(self.elements)):  # assign the basis to the associated elements
            if isinstance(self.basis, str):
                basisname = self.basis
            else:
                basisname = self.basis[el_dict[self.elements[i]]]
                # takes care if basis is not str
                # instead it can be dict
                # not working right now

            if not os.path.exists(f"{folderpath}/{basisname}.{self.elements[i]}.nw"):
                # check if basis file already exists
                # if False it will be downloaded and stored to NWChemBasis folder
                print(f"No basis {self.basis} found for {el_dict[self.elements[i]]}."
                      f" Try to get it from https://www.basissetexchange.org/")
                basis = bse.get_basis(self.basis, elements=[el_dict[self.elements[i]]], fmt="nwchem")
                fname = f"{folderpath}/{basisname}.{self.elements[i]}.nw"
                with open(fname, "w") as f:
                    f.write(basis)
                f.close()
                print(f"Downloaded to {os.path.abspath(fname)}")

            file = open(f"{folderpath}/{basisname}.{self.elements[i]}.nw").read()
            bdict[str(el_dict[self.elements[i]])] = gto.basis.parse(file, optimize=False)
        return bdict

    def _create_Mol(self, **kwargs):
        """
        pyscf mol object will be created.
        be aware here just basis as a string works
        :return: mol object
        """
        mol = gto.Mole()
        mol.atom = self.atomstruc
        mol.basis = self.molbasis
        mol.unit = 'Bohr'  # in Angstrom

        if "verbose" in kwargs:
            mol.verbose = kwargs["verbose"]
        else:
            mol.verbose = 6
        if "symmetry" in kwargs:
            mol.symmetry = kwargs["symmetry"]
        else:
            mol.symmetry = False
        mol.output = 'scf.out'

        # I don't know what I'am doing here:
        try:
            mol.spin = 0
            return mol.build()
        except:
            mol.spin = 1
            return mol.build()

    def dft_calc(self):
        """
        runs the restricted kohn sham using the  b3lyp hybrid functional
        you can get all informations of the solved system by using the particular pyscf functions.
        :return: pyscf.dft.rks.RKS object
        """
        mf = scf.RKS(self.mol)
        mf.xc = self.xc
        mf.kernel()
        return mf

    @property
    def molbasis(self):
        """
        getter for molbasis obj.
        """
        return self._molbasis

    @molbasis.setter
    def molbasis(self, var=None):
        """
        setter for molbasis
        :param var: is none basis is going to be created from NWCHem file
        :return: basis dict
        """
        if var == None:
            self._molbasis = self._get_molbasis_fparser()
        else:
            self._molbasis = var

    @property
    def get_basis(self):
        """
        :return: dict of all basis for the particular elements
        """
        return self._get_molbasis_fparser()

    @property
    def get_coeff(self):
        """
        :return: torch.Tensor, coefficient matrix from the dft calculation.
        """
        return torch.tensor(self.dft.mo_coeff)

    @property
    def get_occ_coeff(self):
        """
         coefficient- matrix of just the occupied orbitals
        """
        return self.get_coeff[self.get_occ > 0]

    @property
    def get_mol(self):
        """
        :return: pyscf mole object
        """
        return self._create_Mol()

    @property
    def get_occ(self):
        """
        :return: occupied orbitals
        """
        return torch.tensor(self.dft.get_occ())

    @property
    def get_ovlp(self):
        """
        :return: overlap matrix of a given mol object
        """
        return torch.Tensor(self.mol.get_ovlp()).type(torch.float64)

    @property
    def get_tot_energy(self):
        """
        :return: float: total energy of the system calculated throw the dft calculation.
        """
        return self.dft.energy_tot()


class MoleDQC:
    def __init__(self, basis: str, atomstruc: list, elementsarr=None, rearrange=True, requires_grad=True):
        """
           MoleDQC provides all relevant data for the basis optimization that refers to dqc
           :param basis: str (like: STO-3G)
           :param atomstruc: list with all elements on their specific positions. (example below)
                                [['H', [0.5, 0.0, 0.0]],
                                 ['H',  [-0.5, 0.0, 0.0]],
                                 ['He', [0.0, 0.5, 0.0]]]
           :param elementsarr: provides information witch elements are in the system eg. H takes element number 1
           :param rearrange: rearranges the dqc basis to be equal as one of pyscf.
           :param requires_grad: if true gradiant can be obtained to optimize basis.
        """
        self.basis = basis  # just str of basis
        self.atomstruc = atomstruc
        self.atomstruc_dqc = self._arr_int_conv()

        if elementsarr == None:
            self.elements = _get_element_arr(self.atomstruc)
        else:
            if type(elementsarr) is list:
                self.elements = elementsarr
            else:
                warnings.warn("elementsarr type is not list or None")

        if rearrange == False:
            if requires_grad == False:
                basisparam = self._loadbasis(requires_grad=False)  # loaded dqc basis
            else:
                basisparam = self._loadbasis()
        else:
            if requires_grad == False:
                basisparam = self._rearrange_basis(requires_grad=False)  # loaded dqc basis
            else:
                basisparam = self._rearrange_basis()

        self.lbasis = basisparam  # basis with parameters in dqc format

    def _arr_int_conv(self):
        """
        converts the atomsruc array to an str for dqc
        :return: str
        """
        return (str(self.atomstruc).replace("[", " ")
                .replace("]", " ")
                .replace("'", "")
                .replace(" , ", ";")
                .replace(",", "")
                .replace("  ", " "))

    def _loadbasis(self, **kwargs):
        """
        load basis from basissetexchange for the dqc module:
        https://www.basissetexchange.org/

        data of basis sets ist stored:
        /home/user/anaconda3/envs/env/lib/python3.8/site-packages/dqc/api/.database

        :param kwargs: requires_grad=False if basis is reference basis
        :return: array of basis
        """

        if type(self.basis) is str:
            bdict = {}
            for i in range(len(self.elements)):
                bdict[el_dict[self.elements[i]]] = dqc.loadbasis(f"{self.elements[i]}:{self.basis}", **kwargs)
            return bdict
        elif type(self.basis) is dict:
            bdict = {}
            for i in range(len(self.basis)):
                bdict[el_dict[self.elements[i]]] = dqc.loadbasis(
                    f"{self.elements[i]}:{self.basis[el_dict[self.elements[i]]]}", **kwargs)
            return bdict
        else:
            print("do nothing to load basis")
            # basis = [dqc.loadbasis(f"{self.elements[i]}:{self.basis[i]}", **kwargs) for i in range(len(self.elements))]
            # return basis

    def _rearrange_basis(self, **kwargs):
        """
        rearrange the sequence of dqc a basis to one that is used by pyscf.
        This is relevant to get the same overlap between dqc and pyscf.
        :param kwargs: kwargs for load basis
        :return: dict
        """
        basis = self._loadbasis(**kwargs)
        bout = {}

        # read highest angmom for each atom:
        max_angmom_arr = []

        for key in basis:
            max_angmom_atom_arr = []
            for j in range(len(basis[key])):
                max_angmom_atom_arr.append(basis[key][j].angmom)
            max_angmom_arr.append(max(max_angmom_atom_arr))

        # now rearrange basis:
        cf_angmon = 0  # counter for angmom for each atom
        for key in basis:
            angmomc = 0
            inner = []

            while angmomc <= max_angmom_arr[cf_angmon]:
                for j in range(len(basis[key])):
                    if basis[key][j].angmom == angmomc:
                        inner.append(basis[key][j])
                angmomc += 1

            cf_angmon += 1
            bout[key] = inner

        warnings.warn("ATTENTION basis of dqc is rearranged")
        return bout

    def _get_ovlp(self):
        """
        :return: overlap matrix calc by dqc
        """
        elem = [self.atomstruc[i][0] for i in range(len(self.atomstruc))]
        basis = [self.lbasis[elem[i]] for i in range(len(elem))]
        atomzs, atompos = parse_moldesc(self.atomstruc_dqc)
        atombases = [AtomCGTOBasis(atomz=atomzs[i], bases=basis[i], pos=atompos[i]) for i in range(len(basis))]
        wrap = dqc.hamilton.intor.LibcintWrapper(
            atombases)  # creates a wrapper object to pass information on lower functions

        return intor.overlap(wrap)

    @property
    def get_atomstruc(self):
        """
        :return: get atom-structure in dqc format
        """
        return self.atomstruc_dqc

    @property
    def get_basis(self):
        """
        :return: get basis in dqc format
        """
        return self.lbasis

    @property
    def get_ovlp(self):
        """
        :return: overlap matrix calc by dqc
        """
        return self._get_ovlp()


class Mole_ase:
    def __init__(self, basis: str, atomstrucstr: str, scf=True, requires_grad=False, rearrange=True):
        """
        Same as Mole just gets the atom-structure (atom positions, charge, multiplicity) from the ase g2 databases.
        :param atomstrucstr: str like H2O, CH4 etc.

        """
        self.molecule = molecule(atomstrucstr)
        self.atomstruc = self._create_atomstruc_from_ase()

        elementsarr = _get_element_arr(self.atomstruc)

        if scf:
            self.SCF = MoleSCF(basis, self.atomstruc, elementsarr)
        self.DQC = MoleDQC(basis, self.atomstruc, elementsarr, requires_grad, rearrange)

    def _create_atomstruc_from_ase(self):
        """
        creates atomstruc from ase database.
        :param atomstruc: molecule string
        :return: array like [element, [x,y,z], ...]
        """
        chem_symb = self.molecule.get_chemical_symbols()
        atompos = self.molecule.get_positions()

        arr = []
        for i in range(len(chem_symb)):
            arr.append([chem_symb[i], list(atompos[i])])
        return arr

    @property
    def get_charge(self):
        """
        :return: molecular charge
        """
        return sum(self.molecule.get_initial_charges())

    @property
    def get_mult(self):
        """
        :return: multiplicity
        """
        return sum(self.molecule.get_initial_magnetic_moments()) + 1


class Mole_w417:
    def __init__(self, basis: str, atomstrucstr: str, scf=True, requires_grad=False, rearrange=True):
        """
        Same as Mole just gets the atom-structure (atom positions, charge, multiplicity) from the ase g2 databases.
        :param atomstrucstr: str like H2O, CH4 etc.

        """
        self.molecule = W417(atomstrucstr)
        self.atomstruc = self.molecule.atom_pos

        elementsarr = _get_element_arr(self.atomstruc)

        if scf:
            self.SCF = MoleSCF(basis, self.atomstruc, elementsarr)
        self.DQC = MoleDQC(basis, self.atomstruc, elementsarr, requires_grad, rearrange)


class MoleDB:
    def __init__(self, basis: str, atomstrucstr: str, db=None, scf=True, requires_grad=False, rearrange=True):

        self.atomstrucstr = atomstrucstr
        self.db = db

        db = self._check_Mole_from_DB()

        if db == "w4-17":
            mole = Mole_w417(basis, self.atomstrucstr, scf, requires_grad, rearrange)
            self.molecule = mole.molecule
        elif db == "g2":
            mole = Mole_ase(basis, self.atomstrucstr, scf, requires_grad, rearrange)
            self.molecule = mole

        self.atomstruc = mole.atomstruc

        if scf:
            self.SCF = mole.SCF

        self.DQC = mole.DQC

    def _check_Mole_from_DB(self):
        """
        Try to get atomic positions from database.
        Therefore check in which Database the module is.
        :return class object
        """
        if self.db == None:
            if self.atomstrucstr in elw417:
                return "w4-17"
            elif self.atomstrucstr in elg2:
                print("pls notice that you get Data from the less accurate g2 Database.")
                return "g2"
            else:
                raise ImportError("Molecule not found in g2 or w4-17 Database")
        elif self.db == "w4-17":
            return "w4-17"
        elif self.db == "g2":
            return "w4-17"


class Mole:
    def __init__(self, basis: str, atomstruc, db=None, scf=True, requires_grad=True, rearrange=True):
        """
        class to define the systems to optimize.
        Therefore, it creates a MoleSCF and a MoleDQC Object.

        :param basis: name of the basis to optimize if you want to use multiple basis do something like
                      basis = ["basis_1", "basis_2",...,"basis_N"] the array has to be the same length as atomstruc
        :param atomstruc: structure of the optb input like
                          array or str to set or load atom positions from array or database
                            array([[element, [position]],
                                  [element , [position]],
                                  ...])
                            therefore position has to be the length 3 with float number for each axis position in
                            cartesian space. For example pos = [1.0, 1.0, 1.0]
        :param db: select which database should be loaded
        :param scf: if True you get the optb as dqc as well as scf optb.
        :param requires_grad: support gradient of torch.Tensor
        :param rearrange: if True the dqc basis will be rearranged to match the basis read by scf
        """

        if type(atomstruc) is list:
            self.atomstruc = atomstruc

            elementsarr = _get_element_arr(atomstruc)

            if scf:
                self.SCF = MoleSCF(basis, self.atomstruc, elementsarr)

            self.DQC = MoleDQC(basis, self.atomstruc, elementsarr, requires_grad, rearrange)
        elif type(atomstruc) is str:
            mole = MoleDB(basis, atomstruc, db, scf, requires_grad, rearrange)
            self.molecule = mole.molecule
            self.atomstruc = mole.atomstruc
            if scf:
                self.SCF = mole.SCF

            self.DQC = mole.DQC


def Mole_minimizer(basis, ref_basis, atomstruc):
    """
    Function that creates a dict with all relevant information to pass throw to the dqc optimizer
    :param basis: the basis you want to optimize
    :param ref_basis: reference basis
    :param atomstruc: molecular structure (atom position)
    :return: dict
    """

    systopt = Mole(basis, atomstruc, scf = False)  # optb to optimize

    sys_ref = Mole(ref_basis, atomstruc, requires_grad=False)

    def _num_gauss(system, ref_system):
        """
        calc the number of primitive Gaussian's in a basis set so that the elements of an overlap matrix can be defined.
        :param system: class type
            basis to get optimized
        :param ref_system: class type
            optimized basis
        :return: int
            number of elements of each basis set

        """
        n_basis = 0
        n_restbasis = 0

        for i in range(len(system.atomstruc)):
            for el in system.DQC.lbasis[system.atomstruc[i][0]]:
                n_basis += 2 * el.angmom + 1

            for el in ref_system.DQC.lbasis[ref_system.atomstruc[i][0]]:
                n_restbasis += 2 * el.angmom + 1
        return torch.tensor([n_basis, n_restbasis])

    def system_dict(system, ref_system):
        """
        def dictionary to input in fcn
        :param system: class obj for the basis you want to optimize
        :param ref_system: class obj for the reference basis
        :return: dict
        """

        basis = system.DQC.lbasis
        bpacker = xt.Packer(basis)
        bparams = bpacker.get_param_tensor()

        basis_ref = ref_system.DQC.lbasis
        bpacker_ref = xt.Packer(basis_ref)
        bparams_ref = bpacker_ref.get_param_tensor()
        ref_basis = bpacker_ref.construct_from_tensor(bparams_ref)

        return {"bparams": bparams,
                "bpacker": bpacker,
                "ref_basis": ref_basis,
                "atomstruc_dqc": system.DQC.atomstruc_dqc,
                "atomstruc": system.DQC.atomstruc,
                "coeffM": ref_system.SCF.get_coeff,
                "occ_scf": ref_system.SCF.get_occ,
                "num_gauss": _num_gauss(system, ref_system)}

    return systopt, sys_ref, system_dict(systopt, sys_ref)
