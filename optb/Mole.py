"""
File that stores class Obj to define a Molecular System
"""

from pyscf import gto, scf

from dataclasses import dataclass, field
from typing import Union
import warnings  # for warnings
import os

import dqc
import torch
import xitorch as xt
from dqc.utils.datastruct import AtomCGTOBasis
import dqc.hamilton.intor as intor
from dqc.api.parser import parse_moldesc

import basis_set_exchange as bse  # basest set exchange library


from optb.data.w417db import *
import optb.data.avdata as avdata
import optb.data.preselected_avdata as presel_avdata

# get atom pos:
from ase.build import molecule as asemolecule

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

        thispath = os.path.dirname(os.path.realpath(__file__))
        dbpath = os.path.join(thispath, "data/.database")

        if os.path.exists(dbpath) == False:  # check if NWChemBasis or data folder exist. if not it will be created
            warnings.warn("No .database folder exist it will be created.")
            os.mkdir(os.path.abspath(dbpath))
            print(f" folder is created to {os.path.abspath(dbpath)}")

        def _normalize_basisname(basisname: str) -> str:
            b = basisname.lower()
            b = b.replace("+", "p")
            b = b.replace("*", "s")
            b = b.replace("(", "_")
            b = b.replace(")", "_")
            b = b.replace(",", "_")
            return b

        bdict = {}

        for i in range(len(self.elements)):  # assign the basis to the associated elements
            if isinstance(self.basis, str):
                basisname = _normalize_basisname(self.basis)
            else:
                basisname = _normalize_basisname(self.basis[el_dict[self.elements[i]]])
                # takes care if basis is not str
                # instead it can be dict

            basisfolder = os.path.join(dbpath, basisname)

            if not os.path.exists(basisfolder):
                # make a folder for each basis.
                # in this folder the elements for each basis will be stored.
                os.mkdir(os.path.abspath(basisfolder))

            if not os.path.exists(f"{basisfolder}/{basisname}.{self.elements[i]}.nw"):
                # check if basis file already exists
                # if False it will be downloaded and stored to NWChemBasis folder
                print(f"No basis {self.basis} found for {el_dict[self.elements[i]]}."
                      f" Try to get it from https://www.basissetexchange.org/")
                basis = bse.get_basis(self.basis, elements=[el_dict[self.elements[i]]], fmt="nwchem")
                fname = f"{basisfolder}/{basisname}.{self.elements[i]}.nw"
                with open(fname, "w") as f:
                    f.write(basis)
                f.close()
                print(f"Downloaded to {os.path.abspath(fname)}")

            file = open(f"{basisfolder}/{basisname}.{self.elements[i]}.nw").read()
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


@dataclass
class Molecarrier:
    basis: str
    atomstruc: list

    def get_dqc(self, **kwargs):
        return MoleDQC(self.basis, self.atomstruc, **kwargs)

    def get_scf(self, **kwargs):
        return MoleSCF(self.basis, self.atomstruc, **kwargs)

@dataclass
class AtomsDB:
    """
    Class to store Data form Databases
    """
    atomstrucstr: str
    atomstruc : list
    mult: int
    charge: int
    energy : float
    molecule : object = field(repr= False)

def loadatomstruc(atomstrucstr: Union[list, str],db = None,preselected = True, owndb = False):
    """
    load molecule informations from a given DB and pass it throw to the AtomsDB class
    The supportet databasis are g2 and w4-17
    :param atomstrucstr: string of molecule like CH4 H2O etc.
    :param db: select specific
    :param preselected: load preselected Database files where multiplicity is 1 and charge is 0
    :param owndb: if you want to use your own db
    :return: AtomsDB
    """
    def _create_atomstruc_from_ase(molec):
        """
        creates atomstruc from ase database.
        :param atomstruc: molecule string
        :return: array like [element, [x,y,z], ...]
        """
        chem_symb = molec.get_chemical_symbols()
        atompos = molec.get_positions()

        arr = []
        for i in range(len(chem_symb)):
            arr.append([chem_symb[i], list(atompos[i])])
        return arr

    def from_W417(atomstrucstr):
        molec = W417(atomstrucstr)
        return AtomsDB(atomstrucstr, molec.atom_pos, molec.mult, molec.charge,
                       molec.energy, molec.molecule)

    def from_g2(atomstrucstr):
        molec= asemolecule(atomstrucstr)
        atomstruc = _create_atomstruc_from_ase(molec)
        mult = sum(molec.get_initial_magnetic_moments()) + 1
        charge = sum(molec.get_initial_charges())
        energy = None
        return AtomsDB(atomstrucstr, atomstruc, mult, charge, energy, molec)

    if owndb:
        pass #not supported jet
    else:
        if db == None:
            if preselected:
                if atomstrucstr in presel_avdata.elw417:
                    return from_W417(atomstrucstr)

                elif atomstrucstr in presel_avdata.elg2:
                    print("Attention you get your data from the less accurate g2 Database.\n"
                          "NO energy detected")
                    return from_g2(atomstrucstr)

            else:

                if atomstrucstr in avdata.elw417:
                    return from_W417(atomstrucstr)

                elif atomstrucstr in avdata.elg2:
                    print("Attention you get your data from the less accurate g2 Database.\n"
                          "NO energy detected")
                    return from_g2(atomstrucstr)




@dataclass
class AtomstrucASE:
    atomstrucstr: str
    atomstruc: list = field(init=False)
    mult: int = field(init=False)
    charge: int = field(init=False)
    molecule: object = field(init=False, repr=False)
    """
        gets the atom-structure (atom positions, charge, multiplicity) from the ase g2 databases.
        :param atomstrucstr: str like H2O, CH4 etc.

        """

    def __post_init__(self):
        self.molecule = asemolecule(self.atomstrucstr)
        self.atomstruc = self._create_atomstruc_from_ase()
        self.mult = sum(self.molecule.get_initial_magnetic_moments()) + 1
        self.charge = sum(self.molecule.get_initial_charges())

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
    def get_ase_molecule(self):
        """
            :return: molecule object of ase
            """

    @property
    def get_charge(self):
        """
            :return: molecular charge
            """
        return self.charge

    @property
    def get_mult(self):
        """
            :return: multiplicity
            """
        return self.mult


@dataclass
class AtomstrucW417:
    """
    Gets the atom-structure (atom positions, charge, multiplicity) from the W417 database.
    :param atomstrucstr: str like H2O, CH4 etc.
    """
    atomstrucstr: str
    atomstruc: list = field(init=False)
    mult: int = field(init = False)
    charge: int = field(init = False)
    energy : int = field(init = False)
    def __post_init__(self):
        molecule = W417(self.atomstrucstr)
        self.atomstruc = molecule.atom_pos
        self.charge = molecule.charge
        self.energy = molecule.energy
        self.mult = molecule.mult


# @dataclass
# def AtomstrucfDB:
#     atomstrucstr: str
#     db: str = "w4-17"
#     preselected : bool = True
#
#     def __post_init__(self):
#
#         if self.preselected:
#             self._check_preselected()
#
#         db = self._check_Mole_from_DB()
#
#
#     def _check_Mole_from_DB(self):
#         """
#         Try to get atomic positions from database.
#         Therefore check in which Database the module is.
#         :return class object
#         """
#         if self.db == "w4-17":
#             return "w4-17"
#         elif self.db == "g2":
#             return "w4-17"
#         elif self.db:
#             pass
#
#     def _check_preselected(self) -> AtomstrucfDB:
#         """
#         checks if data is in preselected_avdata.py if not raises error
#         """
#         if not (self.atomstrucstr in presel_avdata.elw417 or self.atomstrucstr in presel_avdata.elg2):
#             raise ImportError("The molecule does not have a charge of 0 or the multiplicity of 1."
#                               " If this where intended maybe change MoleDB.preselected to false.")


class Mole(Molecarrier):
    def __init__(self, basis: str, atomstruc, db=None, scf=True, requires_grad=True, rearrange=True, preselected=True):
        super().__init__(basis, atomstruc)
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
        :param preselected: if true you only have access to molecules,
                whose charge is zero as well as which multiplicity is 1.
        :param db: select which database should be loaded
        :param scf: if True you get the optb as dqc as well as scf optb.
        :param requires_grad: support gradient of torch.Tensor
        :param rearrange: if True the dqc basis will be rearranged to match the basis read by scf
        """

        self.scf = scf
        self.basis = basis
        self._atomstrucstr = None
        self.requires_grad = requires_grad
        if type(atomstruc) is list:
            self.atomstruc = atomstruc
            # take care for scf getter
            self.elementsarr = _get_element_arr(atomstruc)

            if self.scf:
                self.SCF = self.get_SCF()

            self.DQC = MoleDQC(basis, self.atomstruc, self.elementsarr, self.requires_grad, rearrange)

        elif type(atomstruc) is str:
            self._atomstrucstr = atomstruc
            self._db = db
            mole = MoleDB(self.basis, self.atomstrucstr, db, scf, requires_grad, rearrange, preselected)
            self.molecule = mole.molecule
            self.atomstruc = mole.atomstruc

            if self.scf:
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

    systopt = Mole(basis, atomstruc, scf=False)  # optb to optimize

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


if __name__ == "__main__":
    basis = "STO-3G"

    atomstruc = [['H', [0.5, 0.0, 0.0]],
                 ['H', [-0.5, 0.0, 0.0]]]

    print(loadatomstuc("CH4"))

