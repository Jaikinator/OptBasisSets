"""
File that stores class Obj to define a Molecular System
"""

from pyscf import gto, scf
import warnings  # for warnings
import os
import torch
import basis_set_exchange as bse  # basest set exchange library
from optb.get_element_arr import *


class MoleSCF:
    def __init__(self, basis: str, atomstruc: list, elementsarr=None, atomstrucstr = None ):
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
        self.atomstrucstr = atomstrucstr

        if elementsarr is None:
            self.elements = get_element_arr(self.atomstruc)
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

        if self.atomstrucstr is not None:
            if os.path.exists("./output"):
                mol.output = f'./output/scf_{self.basis}_{self.atomstrucstr}.out'  # try to save it in outputfolder
            else:
                mol.output = f'scf_{self.basis}_{self.atomstrucstr}.out' # save in home folder

        else:
            mol.output = f'scf_{self.basis}_{self.atomstruc}.out'

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
    def get_mo_energy(self):
        """
        :return: torch.Tensor, molecular orbital energies
        """
        return torch.tensor(self.dft.mo_energy)
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


