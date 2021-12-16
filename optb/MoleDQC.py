"""
def the MoleDQC Class which handles the DQC stuff
"""
import warnings  # for warnings

import dqc
from dqc.utils.datastruct import AtomCGTOBasis
import dqc.hamilton.intor as intor
from dqc.api.parser import parse_moldesc

from optb.get_element_arr import *

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
            self.elements = get_element_arr(self.atomstruc)
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
