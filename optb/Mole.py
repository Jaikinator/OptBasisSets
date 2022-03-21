"""
File that stores class Obj to define a Molecular System
"""

from dataclasses import dataclass, field

import xitorch as xt

from optb.MoleDQC import *
from optb.MoleSCF import *
from optb.AtomsDB import *

@dataclass
class Molecarrier:
    basis: str
    atomstruc: list
    def _get_dqc(self, **kwargs):
        return MoleDQC(self.basis, self.atomstruc, **kwargs)

    def _get_scf(self,atomstrucstr= None,  **kwargs):
        return MoleSCF(self.basis, self.atomstruc, atomstrucstr= atomstrucstr,**kwargs)

class Mole(Molecarrier):
    def __init__(self, basis: str, atomstruc: Union[list, str], db=None, scf=True, requires_grad=True, rearrange=True,
                 preselected=True):
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

        if type(atomstruc) is list:
            self.atomstruc = atomstruc

            elementsarr = get_element_arr(atomstruc)

            if self.scf:
                self.SCF = self._get_scf(atomstrucstr = None,elementsarr = elementsarr)

            self.DQC = self._get_dqc(elementsarr = elementsarr,
                                    requires_grad = requires_grad, rearrange = rearrange)

        elif type(atomstruc) is str:

            self._atomstrucstr = atomstruc
            self._db = db

            mole = loadatomstruc(self._atomstrucstr, db = db, preselected = preselected)

            self.molecule = mole.molecule
            self.atomstruc = mole.atomstruc

            self.elementsarr = get_element_arr(self.atomstruc)

            if self.scf:
                self.SCF = self._get_scf(atomstrucstr = self._atomstrucstr, elementsarr=self.elementsarr)

            self.DQC = self._get_dqc(elementsarr=self.elementsarr,
                                    requires_grad=requires_grad, rearrange=rearrange)

    def get_SCF(self,atomstrucstr = None, **kwargs):
        """
        rerun scf calculation and ad it to Mole
        """
        self.SCF = self._get_scf(atomstrucstr = atomstrucstr,elementsarr=self.elementsarr,**kwargs)
        return self

def Mole_minimizer(basis, ref_basis, atomstruc, **kwargs):
    """
    Function that creates a dict with all relevant information to pass throw to the dqc optimizer
    :param basis: the basis you want to optimize
    :param ref_basis: reference basis
    :param atomstruc: molecular structure (atom position)
    :return: dict
    """

    systopt = Mole(basis, atomstruc, scf = False,**kwargs)  # System of Basis to optimize

    sys_ref = Mole(ref_basis, atomstruc, requires_grad = False, **kwargs)

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