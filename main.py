from pyscf import gto, scf
from xitorch._core.pure_function import get_pure_function, make_sibling
import dqc
import numpy as np
import torch
import xitorch as xt
import xitorch.optimize
from dqc.utils.datastruct import AtomCGTOBasis
import dqc.hamilton.intor as intor
from dqc.api.parser import parse_moldesc

import pymatgen as mg
import pymatgen.core.periodic_table as peri

####################################
#check if GPU is used:
# setting device on GPU if available, else CPU
def cuda_device_checker(memory  = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if memory is False:
        print(f"Using device: {device, torch.version.cuda}\n{torch.cuda.get_device_name(0)}")
    else:
        if device.type == 'cuda':
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

cuda_device_checker()
########################################################################################################################
### create a dictionary for the elements and their numbers in the periodic table.
########################################################################################################################

#create first basis set. (reference)
class dft_system:
    def __init__(self, basis, atomstruc):
        """
        class to define the systems to optimize.
        :param element: array, str or number of the element in the periodic table
        :param basis: name of the basis to optimize if you want to use multiple basis do something like
                      basis = ["basis_1", "basis_2",...,"basis_N"] the array has to be the same length as atomstruc
        :param atomstruc: structure of the system input like
                            array([[element, [position]],
                                  [element , [position]],
                                  ...])
                            therefore position has to be the length 3 with float number for each axis position in
                            cartesian space. For example pos = [1.0, 1.0, 1.0]
        """
        self.basis = basis
        self. atomstuc = atomstruc
        self.element_dict = self.generateElementdict()
        self.elements = self.get_element_arr()

        self.mol = self._create_scf_Mol()
    ################################
    #dqc stuff:
    ################################

    def dqc(self):
        if type(self.basis) is str:
            basis = [dqc.loadbasis(f"{self.elements[i]}:{self.basis}") for i in range(len(self.elements))]
        else:
            basis = [dqc.loadbasis(f"{self.elements[i]}:{self.basis[i]}") for i in range(len(self.elements))]
        bpacker = xt.Packer(basis)
        bparams = bpacker.get_param_tensor()
        return {"bpacker" :bpacker,
                "bparams": bparams}

    ################################
    #scf staff:
    ################################

    def _create_scf_Mol(self):
        """
        be aware here just basis as a string works
        :return: mol object
        """
        mol = gto.Mole()
        mol.atom = self.atomstuc
        mol.spin = 0
        mol.unit = 'Bohr'  # in Angstrom
        mol.verbose = 6
        mol.output = 'scf.out'
        mol.symmetry = False
        mol.basis = self.basis
        return mol.build()

    def _coeff_mat_scf(self):
        """
        just creates the coefficiency matrix for different input basis
        """

        mf = scf.RHF(self.mol)
        mf.kernel()
        return torch.tensor(mf.mo_coeff[:, mf.mo_occ > 0.])

    def _get_occ(self):
        mf = scf.RHF(self.mol)
        mf.kernel()
        return torch.tensor(mf.get_occ())

    ################################
    #extra staff to configure class:
    ################################

    def get_element_arr(self):
        elements_arr  = [self.atomstuc[i][0] for i in range(len(atomstruc))]
        for i in range(len(elements_arr)):
            if type(elements_arr[i]) is str:
                elements_arr[i] = self.element_dict[elements_arr[i]]
        return elements_arr

    def generateElementdict(self):
        elementdict = {}
        for i in range(1, 100):  # 100 Elements in Dict
            elementdict[peri.Element.from_Z(i).number] = str(peri.Element.from_Z(i))
            elementdict[ str(peri.Element.from_Z(i))] = peri.Element.from_Z(i).number
        return elementdict




atomstruc = [['H', [1.0, 0.0, 0.0]],
            ['H', [-1.0, 0.0, 0.0]]]
basis = "3-21G"
system = dft_system(basis, atomstruc)
system_dqc = system.dqc()

occ = system._coeff_mat_scf()

########################################################################################################################
# create the second basis set to optimize

rest_basis = [dqc.loadbasis("1:3-21G", requires_grad=False),dqc.loadbasis("1:3-21G", requires_grad=False)] #cc-pvdz
#rest_basis = [dqc.loadbasis("1:cc-pvdz", requires_grad=False)]
bpacker_rest = xt.Packer(rest_basis)
bparams_rest = bpacker_rest.get_param_tensor()

########################################################################################################################
def _num_gauss(basis : list, restbasis : list):
    """
    calc the number of primitive gaussians in a basis set so that the elements of an overlap matrix can be defined.
    :param basis: list
        basis to get opimized
    :param restbasis: list
        optimized basis
    :return: int
        number of elements of each basis set

    """
    n_basis =  0
    n_restbasis = 0

    for b in basis:
        for el in b:
            n_basis += 2 * el.angmom + 1

    for b in restbasis:
        for el in b:
            n_restbasis += 2 * el.angmom + 1

    return [n_basis, n_restbasis]

def _cross_selcet(crossmat : torch.Tensor, num_gauss : list, direction : str ):
    """
    select the cross overlap matrix part.
    The corssoverlap is defined by the overlap between to basis sets.
    For example is b1 the array of the first basis set and b2 the array of the second basisset
    Then is the crossoveralp S:
        S  = [b1*b1 , b1*b2] = [S_11 , S_12]
             [b2*b1 , b2*b2]   [S_21 , S_22]
    you can choose between:
        - basis overlap (S_11)
        - restbasis overlap (S_22)
        - horizontal crossoverlap (S_12)
        - vertical crossoverlap (S_21)
    :param crossmat: crossoverlap mat
    :param num_gauss: number of gaussians in two basis
    :param direction: type str to select the specific crossoverlap
                      you can choose between: S_11, S_12_,S_21, S_22
    :return: torch.tensor
    returns the cross overlap matrix between the new and the old basis func
    """

    if direction == "S_11" :
        return crossmat[0:num_gauss[0],0:num_gauss[0]]
    elif direction == "S_12":
        return crossmat[num_gauss[0]:, 0:(len(crossmat) - num_gauss[1])]
    elif direction == "S_21":
        return crossmat[0:num_gauss[0], num_gauss[0]:]
    elif direction == "S_22":
        return crossmat[num_gauss[0]:, num_gauss[0]:]
    else:
        raise UnicodeError("no direction specified")


def _crossoverlap(atomstruc, basis):
    #calculate cross overlap matrix
    atomzs, atompos = parse_moldesc(atomstruc)
    # atomzs : Atomic number (torch.tensor); len: number of Atoms
    # atompos : atom positions tensor of shape (3x len: number of Atoms )
    # now double both atom informations to use is for the second basis set

    atomzs = torch.cat([atomzs, atomzs])
    atompos = torch.cat([atompos, atompos])

    atombases = [AtomCGTOBasis(atomz=atomzs[i], bases=basis[i], pos=atompos[i]) for i in range(len(basis))]
    # creats an list with AtomCGTOBasis object for each atom (including  all previous informations in one array element)

    wrap = dqc.hamilton.intor.LibcintWrapper(
        atombases)  # creates an wrapper object to pass informations on lower functions
    return intor.overlap(wrap)



def _maximise_overlap(coeff, colap, num_gauss):
    """
    Calculate the Projection from the old to the new Basis:
         P = C^T S_12 S⁻¹_22 S_21 C
    :param coeff: coefficient matrix calc by pyscf (C Matrix in eq.)
    :param colap: crossoverlap Matrix (S and his parts)
    :param num_gauss: array with length of the basis sets
    :return: Projection Matrix
    """

    S_12 = _cross_selcet(colap, num_gauss, "S_12")
    S_21 = _cross_selcet(colap, num_gauss, "S_21")
    S_22 = _cross_selcet(colap, num_gauss, "S_22")
    s21_c = torch.matmul(S_12, coeff)
    s22_s21c = torch.matmul(torch.inverse(S_22), s21_c)
    s12_s22s21c = torch.matmul(S_21, s22_s21c)
    P = torch.matmul(coeff.T, s12_s22s21c)
    return P

########################################################################################################################
#test
########################################################################################################################

def fcn(bparams, bpacker):
    """
    Function to optimize
    :param bparams: torch.tensor with coeff of basis set
    :param bpacker: xitorch._core.packer.Packer
    :return:
    """

    basis = bpacker.construct_from_tensor(bparams) #create a CGTOBasis Object (set informations about gradient, normalization etc)
    rest_basis = bpacker_rest.construct_from_tensor(bparams_rest)
    num_gauss= _num_gauss(basis, rest_basis)
    basis_cross = basis + rest_basis
    atomstruc = "H 1 0 0; H -1 0 0"

    #calculate cross overlap matrix
    colap = _crossoverlap(atomstruc, basis_cross)
    coeff = occ

    # maximize overlap

    projection = _maximise_overlap(coeff,colap,num_gauss)
    projection = projection * system._get_occ()[system._get_occ() > 0]
    #print(projection,sys_scf._get_occ())
    return -torch.trace(projection)/torch.sum(system._get_occ())

if __name__ == "__main__":

    print(fcn(**system_dqc))

    # def _min_fwd_fcn(y, *params):
    #     pfunc = get_pure_function(fcn)
    #     with torch.enable_grad():
    #         y1 = y.clone().requiresgrad()
    #         z = pfunc(y1, *params)
    #     grady, = torch.autograd.grad(z, (y1,), retain_graph=True,
    #                                  create_graph=torch.is_grad_enabled())

    # min_bparams = xitorch.optimize.minimize(fcn, bparams, (bpacker,),
    #                                         method = "Adam",step = 2e-3, maxiter = 1, verbose = True)




