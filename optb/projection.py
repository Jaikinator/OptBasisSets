"""
This file includes all the needed functions to calculate the projection between two given
Basis Sets.
"""


import torch
from xitorch._core.packer import Packer
from dqc.hamilton.intor import  overlap, LibcintWrapper
from dqc.utils.datastruct import AtomCGTOBasis
from dqc.api.parser import parse_moldesc

def blister(atomstruc : list, basis : dict, refbasis :dict):
    """
    function to convert the basis dict, based on the given atomstruc, to a list.
    This is used to create the parameter of AtomCGTOBasis Object.
    :param atomstruc: atomstruc array
    :param basis: basis dict first basis
    :param refbasis: reference basis dictionary
    :return:
    """

    elem = [atomstruc[i][0] for i in range(len(atomstruc))]
    b_arr = [basis[elem[i]] for i in range(len(elem))]
    bref_arr = [refbasis[elem[i]] for i in range(len(elem))]
    return b_arr + bref_arr

def cross_select(crossmat : torch.Tensor, num_gauss : torch.Tensor):
    """
    select the cross overlap matrix part.
        S  = [b1*b1 , b1*b2] = [S_11 , S_12]
             [b2*b1 , b2*b2]   [S_21 , S_22]
    :param crossmat: crossoverlap mat
    :param num_gauss: number of gaussians in two basis
    :return: torch.Tensor
    returns the cross overlap matrices between the new and the old basis func
    """

    S_11 = crossmat[0:num_gauss[0],0:num_gauss[0]]
    S_12 = crossmat[num_gauss[0]:, 0:(len(crossmat) - num_gauss[1])]
    S_21 = crossmat[0:num_gauss[0], num_gauss[0]:]
    S_22 = crossmat[num_gauss[0]:, num_gauss[0]:]
    return S_11, S_12, S_21, S_22

def crossoverlap(atomstruc : str, basis : list):
    """
    calculate the cross overlap matrix between to basis functions.
    The corssoverlap is defined by the overlap between to basis sets.
    For example is b1 the array of the first basis set which will be optimized and
    b2 the array of the second basis set which is the reference basis
    Then is the crossoveralp S:
        S  = [b1*b1 , b1*b2] = [S_11 , S_12]
             [b2*b1 , b2*b2]   [S_21 , S_22]
    :if not self.normalized:param atomstruc: molecular structure in dqc format
    :param basis: list of basis sets eg. [b1, b2] where b1 is the basis that is going to be optimized and
                  b2 is the reference basis.
    :return: torch.Tensor shape (len(b1)+len(b2)) x (len(b1)+len(b2))
    """
    # change normalization state so that it will be normalized by AtomCGTOBasis again
    # change_norm_state(basis)

    # calculate cross overlap matrix:

    atomzs, atompos = parse_moldesc(atomstruc)

    # atomzs : Atomic number (torch.tensor); len: number of Atoms
    # atompos : atom positions tensor of shape (3x len: number of Atoms )
    # now double both atom information to use is for the second basis set

    atomzs = torch.cat([atomzs, atomzs])
    atompos = torch.cat([atompos, atompos])
    atombases = [AtomCGTOBasis(atomz=atomzs[i], bases=basis[i], pos=atompos[i]) for i in range(len(basis))]

    # creates a list with AtomCGTOBasis object for each atom (including  all previous information in one array element)
    wrap = LibcintWrapper(
        atombases)  # creates a wrapper object to pass information on lower functions
    return overlap(wrap)

def projection_mat(coeff : torch.Tensor, colap : torch.Tensor, num_gauss : torch.Tensor):
    """
    Calculate the Projection from the old to the new Basis:
         P = C^T S_12 S⁻¹_11 S_21 C
    :param coeff: coefficient matrix of the bigger reference basis calculated by pyscf (C Matrix in eq.)
    :param colap: crossoverlap Matrix (S and his parts)
    :param num_gauss: array with length of the basis sets
    :return: Projection Matrix
    """
    S_11, S_12, S_21, _ = cross_select(colap, num_gauss)
    s21_c = torch.matmul(S_21, coeff)
    s11_s21c = torch.matmul(torch.inverse(S_11), s21_c)
    s21_s11s12c = torch.matmul(S_12, s11_s21c)
    P = torch.matmul(coeff.T, s21_s11s12c)
    return P

"""
Function for the Projection between two basis function.
This function will be optimized by the xitorch minimizer.
"""

def projection(bparams: torch.Tensor, bpacker: Packer, ref_basis
        , atomstruc_dqc: str, atomstruc: list, coeffM: torch.Tensor, occ_scf: torch.Tensor, num_gauss: torch.Tensor):
    """
    Function to optimize
    :param bparams: torch.tensor with coeff of basis set for the basis that has to been optimized
    :param bpacker: xitorch._core.packer.Packer object to create the CGTOBasis out of the bparams
    :param bparams_ref: torch.tensor like bparams but now for the refenrenence basis on which to optimize.
    :param bpacker_ref: xitorch._core.packer.Packer like Packer but now for the refenrenence basis on which to optimize.
    :param atomstruc_dqc: string of atom structure
    :return:
    """

    basis = bpacker.construct_from_tensor(
        bparams)  # create a CGTOBasis Object (set informations about gradient, normalization etc)

    basis_cross = blister(atomstruc, basis, ref_basis)

    colap = crossoverlap(atomstruc_dqc, basis_cross)

    # maximize overlap

    _projt = projection_mat(coeffM, colap, num_gauss)

    occ_scf = occ_scf[occ_scf > 0]

    _projection = torch.zeros((_projt.shape[0], occ_scf.shape[0]), dtype=torch.float64)

    for i in range(len(occ_scf)):
        _projection[:, i] = torch.mul(_projt[:, i], occ_scf[i])

    return -torch.trace(_projection) / torch.sum(occ_scf)

