import dqc
import torch
from dqc.utils.datastruct import AtomCGTOBasis
import dqc.hamilton.intor as intor
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

def cross_selcet(crossmat : torch.Tensor, num_gauss : torch.Tensor):
    """
    select the cross overlap matrix part.
    The corssoverlap is defined by the overlap between to basis sets.
    For example is b1 the array of the first basis set which will be optimized and
    b2 the array of the second basis set which is the reference basis
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
    :return: torch.tensor
    returns the cross overlap matrix between the new and the old basis func
    """

    S_11 = crossmat[0:num_gauss[0],0:num_gauss[0]]
    S_12 = crossmat[num_gauss[0]:, 0:(len(crossmat) - num_gauss[1])]
    S_21 = crossmat[0:num_gauss[0], num_gauss[0]:]
    S_22 = crossmat[num_gauss[0]:, num_gauss[0]:]
    return S_11, S_12, S_21, S_22

def crossoverlap(atomstruc : str, basis : list):

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
    # creats an list with AtomCGTOBasis object for each atom (including  all previous informations in one array element)
    wrap = dqc.hamilton.intor.LibcintWrapper(
        atombases)  # creates an wrapper object to pass informations on lower functions

    return intor.overlap(wrap)

def projection(coeff : torch.Tensor, colap : torch.Tensor, num_gauss : torch.Tensor):
    """
    Calculate the Projection from the old to the new Basis:
         P = C^T S_12 S⁻¹_11 S_21 C
    :param coeff: coefficient matrix of the bigger reference basis calculated by pyscf (C Matrix in eq.)
    :param colap: crossoverlap Matrix (S and his parts)
    :param num_gauss: array with length of the basis sets
    :return: Projection Matrix
    """
    S_11, S_12, S_21, _ = cross_selcet(colap, num_gauss)
    s21_c = torch.matmul(S_21, coeff)
    s11_s21c = torch.matmul(torch.inverse(S_11), s21_c)
    s21_s11s12c = torch.matmul(S_12, s11_s21c)
    P = torch.matmul(coeff.T, s21_s11s12c)
    return P
