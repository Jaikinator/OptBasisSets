from pyscf import gto, scf
import dqc
import numpy as np
import torch
import xitorch as xt
import xitorch.optimize
from dqc.utils.datastruct import AtomCGTOBasis
import dqc.hamilton.intor as intor
from dqc.api.parser import parse_moldesc

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

#create first basis set. (reference)

basis = [dqc.loadbasis("1:3-21G"), dqc.loadbasis("1:3-21G")]

bpacker = xt.Packer(basis)
bparams = bpacker.get_param_tensor()

atomstruc = "H 1 0 0; H -1 0 0"
atomzs, atompos = parse_moldesc(atomstruc)
atombases = [AtomCGTOBasis(atomz=atomzs[i], bases=basis[i], pos=atompos[i]) for i in range(len(basis))]
wrap = dqc.hamilton.intor.LibcintWrapper(
        atombases)  # creates an wrapper object to pass informations on lower functions
S = intor.overlap(wrap)


########################################################################################################################
#pyscf stuff to calc coefficient Matrix
atom_scf = [['H', [1.0, 0.0, 0.0]],
            ['H', [-1.0, 0.0, 0.0]]]
basis_scf = "3-21G"

########################################################################################################################
# create the second basis set to optimize

rest_basis = [dqc.loadbasis("1:cc-pvdz", requires_grad=False),dqc.loadbasis("1:cc-pvdz", requires_grad=False)]
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

def _dm_HF(atomstruc, basis):
    """
    calulate the density matrix using mol type Objects.
    using Hartree_Fock type calc.
    :param atomstruc: str or 2-elements tuple
                    Description of the molecule system.
                    If string, it can be described like ``"H 1 0 0; H -1 0 0"``.
                    If tuple, the first element of the tuple is the Z number of the atoms while
                    the second element is the position of the atoms: ``(atomzs, atomposs)``.
    :param basis:str, CGTOBasis, list of str, or CGTOBasis
                The string describing the gto basis. If it is a list, then it must have
                the same length as the number of atoms.
    :return:torch.trensor
    """
    m = dqc.Mol(atomstruc, basis = basis, orthogonalize_basis = False)
    qc = dqc.HF(m).run()
    return qc.aodm()

def _coeff_mat_scf(basis,atom):
    """
    just creates the overlap matrix for different input basis
    """
    mol = gto.Mole()
    mol.atom = atom
    mol.spin = 0
    mol.unit = 'Bohr' # in Angstrom
    mol.verbose = 6
    mol.output = 'scf.out'
    mol.symmetry = False
    mol.basis = basis
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    return torch.tensor(mf.mo_coeff)

def _procetion_mat(coeff, colap, num_gauss):
    S_11 = _cross_selcet(colap, num_gauss, "S_11")
    S_12 = _cross_selcet(colap, num_gauss, "S_12")
    S_21 = _cross_selcet(colap, num_gauss, "S_21")
    S_22 = _cross_selcet(colap, num_gauss, "S_22")
    print(S_11.shape,S_12.shape,S_21.shape,S_22.shape)

    s21_c = torch.matmul(S_12, coeff)
    s22_s21c = torch.matmul(torch.inverse(S_22), s21_c)
    s12_s22s21c = torch.matmul(S_21, s22_s21c)
    return torch.matmul(coeff.T,s12_s22s21c)

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
    coeff_mat = _coeff_mat_scf(basis_scf, atom_scf)


    projection = _procetion_mat(coeff_mat,colap,num_gauss)

    # maximize overlap
    
    return projection



# print("Original basis")
#
# print(basis[0:2])
#
# min_bparams = xitorch.optimize.minimize(fcn, bparams, (bpacker,), method="Adam",
#                                          step=2e-3, maxiter=80, verbose=True)
#
#
# basis = bpacker.construct_from_tensor(min_bparams)
#
# rest_basis_in = bpacker_rest.construct_from_tensor(bparams_rest)
# basis = basis + rest_basis
# # m = dqc.Mol("H 1 0 0; H -1 0 0", basis = basis[0:2])
# atomzs, atompos = parse_moldesc("H 1 0 0; H -1 0 0")
# atomz = torch.cat([atomzs, atomzs])
# atompos = torch.cat([atompos, atompos])
# # print("shapes:", atomz.shape, atompos.shape)
# atombases = [
#     AtomCGTOBasis(atomz=atomz[i], bases=basis[i], pos=atompos[i]) \
#     for i in range(len(basis))
# ]
# wrap = dqc.hamilton.intor.LibcintWrapper(atombases)
# #print(intor.overlap(wrap))
# # print("Optimized basis")
# # print(opt_basis)
#
#
# # wrap = dqc.hamilton.intor.LibcintWrapper(atombases)
# #
# # intor.overlap(wrap)
if __name__ == "__main__":

    print(fcn(bparams, bpacker))


