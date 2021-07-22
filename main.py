import dqc
import torch
import xitorch as xt
import xitorch.optimize
from dqc.utils.datastruct import AtomCGTOBasis
import dqc.hamilton.intor as intor
from dqc.api.parser import parse_moldesc

########################################################################################################################
#create first basis set. (reference)

basis = [dqc.loadbasis("1:3-21G"), dqc.loadbasis("1:3-21G")]
bpacker = xt.Packer(basis)
bparams = bpacker.get_param_tensor()

########################################################################################################################
# create the second basis set to optimize

rest_basis = [dqc.loadbasis("1:cc-pvdz", requires_grad=False),dqc.loadbasis("1:cc-pvdz", requires_grad=False)]
bparams_rest = xt.Packer(rest_basis).get_param_tensor()

########################################################################################################################

def fcn(bparams, bpacker):
    """
    Function to optimize
    :param bparams: torch.tensor with coeff of basis set
    :param bpacker: xitorch._core.packer.Packer
    :return:
    """


    basis = bpacker.construct_from_tensor(bparams) #create a CGTOBasis Object (set informations about gradient, normalization etc)
    
    atomzs, atompos = parse_moldesc("H 1 0 0; H -1 0 0")

    # atomzs : Atomic number (torch.tensor); len: number of Atoms
    # atompos : atom positions tensor of shape (3x len: number of Atoms )
    # now double both atom informations to use is for the second basis set

    atomzs = torch.cat([atomzs, atomzs])
    atompos = torch.cat([atompos, atompos])

    atombases = [AtomCGTOBasis(atomz=atomzs[i], bases = basis[i], pos=atompos[i]) for i in range(len(basis))]
    # creats an list with AtomCGTOBasis object for each atom (including  all previous informations in one array element)

    wrap = dqc.hamilton.intor.LibcintWrapper(atombases) #creates an wrapper object to pass informations on lower functions

    print(intor.overlap(wrap).shape)

    return -torch.sum(intor.overlap(wrap))


print("Original basis")
# print(basis[0:2])

min_bparams = xitorch.optimize.minimize(fcn, bparams, (bpacker,), method="Adam",
                                         step=2e-3, maxiter=1, verbose=True)


basis = bpacker.construct_from_tensor(min_bparams)
# rest_basis_in =  bpacker.construct_from_tensor(bparams_rest)
basis = basis + rest_basis
# m = dqc.Mol("H 1 0 0; H -1 0 0", basis = basis[0:2])
atomzs, atompos = parse_moldesc("H 1 0 0; H -1 0 0")
atomz = torch.cat([atomzs, atomzs])
atompos = torch.cat([atompos, atompos])
# print("shapes:", atomz.shape, atompos.shape)
atombases = [
    AtomCGTOBasis(atomz=atomz[i], bases=basis[i], pos=atompos[i]) \
    for i in range(len(basis))
]
wrap = dqc.hamilton.intor.LibcintWrapper(atombases)
print(intor.overlap(wrap))
# print("Optimized basis")
# print(opt_basis)


# wrap = dqc.hamilton.intor.LibcintWrapper(atombases)
#
# intor.overlap(wrap)