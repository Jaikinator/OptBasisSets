import dqc
import torch
import xitorch as xt
import xitorch.optimize
from dqc.utils.datastruct import AtomCGTOBasis
import dqc.hamilton.intor as intor
from dqc.api.parser import parse_moldesc

basis = [dqc.loadbasis("1:3-21G"), dqc.loadbasis("1:3-21G")]

# rest_basis = [dqc.loadbasis("1:cc-pvdz"),dqc.loadbasis("1:cc-pvdz")]
#
# m = dqc.Mol("H 1 0 0; H -1 0 0", basis=basis)
#
# atombases = [
#         AtomCGTOBasis(atomz=m.atomzs[i], bases=basis[i], pos=m.atompos[i]) \
#         for i in range(len(basis))
#     ]

bpacker = xt.Packer(basis)
bparams = bpacker.get_param_tensor()
# print(f'bparams: {bparams} \n params list: {bpacker.get_param_tensor_list()} \n basis: {basis}')

rest_basis = [dqc.loadbasis("1:cc-pvdz", requires_grad=False),dqc.loadbasis("1:cc-pvdz", requires_grad=False)]

bparams_rest = xt.Packer(rest_basis).get_param_tensor()

#basis_2 = [dqc.loadbasis("1:3-21G"), dqc.loadbasis("1:3-21G"),dqc.loadbasis("1:cc-pvdz"),dqc.loadbasis("1:cc-pvdz") ]
#bpacker = xt.Packer(basis_2)
#bparams = bpacker.get_param_tensor()

def fcn(bparams, bpacker):

    basis = bpacker.construct_from_tensor(bparams)
    #rest_basis_in =  bpacker.construct_from_tensor(bparams_rest)
    basis = basis + rest_basis
    print(len(basis))
    #m = dqc.Mol("H 1 0 0; H -1 0 0", basis = basis[0:2])
    atomzs, atompos = parse_moldesc("H 1 0 0; H -1 0 0")

    atomz = torch.cat([atomzs, atomzs])
    atompos = torch.cat([atompos, atompos])

    # print("shapes:", atomz.shape, atompos.shape)
    atombases = [
        AtomCGTOBasis(atomz=atomz[i], bases = basis[i], pos=atompos[i]) \
        for i in range(len(basis))
    ]
    wrap = dqc.hamilton.intor.LibcintWrapper(atombases)
    # print(intor.overlap(wrap))
    return -torch.sum(intor.overlap(wrap))


print("Original basis")
# print(basis[0:2])

min_bparams = xitorch.optimize.minimize(fcn, bparams, (bpacker,), method="Adam",
                                         step=2e-3, maxiter=100, verbose=True)


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