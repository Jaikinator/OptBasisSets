from pyscf import gto, scf
import torch
import dqc
import numpy as np
import xitorch as xt
from dqc.utils.datastruct import AtomCGTOBasis
import dqc.hamilton.intor as intor
from dqc.api.parser import parse_moldesc


#using pscf to calc:
#   the overlap Matrix
#   as well as print the coefficients of the basis sets
def overlap_basis_changer(basis):
    mol = gto.Mole()
    mol.atom = [['H', [1.0, 0.0, 0.0]],
                ['H', [-1.0, 0.0, 0.0]]]
    mol.spin = 0
    mol.unit = 'Bohr' # in Angstrom
    mol.verbose = 6
    mol.output = 'scf.out'
    mol.symmetry = False
    mol.basis = basis
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    mf.mo_coeff
    dm = mf.make_rdm1()
    overlap = mf.get_ovlp()
    return overlap

print(f"3-21G: \n\t{overlap_basis_changer('3-21G')},\ncc-pvdz\n\t{overlap_basis_changer('cc-pvdz')}")


def get_ovlp_c(mol):
    atom1 = []
    basis1 = {}
    atom2 = []
    basis2 = {}
    for i in range(mol.natm):
        atom1.append(['ghost-' + str(i), tuple(mol.atom_coord(i))])
        basis1['ghost-' + str(i)] = gto.basis.load('3-21G', mol.atom_symbol(i))
        atom2.append(['ghost-' + str(i), tuple(mol.atom_coord(i))])
        basis2['ghost-' + str(i)] = gto.basis.load('cc-pvdz', mol.atom_symbol(i))

    mol1 = gto.M(atom=atom1, basis=basis1, unit='Bohr')
    mol2 = gto.M(atom=atom2, basis=basis2, unit='Bohr')

    n_bas1 = len(mol1.ao_labels())
    n_bas2 = len(mol2.ao_labels())

    mol_aux = mol1 + mol2
    ovl = mol_aux.get_ovlp()
    mol_aux.unit = 'Bohr'
    return ovl

basis = [dqc.loadbasis("1:3-21G"), dqc.loadbasis("1:3-21G")]
print(dqc.loadbasis("1:3-21G"))
bpacker = xt.Packer(basis)
bparams = bpacker.get_param_tensor()

atomstruc = "H 1 0 0; H -1 0 0"
atomzs, atompos = parse_moldesc(atomstruc)
atombases = [AtomCGTOBasis(atomz=atomzs[i], bases=basis[i], pos=atompos[i]) for i in range(len(basis))]

wrap = dqc.hamilton.intor.LibcintWrapper(
        atombases)  # creates an wrapper object to pass informations on lower functions
S = intor.overlap(wrap)
print("\t",S)

########################################################################################################################
# create the second basis set to optimize

rest_basis = [dqc.loadbasis("1:cc-pvdz", requires_grad=False)]

bpacker_rest = xt.Packer(rest_basis)
bparams_rest = bpacker_rest.get_param_tensor()

#c_overlap_dqc = m.fcn(bparams, bpacker)

#np.einsum('mi,ni,i->mn', mf.mo_coeff, mf.mo_coeff, mf.mo_occ)