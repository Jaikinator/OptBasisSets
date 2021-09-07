from pyscf import gto, scf
import dqc
import numpy as np
import xitorch as xt
from dqc.utils.datastruct import AtomCGTOBasis
import dqc.hamilton.intor as intor
from dqc.api.parser import parse_moldesc

#using pscf to calc:
#   the overlap Matrix
#   as well as print the coefficients of the basis sets
def overlapb_scf(basis):
    """
    just creates the overlap matrix for different input basis
    """
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

# print(f"coefficients of 3-21G: {gto.basis.load('3-21G',symb='H')}")
# print(f"coefficients of cc-pvdz: {gto.basis.load('cc-pvdz',symb='H')}")
# print(f"3-21G: \n\t{overlapb_scf('3-21G')},\ncc-pvdz\n\t{overlapb_scf('cc-pvdz')}")


########################################################################################################################
#using the dqc module to calc the basis and the overlap matrix

basis_dqc = [dqc.loadbasis("1:3-21G"), dqc.loadbasis("1:3-21G")]
basis2_dqc = [dqc.loadbasis("1:cc-pvdz"), dqc.loadbasis("1:cc-pvdz")]

def overlapb_dqc(basis):
    """
      just creates the overlap matrix for different input basis
    """
    bpacker = xt.Packer(basis)
    bparams = bpacker.get_param_tensor()

    atomstruc = "H 1 0 0; H -1 0 0"
    atomzs, atompos = parse_moldesc(atomstruc)
    atombases = [AtomCGTOBasis(atomz=atomzs[i], bases=basis[i], pos=atompos[i]) for i in range(len(basis))]

    wrap = dqc.hamilton.intor.LibcintWrapper(
            atombases)  # creates an wrapper object to pass informations on lower functions
    return intor.overlap(wrap)

#print(f"3-21G: \n\t{overlapb_dqc(basis_dqc)},\ncc-pvdz\n\t{overlapb_dqc(basis2_dqc)}")

print(f"the pyscf:\n3-21G: \n\t{gto.basis.load('3-21G',symb='H')}\n"
      f"coefficients of cc-pvdz: \n\t{gto.basis.load('cc-pvdz',symb='H')} ")
print(f"\nthe dqc:\n1:3-21G:\n\t {basis_dqc[0]},\n1:cc-pvdz\n\t{basis2_dqc[0]} ")

#checks that the overlap matrix is the same:
print(f"3-21G: {np.all(np.isclose(overlapb_scf('3-21G'),overlapb_dqc(basis_dqc)))}"
      f"\ncc-pvdz: {np.all(np.isclose(overlapb_scf('cc-pvdz'),overlapb_dqc(basis2_dqc)))}")