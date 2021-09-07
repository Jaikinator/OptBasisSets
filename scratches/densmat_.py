from pyscf import gto, scf
import dqc
import torch
import numpy as np
from dqc.api.parser import parse_moldesc
from dqc.utils.datastruct import AtomCGTOBasis
import dqc.hamilton.intor as intor

mol = gto.Mole()
mol.atom = [['H', [1.0, 0.0, 0.0]],
            ['H', [-1.0, 0.0, 0.0]]]
mol.spin = 0
mol.unit = 'Bohr' # in Angstrom
mol.verbose = 6
mol.output = 'scf.out'
mol.symmetry = False
mol.basis = '3-21G'
mol.build()

mf = scf.RHF(mol)
mf.kernel()
mf.mo_coeff
dm_scf = mf.make_rdm1()
overlap_scf =  mf.get_ovlp()
print(np.trace(np.matmul(overlap_scf,dm_scf)))


########################################################################################################################

basis = [dqc.loadbasis("1:3-21G"), dqc.loadbasis("1:3-21G")]
atomstruc = "H 1 0 0; H -1 0 0"
m = dqc.Mol(atomstruc, basis = basis, orthogonalize_basis=False)

qc = dqc.HF(m).run()
dm_dqc = qc.aodm()

atomzs, atompos = parse_moldesc("H 1 0 0; H -1 0 0")
atombases = [AtomCGTOBasis(atomz=atomzs[i], bases=basis[i], pos=atompos[i])
                  for i in range(len(basis))]
print(atombases)
wrap = dqc.hamilton.intor.LibcintWrapper(atombases)
overlap_sdqc = intor.overlap(wrap)

print(torch.trace(torch.matmul(overlap_sdqc, dm_dqc)))


 from pyscf import gto, scf
mol = gto.Mole()
mol.atom = [['Ni', [-1, 0.0, 0.0]]]
mol.spin = 0
mol.basis = '3-21G'
pyscf_o = mol.get_ovlp()


