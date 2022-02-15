from pyscf import gto, scf
import json
from  optb import loadatomstruc
import scipy as sci
import numpy as np

basis = json.load(open("output/STO-3G_cc-pvtz/ethanol/ethanol_opt_STO-3G_basis_011.json"))
Mol = loadatomstruc("ethanol")
mol = gto.Mole()
mol.atom = Mol.atomstruc
mol.basis = basis
mol.unit = 'Bohr'  # in Angstrom
mol.verbose = 6
mol.spin = 0
mol.build()
mf = scf.RKS(mol)
mf.xc = "B3LYP"
mf.kernel()


print(mf.energy_tot())