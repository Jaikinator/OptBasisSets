from ase.build import molecule
from ase.collections import g2
print(g2.names)
print(molecule("CH3CH2O").get_chemical_symbols())
print(molecule("CH3CH2O").get_positions())

testmol = molecule("CH3CH2O")
testmol.get_calculator

print(molecule("CH3CH2O").get_initial_magnetic_moments())
print(molecule("CH3CH2O").get_initial_charges())