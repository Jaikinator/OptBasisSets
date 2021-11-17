from ase.build import molecule
from ase.collections import g2
print(g2.names)
print(molecule("H2").get_chemical_symbols())
print(molecule("H2").get_positions())