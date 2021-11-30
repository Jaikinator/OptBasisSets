import ase.io
from ase.io.gaussian import read_gaussian_in
from ase.build import molecule
from ase.collections import g2
# print(g2.names)
# print(molecule("CH3CH2O").get_chemical_symbols())
# print(molecule("CH3CH2O").get_positions())
#
# testmol = molecule("CH3CH2O")
# testmol.get_calculator
#
# arr = g2.names

# multdict = {}
# for i in arr:
#     multdict[i] = int(sum(molecule(i).get_initial_magnetic_moments())+1)
#
# inp = True
# while inp:
#     try:
#         el = str(input("choose element: "))
#         print(el , multdict[el])
#     except:
#         print("element not found")
#     if bool(el) ==  False:
#         inp = False
# print(molecule("CH3CH2O").get_initial_magnetic_moments())
# print(molecule("CH3CH2O").get_initial_charges())

print(read_gaussian_in("/W4-17_All/h2o.com"))

