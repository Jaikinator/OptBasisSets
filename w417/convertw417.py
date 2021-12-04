import pandas as pd


df = pd.read_csv("/w417/nonMR.dat", delimiter=" ", names=["mol", "charge", "mult"])

w417 = {}

# file = open(f"/home/jacob/PycharmProjects/OptBasisSets/data/w417/xyz/acetaldehyde.xyz")
# line = file.readline()
index = df.index


for molecule in df["mol"]:
        file = open(f"w417/xyz/{molecule}.xyz")
        ind = index[df["mol"] == molecule] # index of the specific molecule in dataframe

        line = file.readline()
        i = 0
        atompos = []

        w417[molecule] = {} # output arr
        w417[molecule]["charge"] = int(df["charge"][ind])  # input charge
        w417[molecule]["mult"] = int(df["mult"][ind])  # input multiplicity
        while line:
            i += 1
            if i == 2:
                w417[molecule]["energy"] = float(line.split()[2])
            if i > 2:
                line_arr = line.split()
                innerarr = []
                for el in range(len(line_arr)):
                    if el == 0:
                        innerarr.append(line_arr[el])
                    else:
                        innerarr.append(float(line_arr[el]))
                atompos.append(innerarr)
            line = file.readline()
        file.close()
        w417[molecule]["atompos"] = atompos

import pprint
#pprint.pprint(w417)


from ase.collections import g2
pprint.pprint(g2.names)
# print(f.split())
#dtype=["S2", float, float, float]