"""
load and store molecule information from a database and configure the atom-structure
"""

from dataclasses import dataclass, field
from typing import Union

from optb.data.w417db import *
import optb.data.avdata as avdata
import optb.data.preselected_avdata as presel_avdata

# get atom pos:
from ase.build import molecule as asemolecule


@dataclass
class AtomsDB:
    """
    Class to store Data form Databases
    """
    atomstrucstr: str
    atomstruc: list
    mult: int
    charge: int
    energy: float
    molecule: object = field(repr=False)



def loadatomstruc(atomstrucstr: Union[list, str], db=None, preselected=True):
    """
    load molecule informations from a given DB and pass it throw to the AtomsDB class
    The supportet databasis are g2 and w4-17
    :param atomstrucstr: string of molecule like CH4 H2O etc.
    :param db: select specific
    :param preselected: load preselected Database files where multiplicity is 1 and charge is 0
    :return: AtomsDB
    """

    def _create_atomstruc_from_ase(molec):
        """
        creates atomstruc from ase database.
        :param atomstruc: molecule string
        :return: array like [element, [x,y,z], ...]
        """
        chem_symb = molec.get_chemical_symbols()
        atompos = molec.get_positions()

        arr = []
        for i in range(len(chem_symb)):
            arr.append([chem_symb[i], list(atompos[i])])
        return arr

    def from_W417(atomstrucstr):
        molec = W417(atomstrucstr)
        return AtomsDB(atomstrucstr, molec.atom_pos, molec.mult, molec.charge,
                       molec.energy, molec.molecule)

    def from_g2(atomstrucstr):
        molec = asemolecule(atomstrucstr)
        atomstruc = _create_atomstruc_from_ase(molec)
        mult = sum(molec.get_initial_magnetic_moments()) + 1
        charge = sum(molec.get_initial_charges())
        energy = None
        return AtomsDB(atomstrucstr, atomstruc, mult, charge, energy, molec)

    def dberror():
        if db is None:
            db_text = "w417 or g2"
        else:
            db_text = db
        text = f"Your Molecule is not available in {db_text} Database or your database is not supported"
        raise ImportError(text)

    if db is None and preselected is True:
            if atomstrucstr in presel_avdata.elw417:
                return from_W417(atomstrucstr)

            elif atomstrucstr in presel_avdata.elg2:
                print("Attention you get your data from the less accurate g2 Database.\n"
                      "No energy detected")
                return from_g2(atomstrucstr)

            else:
                dberror()

    elif db is None and preselected is False:

            if atomstrucstr in avdata.elw417:
                return from_W417(atomstrucstr)

            elif atomstrucstr in avdata.elg2:

                print("Attention you get your data from the less accurate g2 Database.\n"
                      "No energy detected")
                return from_g2(atomstrucstr)

            else:
                dberror()

    elif db is not None and preselected is True:

        if db == "w417":

            if atomstrucstr in presel_avdata.elw417:
                return from_W417(atomstrucstr)
            else:
                dberror()

        elif db == "g2":

            if atomstrucstr in presel_avdata.elg2:
                print("Attention you get your data from the less accurate g2 Database (no energy detected).")
                return from_g2(atomstrucstr)
            else:
                dberror()

        else:
            text = f"Your Molecule is not available in {db} Database or your database is not supported." \
                   f"Or it does not match expectations "
            raise ImportError(text)

    elif db is not None and preselected is False:

        if db == "w417":
            if atomstrucstr in avdata.elw417:
                return from_W417(atomstrucstr)
            else:
                dberror()
        elif db == "g2":
            if atomstrucstr in avdata.elg2:
                print("Attention you get your data from the less accurate g2 Database.\n"
                      "NO energy detected")
                return from_g2(atomstrucstr)
            else:
                dberror()
        else:
            dberror()

    else:
        text = "Your Molecule is not be loaded maybe check db, preselected"
        raise ImportError(text)
