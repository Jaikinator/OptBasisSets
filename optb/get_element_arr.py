"""
extract array of elements out of a given atom-structure
"""

from optb.data.params_periodic_system import el_dict  # contains dict with all numbers and Symbols of the periodic table

def get_element_arr(atomstruc):
    """
    create array with all elements in the optb
    """
    elements_arr = [atomstruc[i][0] for i in range(len(atomstruc))]
    for i in range(len(elements_arr)):
        if type(elements_arr[i]) is str:
            elements_arr[i] = el_dict[elements_arr[i]]
    return elements_arr

