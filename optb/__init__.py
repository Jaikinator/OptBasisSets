from optb.MoleDQC import *
from optb.MoleSCF import *
from optb.AtomsDB import *
from optb.Mole import *
from optb.projection import *
from optb.get_element_arr import *
from optb.basis_converter import *
from optb.output import *
from optb.optimize_basis import *
from optb.data import *

from optb.version import get_version as _get_version
__version__ = _get_version()
print('optb version:', __version__)