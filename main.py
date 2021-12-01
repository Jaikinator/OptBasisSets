import xitorch.optimize
from dqc.utils.misc import gaussian_int

from optb import *

########################################################################################################################
# check if GPU is used:
# setting device on GPU if available, else CPU
########################################################################################################################
def cuda_device_checker(memory  = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if memory is False:
        print(f"Using device: {device, torch.version.cuda}\n{torch.cuda.get_device_name(0)}")
    else:
        if device.type == 'cuda':
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

# cuda_device_checker()

########################################################################################################################
# configure torch tensor
########################################################################################################################

torch.set_printoptions(linewidth = 200)


def fcn(bparams : torch.Tensor, bpacker: xitorch._core.packer.Packer, ref_basis
        , atomstruc_dqc: str, atomstruc : list, coeffM : torch.Tensor, occ_scf : torch.Tensor, num_gauss : torch.Tensor):

    """
    Function to optimize
    :param bparams: torch.tensor with coeff of basis set for the basis that has to been optimized
    :param bpacker: xitorch._core.packer.Packer object to create the CGTOBasis out of the bparams
    :param bparams_ref: torch.tensor like bparams but now for the refenrenence basis on which to optimize.
    :param bpacker_ref: xitorch._core.packer.Packer like Packer but now for the refenrenence basis on which to optimize.
    :param atomstruc_dqc: string of atom structure
    :return:
    """

    basis = bpacker.construct_from_tensor(bparams) #create a CGTOBasis Object (set informations about gradient, normalization etc)

    basis_cross = blister(atomstruc,basis,ref_basis)

    colap = crossoverlap(atomstruc_dqc, basis_cross)

    # maximize overlap

    _projt = projection(coeffM,colap,num_gauss)

    occ_scf = occ_scf[occ_scf > 0]

    _projection = torch.zeros((_projt.shape[0], occ_scf.shape[0]), dtype = torch.float64)

    for i in range(len(occ_scf)):
        _projection[:,i] = torch.mul(_projt[:,i] , occ_scf[i])

    return -torch.trace(_projection)/torch.sum(occ_scf)


def scf_dft_energy(basis, atomstruc):
    if type(atomstruc) is str:
        mol_ase =  molecule(atomstruc)
        chem_symb = mol_ase.get_chemical_symbols()
        atompos = mol_ase.get_positions()

        arr = []
        for i in range(len(chem_symb)):
            arr.append([chem_symb[i], list(atompos[i])])
    else:
        arr = atomstruc
    mol = gto.Mole()
    mol.atom =arr
    mol.unit = 'Bohr'  # in Angstrom
    mol.verbose = 6
    mol.output = 'scf.out'
    mol.symmetry = False
    mol.basis = basis

    try:
        mol.spin = 0
        mol.build()
    except:
        mol.spin = 1
        mol.build()

    mf = scf.RKS(mol)
    mf.kernel()
    mf.xc = 'B3LYP'
    return mf.energy_tot()


if __name__ == "__main__":


    ####################################################################################################################
    # configure atomic optb:
    ####################################################################################################################

    # atomstruc = [['H', [0.5, 0.0, 0.0]],
    #              ['H',  [-0.5, 0.0, 0.0]],
    #              ['He', [0.0,0.5, 0.0 ]]]


    atomstruc = "CH4"

    ####################################################################################################################
    # configure basis to optimize:
    ####################################################################################################################

    basis = "STO-3G"
    #optb = dft_system(basis, atomstruc)

    ####################################################################################################################
    # configure reference basis:
    ####################################################################################################################
    basis_ref = "cc-pvtz"
    #system_ref = dft_system(basis_ref, atomstruc)

    ####################################################################################################################

    bsys1, bsys2, func_dict = Mole_minimizer(basis, basis_ref, atomstruc)



    print("\n start optimization")

    min_bparams = xitorch.optimize.minimize(fcn,  func_dict["bparams"], (func_dict["bpacker"],
                                                                        func_dict["ref_basis"],
                                                                        func_dict["atomstruc_dqc"],
                                                                        func_dict["atomstruc"],
                                                                        func_dict["coeffM"],
                                                                        func_dict["occ_scf"],
                                                                        func_dict["num_gauss"], ),step = 2e-6,method = "Adam", maxiter = 1, verbose = True)# ,method = "Adam"

    # testsystem = system_ase(basis, atomstruc)
    testsystem = Mole_ase(basis,atomstruc)

    initenergy = testsystem.SCF.get_tot_energy

    print(f"total energy scf with {basis} as initial basis:\n",
          initenergy)
    print(f"total energy scf with {basis_ref} as reference basis:\n",
          bsys2.SCF.get_tot_energy)
    print("energy after optimization of basis as basis set:\n",
          scf_dft_energy(bconv(min_bparams, func_dict["bpacker"]), atomstruc))

    print("basis scf initial:   ", testsystem.SCF.get_basis["C"])
    print("basis after one step:", bconv(min_bparams, func_dict["bpacker"])["C"])

    basist = func_dict["bpacker"].construct_from_tensor(min_bparams)


"""    
def _min_fwd_fcn(y, *params):
         pfunc = get_pure_function(fcn)
         with torch.enable_grad():
             y1 = y.clone().requiresgrad()
             z = pfunc(y1, *params)
         grady, = torch.autograd.grad(z, (y1,), retain_graph=True,
                                      create_graph=torch.is_grad_enabled())
"""


"""
To configure verbose options on which step something should be printed look at 
~/xitorch/_impls/optimize/minimizer.py line 183

"""


