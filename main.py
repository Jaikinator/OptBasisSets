import xitorch.optimize
import sys
from optb import *
from torch.utils.tensorboard import SummaryWriter

# sys.stdout = open("out.txt", "w")
########################################################################################################################
# check if GPU is used:
# setting device on GPU if available, else CPU
########################################################################################################################
def cuda_device_checker(memory=False):
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

torch.set_printoptions(linewidth=200)


def fcn(bparams: torch.Tensor, bpacker: xitorch._core.packer.Packer, ref_basis
        , atomstruc_dqc: str, atomstruc: list, coeffM: torch.Tensor, occ_scf: torch.Tensor, num_gauss: torch.Tensor):
    """
    Function to optimize
    :param bparams: torch.tensor with coeff of basis set for the basis that has to been optimized
    :param bpacker: xitorch._core.packer.Packer object to create the CGTOBasis out of the bparams
    :param bparams_ref: torch.tensor like bparams but now for the refenrenence basis on which to optimize.
    :param bpacker_ref: xitorch._core.packer.Packer like Packer but now for the refenrenence basis on which to optimize.
    :param atomstruc_dqc: string of atom structure
    :return:
    """

    basis = bpacker.construct_from_tensor(
        bparams)  # create a CGTOBasis Object (set informations about gradient, normalization etc)

    basis_cross = blister(atomstruc, basis, ref_basis)

    colap = crossoverlap(atomstruc_dqc, basis_cross)

    # maximize overlap

    _projt = projection(coeffM, colap, num_gauss)

    occ_scf = occ_scf[occ_scf > 0]

    _projection = torch.zeros((_projt.shape[0], occ_scf.shape[0]), dtype=torch.float64)

    for i in range(len(occ_scf)):
        _projection[:, i] = torch.mul(_projt[:, i], occ_scf[i])

    return -torch.trace(_projection) / torch.sum(occ_scf)


def scf_dft_energy(basis, atomstruc):
    mol = gto.Mole()
    mol.atom = atomstruc
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
    mf.xc = 'B3LYP'
    mf.kernel()
    return mf.energy_tot()


if __name__ == "__main__":
    ####################################################################################################################
    # configure atomic optb:
    ####################################################################################################################

    # atomstruc = [['H', [0.5, 0.0, 0.0]],
    #              ['H',  [-0.5, 0.0, 0.0]]]

    atomstruc = "CH4"

    ####################################################################################################################
    # configure basis to optimize:
    ####################################################################################################################

    basis = "STO-3G"

    ####################################################################################################################
    # configure reference basis:
    ####################################################################################################################

    basis_ref = "cc-pvtz"

    ####################################################################################################################

    bsys1, bsys2, func_dict = Mole_minimizer(basis, basis_ref, atomstruc)

    ####################################################################################################################
    # set up xitorch.optimize.minimize
    ####################################################################################################################
    step = 2e-6
    maxiter = 0
    f_rtol = 1e-8

    print("\n start optimization")
    writer = SummaryWriter(f"molecule_{atomstruc}", comment=f"step:{step}, f_rtol: {f_rtol} ")


    min_bparams = xitorch.optimize.minimize(fcn,
                                            func_dict["bparams"],
                                            (func_dict["bpacker"],
                                             func_dict["ref_basis"],
                                             func_dict["atomstruc_dqc"],
                                             func_dict["atomstruc"],
                                             func_dict["coeffM"],
                                             func_dict["occ_scf"],
                                             func_dict["num_gauss"],),
                                            step=step,
                                            method="Adam",
                                            maxiter=maxiter,
                                            f_rtol=f_rtol,
                                            verbose=True,
                                            writer=writer)

    testsystem = Mole(basis, atomstruc)

    initenergy = testsystem.SCF.get_tot_energy

    print(f"total energy scf with {basis} as initial basis:\n",
          initenergy)
    print(f"total energy scf with {basis_ref} as reference basis:\n",
          bsys2.SCF.get_tot_energy)
    print("energy after optimization of basis as basis set:\n",
          scf_dft_energy(bconv(min_bparams, func_dict["bpacker"]), testsystem.SCF.atomstruc))

    # print("basis scf initial:")
    # pprint.pprint(testsystem.SCF.get_basis)
    # print("basis after optimization:")
    # pprint.pprint(bconv(min_bparams, func_dict["bpacker"]))

    basist = func_dict["bpacker"].construct_from_tensor(min_bparams)


    test_dict = bconv(min_bparams, func_dict["bpacker"])

    # print(bse.write_formatted_basis_str(test_dict, "nwchem"))



"""
To configure verbose options on which step something should be printed look at
~/xitorch/_impls/optimize/minimizer.py line 183

"""
