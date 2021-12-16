
import xitorch.optimize

from optb import *
from torch.utils.tensorboard import SummaryWriter
from output import *

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

def optimize_basis(basis, basis_ref, atomstruc, step , maxiter, minimize_kwargs: dict = {}, out_kwargs : dict = {}):
    """
    function to optimize basis functions.
    :param basis: basis which should be optimized
    :param basis_ref: reference basis
    :param atomstruc: Molecule Structure
    :param step: learning rate
    :param maxiter: maximal learning steps
    :param minimize_kwargs: kwargs for the xitorch minimizer
    :param out_kwargs: kwargs to configure the output save functions
    :return: optimized basis
    """

    bsys1, bsys2, func_dict = Mole_minimizer(basis, basis_ref, atomstruc)

    writerpath, outpath = conf_output(basis, basis_ref, atomstruc, step, f_rtol,**out_kwargs)
    writer = SummaryWriter(writerpath)

    print(f"\n start optimization of {basis} Basis for the Molecule {atomstruc}")

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
                                            verbose=True,
                                            writer=writer, **minimize_kwargs)

    bsys1.get_SCF()
    energy_small_basis = bsys1.SCF.get_tot_energy
    energy_ref_basis = bsys2.SCF.get_tot_energy

    optbasis = bconv(min_bparams, func_dict["bpacker"])
    optbasis_energy = scf_dft_energy(optbasis, bsys1.atomstruc)

    save_output(outpath, basis, energy_small_basis, basis_ref, energy_ref_basis, optbasis, optbasis_energy)

    return optbasis

if __name__ == "__main__":


    ####################################################################################################################
    # configure atomic optb:
    ####################################################################################################################

    # atomstruc = [['H', [0.5, 0.0, 0.0]],
    #              ['H',  [-0.5, 0.0, 0.0]]]

    atomstruc = "bf"

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
    step = 2e-5
    maxiter = 100
    f_rtol = 1e-12

    ####################################################################################################################
    # run actual optimization
    ####################################################################################################################

    optimizebasis(basis,basis_ref,atomstruc,step,maxiter, minimize_kwargs = {"f_rtol" : f_rtol})