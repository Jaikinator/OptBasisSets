
import xitorch.optimize

from optb import *
from torch.utils.tensorboard import SummaryWriter
from output import *
import argparse

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


def scf_dft_energy(basis, atomstruc, atomstrucstr):
    mol = gto.Mole()
    mol.atom = atomstruc
    mol.unit = 'Bohr'  # in Angstrom
    mol.verbose = 6
    if atomstrucstr is not None:
        if os.path.exists("./output"):
            mol.output = f'./output/scf_optB_{atomstrucstr}.out'  # try to save it in outputfolder
        else:
            mol.output = f'scf_optB_{atomstrucstr}.out'  # save in home folder
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

def optimize_basis(basis: str, basis_ref : str, atomstruc : Union[str, list],step: list[float] , maxiter = 100000000,
                   method: str = "Adam" , output_path = None ,  minimize_kwargs: dict = {}, out_kwargs : dict = {}):
    """
    function to optimize basis functions.
    :param basis: basis which should be optimized
    :param basis_ref: reference basis
    :param atomstruc: Molecule Structure
    :param step: learning rate
    :param maxiter: maximal learning steps
    :param method: method to minimize
    :param minimize_kwargs: kwargs for the xitorch minimizer
    :param out_kwargs: kwargs to configure the output save functions
    :return: optimized basis
    """

    try:
        f_rtol = minimize_kwargs["f_rtol"]
        minimize_kwargs.pop("f_rtol")
        if type(f_rtol) is list and len(f_rtol) == 1:
            f_rtol = [f_rtol[0]]
    except:
        f_rtol = [1e-8]

    if type(step) is list and type(f_rtol) is float:
        f_rtol_arr = []
        for i in range(len(step)):
            f_rtol_arr.append(f_rtol)
        f_rtol = f_rtol_arr

    elif type(step) is list and len(step) != len(f_rtol):
        f_rtol_arr = []
        for i in range(len(step)):
            f_rtol_arr.append(f_rtol)
        f_rtol = f_rtol_arr

    if type(atomstruc) is list and type(step) is list:

        if len(atomstruc) == len(step):

            for i in range(len(atomstruc)):

                bsys1, bsys2, func_dict = Mole_minimizer(basis, basis_ref, atomstruc[i])

                writerpath, outpath = conf_output(basis, basis_ref, atomstruc[i], step[i], f_rtol,
                                                  outf= output_path, **out_kwargs)
                writer = SummaryWriter(writerpath)

                print(f"\n start optimization of {basis} Basis for the Molecule {atomstruc[i]}")

                min_bparams = xitorch.optimize.minimize(fcn,
                                                        func_dict["bparams"],
                                                        (func_dict["bpacker"],
                                                         func_dict["ref_basis"],
                                                         func_dict["atomstruc_dqc"],
                                                         func_dict["atomstruc"],
                                                         func_dict["coeffM"],
                                                         func_dict["occ_scf"],
                                                         func_dict["num_gauss"],),
                                                        step=step[i],
                                                        method=method,
                                                        maxiter=maxiter,
                                                        verbose=True,
                                                        writer=writer,
                                                        f_rtol = f_rtol[i]
                                                        ,**minimize_kwargs)

                bsys1.get_SCF(atomstrucstr = atomstruc[i])
                energy_small_basis = bsys1.SCF.get_tot_energy
                energy_ref_basis = bsys2.SCF.get_tot_energy

                optbasis = bconv(min_bparams, func_dict["bpacker"])
                optbasis_energy = scf_dft_energy(optbasis, bsys1.atomstruc, atomstruc[i])

                save_output(outpath, basis, energy_small_basis, basis_ref, energy_ref_basis, optbasis,
                            optbasis_energy,
                            atomstruc[i], step[i], maxiter, method,f_rtol, optkwargs=minimize_kwargs)
            return optbasis

        else:
            for atom in atomstruc:
                bsys1, bsys2, func_dict = Mole_minimizer(basis, basis_ref, atom)

                for s in range(len(step)):

                    writerpath, outpath = conf_output(basis, basis_ref, atom, step[s], f_rtol,
                                                      outf=output_path, **out_kwargs)
                    writer = SummaryWriter(writerpath)

                    print(f"\n start optimization of {basis} Basis for the Molecule {atom}")

                    min_bparams = xitorch.optimize.minimize(fcn,
                                                            func_dict["bparams"],
                                                            (func_dict["bpacker"],
                                                             func_dict["ref_basis"],
                                                             func_dict["atomstruc_dqc"],
                                                             func_dict["atomstruc"],
                                                             func_dict["coeffM"],
                                                             func_dict["occ_scf"],
                                                             func_dict["num_gauss"],),
                                                            step=step[s],
                                                            method=method,
                                                            maxiter=maxiter,
                                                            verbose=True,
                                                            writer=writer,
                                                            f_rtol = f_rtol[s],
                                                            **minimize_kwargs)

                    bsys1.get_SCF(atomstrucstr=atom)
                    energy_small_basis = bsys1.SCF.get_tot_energy
                    energy_ref_basis = bsys2.SCF.get_tot_energy

                    optbasis = bconv(min_bparams, func_dict["bpacker"])
                    optbasis_energy = scf_dft_energy(optbasis, bsys1.atomstruc, atom)

                    save_output(outpath, basis, energy_small_basis, basis_ref, energy_ref_basis, optbasis,
                                optbasis_energy,
                                atom, step[s], maxiter, method ,f_rtol[s], optkwargs=minimize_kwargs)

            return optbasis

    elif type(atomstruc) is list and type(step) is float:

        for i in range(len(atomstruc)):
            bsys1, bsys2, func_dict = Mole_minimizer(basis, basis_ref, atomstruc[i])

            writerpath, outpath = conf_output(basis, basis_ref, atomstruc[i], step, f_rtol,
                                              outf=output_path, **out_kwargs)
            writer = SummaryWriter(writerpath)

            print(f"\n start optimization of {basis} Basis for the Molecule {atomstruc[i]}")

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
                                                    method=method,
                                                    maxiter=maxiter,
                                                    verbose=True,
                                                    writer=writer,
                                                    f_rtol = f_rtol
                                                    ,**minimize_kwargs)

            bsys1.get_SCF(atomstrucstr=atomstruc[i])
            energy_small_basis = bsys1.SCF.get_tot_energy
            energy_ref_basis = bsys2.SCF.get_tot_energy

            optbasis = bconv(min_bparams, func_dict["bpacker"])
            optbasis_energy = scf_dft_energy(optbasis, bsys1.atomstruc, atomstruc[i])

            save_output(outpath, basis, energy_small_basis, basis_ref, energy_ref_basis, optbasis,
                        optbasis_energy,
                        atomstruc[i], step, maxiter, method, f_rtol, optkwargs = minimize_kwargs)
        return optbasis

    elif type(atomstruc) is str and type(step) is list:

        for i in range(len(step)):
            bsys1, bsys2, func_dict = Mole_minimizer(basis, basis_ref, atomstruc)

            writerpath, outpath = conf_output(basis, basis_ref, atomstruc, step[i], f_rtol,
                                              outf=output_path, **out_kwargs)
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
                                                    step=step[i],
                                                    method=method,
                                                    maxiter=maxiter,
                                                    verbose=True,
                                                    writer=writer,
                                                    f_rtol = f_rtol[i],**minimize_kwargs)

            bsys1.get_SCF(atomstrucstr=atomstruc)
            energy_small_basis = bsys1.SCF.get_tot_energy
            energy_ref_basis = bsys2.SCF.get_tot_energy

            optbasis = bconv(min_bparams, func_dict["bpacker"])
            optbasis_energy = scf_dft_energy(optbasis, bsys1.atomstruc, atomstruc)

            save_output(outpath, basis, energy_small_basis, basis_ref, energy_ref_basis, optbasis,
                        optbasis_energy,
                        atomstruc, step[i], maxiter, method,f_rtol = f_rtol[i], optkwargs=minimize_kwargs)
        return optbasis

    elif type(atomstruc) is str and type(step) is float:
        print("elif type(atomstruc) is str and type(step) is float:")
        bsys1, bsys2, func_dict = Mole_minimizer(basis, basis_ref, atomstruc)

        writerpath, outpath = conf_output(basis, basis_ref, atomstruc, step, f_rtol, **out_kwargs)
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
                                                method=method,
                                                maxiter=maxiter,
                                                verbose=True,
                                                writer=writer, **minimize_kwargs)

        bsys1.get_SCF(atomstrucstr=atomstruc)
        energy_small_basis = bsys1.SCF.get_tot_energy
        energy_ref_basis = bsys2.SCF.get_tot_energy

        optbasis = bconv(min_bparams, func_dict["bpacker"])
        optbasis_energy = scf_dft_energy(optbasis, bsys1.atomstruc, atomstruc)

        save_output(outpath, basis, energy_small_basis, basis_ref, energy_ref_basis, optbasis, optbasis_energy,
                    atomstruc, step, maxiter, method,f_rtol = f_rtol, optkwargs=minimize_kwargs)

        return optbasis

    else:
       raise "Something went wrong"


if __name__ == "__main__":

    savepath =  os.path.dirname(os.path.realpath(__file__))
    _outf = os.path.join(savepath , "output")
    if not os.path.exists(_outf):
        os.mkdir(_outf)
    savepath = _outf

    ####################################################################################################################
    # setup arg parser throw terminal inputs
    ####################################################################################################################

    parser = argparse.ArgumentParser(description='Optimize Basis')

    ####################################################################################################################
    # configure atomic optb:
    ####################################################################################################################

    parser.add_argument("--mol",dest="atomstruc", type = str, nargs="+",
                        help='Name or set of names to define the atomic structure that you want to optimize.')

    ####################################################################################################################
    # configure basis to optimize and reference basis:
    ####################################################################################################################

    parser.add_argument('-b', "--basis", type=str, metavar="", nargs=2, default=["STO-3G", "cc-pvtz"],
                        help='names of the two basis first has to the basis,'
                             'you want to optimize. The second basis acts as reference basis.')

    ####################################################################################################################
    # set up xitorch.optimize.minimize
    ####################################################################################################################

    parser.add_argument("--maxiter", type=int, default = 1e6)
    parser.add_argument("-lr", "--steps", type=float, nargs='+', default = 2e-5,
                        help="learning rate (if set you opt. the same atomic structures for multiple learning rates."
                             " If len of atomstuc is the same as the len of -lr than each atomstruc get specific lr, "
                             "otherwise the each atomstruc will be trained with every lr)")
    parser.add_argument("--frtol", type= float, nargs='+', default = 1e-8,
                        help="The relative tolerance of the norm of the input (if set you opt. " \
                             "the same atomic structures for multiple frtol." \
                             " If lr and frtol are sets than you opt the they pair together")

    ####################################################################################################################
    # parse arguments
    ####################################################################################################################

    args = parser.parse_args()

    basis = args.basis[0]
    basis_ref = args.basis[1]
    step = args.steps
    f_rtol = args.frtol
    print(args.maxiter)
    maxiter = int(args.maxiter)

    atomstruc = args.atomstruc

    if atomstruc is None:
        atomstruc = "CH4"


    ####################################################################################################################
    # run actual optimization
    ####################################################################################################################

    optimize_basis(basis,basis_ref,atomstruc, step,maxiter = maxiter, output_path= savepath
                   ,minimize_kwargs = {"f_rtol" : f_rtol})