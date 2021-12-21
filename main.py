from optb.optimize_basis import *
from optb.data.preselected_avdata import elw417 ,elg2
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


if __name__ == "__main__":


    ####################################################################################################################
    # setup arg parser throw terminal inputs
    ####################################################################################################################

    parser = argparse.ArgumentParser(description='Optimize Basis')

    ####################################################################################################################
    # configure atomic optb:
    ####################################################################################################################

    parser.add_argument("--mol",dest="atomstruc", type = str, nargs="+", metavar="",
                        help='Name or set of names to define the atomic structure that you want to optimize.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--allw417", action = "store_true",
                        help='optimize every atom-structure that is available in the w4-17 database.')
    group.add_argument("--allg2", action="store_true",
                       help='optimize every atom-structure that is available in the g2 database.')
    ####################################################################################################################
    # configure basis to optimize and reference basis:
    ####################################################################################################################

    parser.add_argument('-b', "--basis", type=str, metavar="", nargs=2, default=["STO-3G", "cc-pvtz"],
                        help='names of the two basis first has to the basis,'
                             'you want to optimize. The second basis acts as reference basis.')

    ####################################################################################################################
    # set up xitorch.optimize.minimize
    ####################################################################################################################

    parser.add_argument("--maxiter", type=int, default = 1e6 , metavar="",help = "maximal learning iterations")
    parser.add_argument("-lr", "--steps", type=float, nargs='+' , metavar="", default = 2e-5,
                        help="learning rate (if set you opt. the same atomic structures for multiple learning rates."
                             " If len of atomstuc is the same as the len of -lr than each atomstruc get specific lr, "
                             "otherwise the each atomstruc will be trained with every lr)")
    parser.add_argument("--frtol", type= float, nargs='+', metavar="", default = 1e-8,
                        help="The relative tolerance of the norm of the input (if set you opt. " \
                             "the same atomic structures for multiple frtol." \
                             " If lr and frtol are sets than you opt the they pair together")

    ####################################################################################################################
    # setup save path
    ####################################################################################################################
    parser.add_argument("--saveto", type=str, nargs='+', metavar="", default= os.path.dirname(os.path.realpath(__file__)),
                        help="specify the path where you want to create the output folder (default is current path).")
    ####################################################################################################################
    # parse arguments
    ####################################################################################################################

    args = parser.parse_args()

    basis = args.basis[0]
    basis_ref = args.basis[1]
    step = args.steps
    f_rtol = args.frtol

    maxiter = int(args.maxiter)

    atomstruc = args.atomstruc

    if atomstruc is None:
        if args.allw417:
            atomstruc = elw417
            print(f"{len(elw417)} molecules will be optimized")
        elif args.allg2:
            atomstruc = elg2
            print(f"{len(elg2)} molecules will be optimized")
        else:
            #if you dont want to run the code over terminal change this one
            atomstruc = None
            step = 2e-6
            f_rtol = 2e-12

    ####################################################################################################################
    # create output folder to current path
    ####################################################################################################################
    savepath = args.saveto
    _outf = os.path.join(savepath, "output")
    if not os.path.exists(_outf):
        os.mkdir(_outf)
    savepath = _outf

    ####################################################################################################################
    # run actual optimization
    ####################################################################################################################
    optimize_basis(basis,basis_ref,atomstruc, step,maxiter = maxiter, output_path= savepath
                   ,minimize_kwargs = {"f_rtol" : f_rtol})