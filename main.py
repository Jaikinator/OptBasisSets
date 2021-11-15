from pyscf import gto, scf
import dqc
import torch
import xitorch as xt
import xitorch.optimize
from dqc.utils.datastruct import AtomCGTOBasis
from dqc.utils.misc import gaussian_int
import dqc.hamilton.intor as intor
from dqc.api.parser import parse_moldesc
import warnings #for warnings
import os
import basis_set_exchange as bse # basist set exchange libary
import numpy as np

import pymatgen.core.periodic_table as peri

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

cuda_device_checker()

########################################################################################################################
# configure torch tensor
########################################################################################################################

torch.set_printoptions(linewidth = 200)

########################################################################################################################
# first create class to configure system
########################################################################################################################

class dft_system:
    def __init__(self, basis :str, atomstruc : list, scf = True, requires_grad = False, rearrange = True ):
        """
        class to define the systems to optimize.
        :param element: array, str or number of the element in the periodic table
        :param basis: name of the basis to optimize if you want to use multiple basis do something like
                      basis = ["basis_1", "basis_2",...,"basis_N"] the array has to be the same length as atomstruc
        :param atomstruc: structure of the system input like
                            array([[element, [position]],
                                  [element , [position]],
                                  ...])
                            therefore position has to be the length 3 with float number for each axis position in
                            cartesian space. For example pos = [1.0, 1.0, 1.0]
        :param scf: if True you get the system as dqc as well as scf system.
        :param requires_grad: support gradient of torch.Tensor
        :param rearrange: if True the dqc basis will be rearranged to match the basis read by scf
        """
        self.basis = basis
        self.atomstruc = atomstruc
        self.atomstruc_dqc = self._arr_int_conv()
        self.element_dict = self._generateElementdict()
        self.elements = self._get_element_arr()

        if rearrange == False:
            if requires_grad == False :
                self.lbasis = self._loadbasis_dqc(requires_grad=False)  # loaded dqc basis
            else:
                self.lbasis = self._loadbasis_dqc()
        else:
            if requires_grad == False :
                self.lbasis = self._rearrange_basis_dqc(requires_grad=False)  # loaded dqc basis
            else:
                self.lbasis = self._rearrange_basis_dqc()

        if scf == True:

            self.mol = self._create_scf_Mol()

    ################################
    #dqc stuff:
    ################################

    def _arr_int_conv(self):
        """
        converts the atomsruc array to an str for dqc
        :return: str
        """
        return (str(self.atomstruc).replace("["," ")
                                .replace("]"," ")
                                .replace("'","")
                                .replace(" , " , ";")
                                .replace(",","")
                                .replace("  ", " "))

    def _loadbasis_dqc(self,**kwargs):
        """
        load basis from basissetexchange for the dqc module:
        https://www.basissetexchange.org/

        data of basis sets ist stored:
        /home/user/anaconda3/envs/env/lib/python3.8/site-packages/dqc/api/.database

        :param kwargs: requires_grad=False if basis is reference basis
        :return: array of basis
        """

        if type(self.basis) is str:
            bdict = {}
            for i in range(len(self.elements)):
                bdict[self.element_dict[self.elements[i]]] = dqc.loadbasis(f"{self.elements[i]}:{self.basis}", **kwargs)
            return bdict
        elif type(self.basis) is dict:
            bdict = {}
            for i in range(len(self.basis)):
                bdict[self.element_dict[self.elements[i]]] = dqc.loadbasis(f"{self.elements[i]}:{self.basis[self.element_dict[self.elements[i]]]}", **kwargs)
            return bdict
        else:
            print("do nothing to load basis")
            # basis = [dqc.loadbasis(f"{self.elements[i]}:{self.basis[i]}", **kwargs) for i in range(len(self.elements))]
            # return basis


    def _rearrange_basis_dqc(self, **kwargs):
        basis = self._loadbasis_dqc(**kwargs)
        bout = {}


        # read highest angmom for each atom:
        max_angmom_arr = []

        for key in basis:
            max_angmom_atom_arr = []
            for j in range(len(basis[key])):
                max_angmom_atom_arr.append(basis[key][j].angmom)
            max_angmom_arr.append(max(max_angmom_atom_arr))

        # now rearrange basis:
        cf_angmon = 0 # counter for angmom for each atom
        for key in basis:
            angmomc = 0
            inner = []

            while angmomc <= max_angmom_arr[cf_angmon]:
                for j in range(len(basis[key])):
                    if basis[key][j].angmom == angmomc:
                        inner.append(basis[key][j])
                angmomc += 1

            cf_angmon += 1
            bout[key] = inner

        warnings.warn("ATTENTION basis of dqc is rearranged")
        return  bout

    def _get_ovlp_dqc(self):

        elem = [self.atomstruc[i][0] for i in range(len(self.atomstruc))]
        basis = [self.lbasis[elem[i]] for i in range(len(elem))]
        atomzs, atompos = parse_moldesc(system.get_atomstruc_dqc())
        atombases = [AtomCGTOBasis(atomz=atomzs[i], bases=basis[i], pos=atompos[i]) for i in range(len(basis))]
        wrap = dqc.hamilton.intor.LibcintWrapper(
            atombases)  # creates an wrapper object to pass informations on lower functions

        return intor.overlap(wrap)

    @property
    def get_atomstruc_dqc(self):
        return self.atomstruc_dqc

    @property
    def get_basis(self):
        return self.lbasis

    @property
    def get_ovlp_dqc(self):
        return self._get_ovlp_dqc()
    ################################
    #scf staff:
    ################################
    def _get_molbasis_fparser_scf(self):
        """
        function to override scf basis function with data from:
        https://www.basissetexchange.org/
        If the file doesn't exist it will be downloaded in the NWChem format.
        :return: mol.basis object
        """
        folderpath = "NWChem_basis"
        if os.path.exists("NWChem_basis") == False:
            warnings.warn("No NWChem_basis folder it will be created.")
            os.mkdir("NWChem_basis")
            fullpath = os.path.abspath("NWChem_basis")
            print(f"NWChem_basis folder is created to {fullpath}")

        bdict = {}

        for i in range(len(self.elements)):
            if isinstance(self.basis,str):
                basisname = self.basis
            else:
                basisname = self.basis[self.element_dict[self.elements[i]]]

            if os.path.exists(f"{folderpath}/{basisname}.{self.elements[i]}.nw") == False:
                print(f"No basis {self.basis} found for {self.element_dict[self.elements[i]]}."
                      f" Try to get it from https://www.basissetexchange.org/")
                basis = bse.get_basis(self.basis, elements=[self.element_dict[self.elements[i]]], fmt="nwchem")
                fname = f"{folderpath}/{basisname}.{self.elements[i]}.nw"
                with open(fname, "w") as f:
                    f.write(basis)
                print(f"Downloaded to {os.path.abspath(fname)}")

            file = open(f"{folderpath}/{basisname}.{self.elements[i]}.nw").read()
            bdict[str(self.element_dict[self.elements[i]])] = gto.basis.parse(file)
        return bdict

    def _create_scf_Mol(self):
        """
        be aware here just basis as a string works
        :return: mol object
        """

        """
        basis path from 
        ~/anaconda3/envs/OptimizeBasisFunc/lib/python3.8/site-packages/pyscf/gto/basis
        """
        mol = gto.Mole()
        mol.atom = self.atomstruc
        mol.unit = 'Bohr'  # in Angstrom
        mol.verbose = 6
        mol.output = 'scf.out'
        mol.symmetry = False
        mol.basis  = self._get_molbasis_fparser_scf()
        #mol.basis = gto.basis.load(self.basis, 'Li')
        try:
            mol.spin = 0
            return mol.build()
        except:
            mol.spin = 1
            return mol.build()


    def _coeff_mat_scf(self):
        """
        just creates the coefficiency matrix for different input basis
        :return coefficient- matrix
        """

        mf = scf.RHF(self.mol)
        mf.kernel()
        return torch.tensor(mf.mo_coeff)

    def _get_occ(self):
        """
        get density matrix
        """

        mf = scf.RHF(self.mol)
        mf.kernel()
        return torch.tensor(mf.get_occ())

    def _get_ovlp_sfc(self):
        """
        create the overlap matrix of the pyscf system
        :return: torch.Tensor (torch.float64)
        """
        return torch.Tensor(self.mol.get_ovlp()).type(torch.float64)

    @property
    def get_basis_scf(self):
        return self._get_molbasis_fparser_scf()

    @property
    def get_coeff_scf(self):
        return self._coeff_mat_scf()

    @property
    def get_occ_coeff_scf(self):
        """
         coefficient- matrix of just the occupied orbitals
        """
        return self._coeff_mat_scf[self._coeff_mat_scf>0]

    @property
    def get_mol_scf(self):
        return self._create_scf_Mol()

    @property
    def get_occ_scf(self):
        return self._get_occ()

    @property
    def get_ovlp_scf(self):
        return self._get_ovlp_sfc()
    ################################
    #extra staff to configure class:
    ################################

    def _get_element_arr(self):
        """
        create array with all elements in the system
        """
        elements_arr  = [self.atomstruc[i][0] for i in range(len(atomstruc))]
        for i in range(len(elements_arr)):
            if type(elements_arr[i]) is str:
                elements_arr[i] = self.element_dict[elements_arr[i]]
        return elements_arr

    def _generateElementdict(self):
        """
        create a dictionary for the elements and their numbers in the periodic table.
        """
        elementdict = {}
        for i in range(1, 100):  # 100 Elements in Dict
            elementdict[peri.Element.from_Z(i).number] = str(peri.Element.from_Z(i))
            elementdict[ str(peri.Element.from_Z(i))] = peri.Element.from_Z(i).number
        return elementdict

    def ovlp_dqc_scf_eq(self, way : str):
        """
        checks if the overlap matrix of dqc and scf are equal
        :param all: if True outputs the full array
        :return: array or bool
        """

        pyscf_o = self._get_ovlp_sfc()

        dqc_o = self._get_ovlp_dqc()

        if way == "bool_arr":
            """
            return complete array of bool entries
            """
            return torch.isclose(dqc_o, pyscf_o)
        elif way == "bool":
            """
            just retruns a bool if both arrays equal
            """
            return torch.all(torch.isclose(dqc_o, pyscf_o, atol = 1e-03))
        elif way == "bool_dqc_scf_re":
            dqc_o_re = self._get_ovlp_dqc(rearrange=True)
            return torch.all(torch.isclose(dqc_o_re, pyscf_o, atol = 1e-03))

        elif way == "dqc":
            """
            compares rearrange array with not rearrange array
            """
            dqc_o_re = self._get_ovlp_dqc(rearrange = True) # rearranged basis array
            eq_arr = dqc_o == dqc_o_re

            Findex = torch.where(eq_arr == False)

            list_dqcore = dqc_o_re[Findex] #list of all elements of the rearranged Matrix where their are not equal
            list_dqco = dqc_o[Findex]  #list of all elements of the not rearranged Matrix where their are not equal

            counter_1 = 0 #couts elements of list_dqcore in list_dqco
            counter_2 = 0  #couts elements of list_dqco in list_dqcore

            for i in range(len(list_dqco)):
                for j in range(len(list_dqcore)):
                    if list_dqcore[j] == list_dqco[i]:
                        counter_1 += 1
                    if  list_dqco[j] == list_dqcore[i]:
                        counter_2 += 1

            # print(dqc_o)
            # print(dqc_o_re)
            # print(torch.all(torch.isclose(dqc_o_re, pyscf_o ,atol= 1e-05)))

            if counter_1 == counter_2:
                print("elements in the array are just on the wrong position")
                return True
            else:
                print("overlap wrong calculated")
                return False

        elif way == "check_dqc_scf":
            """
            compares the overlap of pyscf and dqc and compares whether 
            the same elements are present in both matrices.
            """

            eq_arr = dqc_o == pyscf_o

            Findex = torch.where(eq_arr == False)

            list_dqcore = pyscf_o[Findex]  # list of all elements of the rearranged Matrix where their are not equal
            list_dqco = dqc_o[Findex]  # list of all elements of the not rearranged Matrix where their are not equal

            counter_1 = 0  # couts elements of list_dqcore in list_dqco
            counter_2 = 0  # couts elements of list_dqco in list_dqcore

            for i in range(len(list_dqco)):
                for j in range(len(list_dqcore)):
                    if list_dqcore[j] == list_dqco[i]:
                        counter_1 += 1
                    if list_dqco[j] == list_dqcore[i]:
                        counter_2 += 1

            if counter_1 == counter_2:
                print("elements in the array are just on the wrong position")
                return True
            else:
                print("overlap wrong calculated")
                return False

    ################################
    # relevant export dict:
    ################################
    def fcn_dict(self,ref_system):
        """
        def dictionary to input in fcn
        :param ref_system: class obj for the reference basis
        :return:
        """

        basis = self.lbasis
        bpacker = xt.Packer(basis)
        bparams = bpacker.get_param_tensor()

        basis_ref = ref_system.lbasis
        bpacker_ref = xt.Packer(basis_ref)
        bparams_ref = bpacker_ref.get_param_tensor()

        return {"bparams": bparams,
                "bpacker": bpacker,
                "bparams_ref": bparams_ref,
                "bpacker_ref": bpacker_ref,
                "atomstruc_dqc" : self.atomstruc_dqc,
                "atomstruc" : self.atomstruc,
                "coeffM" : ref_system.get_coeff_scf,
                "occ_scf" : self._get_occ()}


########################################################################################################################
# now do the actual calculations
########################################################################################################################
def blister(atomstruc : list, basis : dict, refbasis :dict):
    """
    function to convert the basis dict based on the given atomstruc array
    :param atomstruc: atomstruc array
    :param basis: basis dict first basis
    :param refbasis: reference basis dictionary
    :return:
    """

    elem = [atomstruc[i][0] for i in range(len(atomstruc))]
    b_arr = [basis[elem[i]] for i in range(len(elem))]
    bref_arr = [refbasis[elem[i]] for i in range(len(elem))]
    return b_arr + bref_arr


# def wfnormalize_(CGTOB):
#     """
#     copy of ~/dqc/utils/datastruct.py wfnormalize_ of CGTBasis Class
#     :param CGTOBasis: Class Object
#     :return: normalized basis set coeff.
#     """
#     # wavefunction normalization
#     # the normalization is obtained from CINTgto_norm from
#     # libcint/src/misc.c, or
#     # https://github.com/sunqm/libcint/blob/b8594f1d27c3dad9034984a2a5befb9d607d4932/src/misc.c#L80
#
#     # Please note that the square of normalized wavefunctions do not integrate
#     # to 1, but e.g. for s: 4*pi, p: (4*pi/3)
#
#     # if the basis has been normalized before, then do nothing
#     #
#     # if self.normalized:
#     #     return self
#
#     coeffs = CGTOB.coeffs
#
#     # normalize to have individual gaussian integral to be 1 (if coeff is 1)
#     if not CGTOB.normalized:
#         coeffs = coeffs / torch.sqrt(gaussian_int(2 * CGTOB.angmom + 2, 2 * CGTOB.alphas))
#     # normalize the coefficients in the basis (because some basis such as
#     # def2-svp-jkfit is not normalized to have 1 in overlap)
#     ee = CGTOB.alphas.unsqueeze(-1) + CGTOB.alphas.unsqueeze(-2)  # (ngauss, ngauss)
#     ee = gaussian_int(2 * CGTOB.angmom + 2, ee)
#     s1 = 1 / torch.sqrt(torch.einsum("a,ab,b", coeffs, ee, coeffs))
#     coeffs = coeffs * s1
#
#     CGTOB.coeffs = coeffs
#     CGTOB.normalized = True
#     return CGTOB


def _num_gauss(basis : list, restbasis : list, atomstruc = False):
    """
    calc the number of primitive Gaussian's in a basis set so that the elements of an overlap matrix can be defined.
    :param basis: list
        basis to get optimized
    :param restbasis: list
        optimized basis
    :return: int
        number of elements of each basis set

    """
    n_basis =  0
    n_restbasis = 0


    for i in range(len(atomstruc)):
        for el in basis[atomstruc[i][0]]:
            n_basis += 2 * el.angmom + 1

        for el in restbasis[atomstruc[i][0]]:
            n_restbasis += 2 * el.angmom + 1
    return [n_basis, n_restbasis]



def _cross_selcet(crossmat : torch.Tensor, num_gauss : list, direction : str ):
    """
    select the cross overlap matrix part.
    The corssoverlap is defined by the overlap between to basis sets.
    For example is b1 the array of the first basis set which will be optimized and
    b2 the array of the second basis set which is the reference basis
    Then is the crossoveralp S:
        S  = [b1*b1 , b1*b2] = [S_11 , S_12]
             [b2*b1 , b2*b2]   [S_21 , S_22]
    you can choose between:
        - basis overlap (S_11)
        - restbasis overlap (S_22)
        - horizontal crossoverlap (S_12)
        - vertical crossoverlap (S_21)
    :param crossmat: crossoverlap mat
    :param num_gauss: number of gaussians in two basis
    :param direction: type str to select the specific crossoverlap
                      you can choose between: S_11, S_12_,S_21, S_22
    :return: torch.tensor
    returns the cross overlap matrix between the new and the old basis func
    """

    if direction == "S_11" :
        return crossmat[0:num_gauss[0],0:num_gauss[0]]
    elif direction == "S_12":
        return crossmat[num_gauss[0]:, 0:(len(crossmat) - num_gauss[1])]
    elif direction == "S_21":
        return crossmat[0:num_gauss[0], num_gauss[0]:]
    elif direction == "S_22":
        return crossmat[num_gauss[0]:, num_gauss[0]:]
    else:
        raise UnicodeError("no direction specified")

def _crossoverlap(atomstruc : str, basis : list):

    # change normalization state so that it will be normalized by AtomCGTOBasis again
    # change_norm_state(basis)

    # calculate cross overlap matrix:

    atomzs, atompos = parse_moldesc(atomstruc)

    # atomzs : Atomic number (torch.tensor); len: number of Atoms
    # atompos : atom positions tensor of shape (3x len: number of Atoms )
    # now double both atom information to use is for the second basis set

    atomzs = torch.cat([atomzs, atomzs])
    atompos = torch.cat([atompos, atompos])
    atombases = [AtomCGTOBasis(atomz=atomzs[i], bases=basis[i], pos=atompos[i]) for i in range(len(basis))]
    # creats an list with AtomCGTOBasis object for each atom (including  all previous informations in one array element)
    wrap = dqc.hamilton.intor.LibcintWrapper(
        atombases)  # creates an wrapper object to pass informations on lower functions

    return intor.overlap(wrap)

def projection(coeff : torch.Tensor, colap : torch.Tensor, num_gauss : torch.Tensor):
    """
    Calculate the Projection from the old to the new Basis:
         P = C^T S_12 S⁻¹_11 S_21 C
    :param coeff: coefficient matrix of the bigger reference basis calculated by pyscf (C Matrix in eq.)
    :param colap: crossoverlap Matrix (S and his parts)
    :param num_gauss: array with length of the basis sets
    :return: Projection Matrix
    """

    S_12 = _cross_selcet(colap, num_gauss, "S_12")
    S_21 = _cross_selcet(colap, num_gauss, "S_21")
    S_11 = _cross_selcet(colap, num_gauss, "S_11")

    s21_c = torch.matmul(S_21, coeff)
    s11_s21c = torch.matmul(torch.inverse(S_11), s21_c)
    s21_s11s12c = torch.matmul(S_12, s11_s21c)
    P = torch.matmul(coeff.T, s21_s11s12c)
    return P

########################################################################################################################
# test function
########################################################################################################################

def fcn(bparams : torch.Tensor, bpacker: xitorch._core.packer.Packer
        , bparams_ref: torch.Tensor, bpacker_ref: xitorch._core.packer.Packer
        , atomstruc_dqc: str, atomstruc : list, coeffM : torch.Tensor, occ_scf : torch.Tensor):

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

    ref_basis = bpacker_ref.construct_from_tensor(bparams_ref)
    if len(atomstruc) !=  len(basis):
        num_gauss = _num_gauss(basis, ref_basis, atomstruc)
    else:
        num_gauss = _num_gauss(basis, ref_basis)
    basis_cross = blister(atomstruc,basis,ref_basis)

    colap = _crossoverlap(atomstruc_dqc, basis_cross)
    # maximize overlap

    _projt = projection(coeffM,colap,num_gauss)
    occ_scf = occ_scf[occ_scf > 0]

    _projection = torch.zeros((_projt.shape[0], occ_scf.shape[0]), dtype = torch.float64)

    for i in range(len(occ_scf)):
        _projection[:,i] = _projt[:,i] * occ_scf[i]

    return -torch.trace(_projection)/torch.sum(occ_scf)

if __name__ == "__main__":

    ####################################################################################################################
    # configure atomic system:
    ####################################################################################################################

    atomstruc = [['H', [0.5, 0.0, 0.0]],
                 ['O', [-0.5, 0.0, 0.0 ]],
                 ['H', [0.0, 1.0, 0.0]]]

    ####################################################################################################################
    # configure basis to optimize:
    ####################################################################################################################

    basis = "3-21G"
    system = dft_system(basis, atomstruc)
    ####################################################################################################################
    # configure reference basis:
    ####################################################################################################################
    basis_ref = "cc-pvdz"
    system_ref = dft_system(basis_ref, atomstruc)

    ####################################################################################################################

    func_dict = system.fcn_dict(system_ref)

    min_bparams = xitorch.optimize.minimize(fcn,  func_dict["bparams"], (func_dict["bpacker"],
                                                                        func_dict["bparams_ref"],
                                                                        func_dict["bpacker_ref"],
                                                                        func_dict["atomstruc_dqc"],
                                                                        func_dict["atomstruc"],
                                                                        func_dict["coeffM"],
                                                                        func_dict["occ_scf"],),step = 2e-6, method = "gd",maxiter = 1000, verbose = True)# ,method = "Adam"

    print(f"{basis}: \t", func_dict["bparams"],"len: ",len(func_dict["bparams"]))
    print(f"{basis_ref}:\t", func_dict["bparams_ref"],"len: ",len(func_dict["bparams_ref"]))
    print("Opt params:\t", min_bparams,"len: ",len(min_bparams))



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
Basis Names:
    3-21G
    cc-pvdz

"""

"""
To configure verbose options on which step something should be printet look at 
~/xitorch/_impls/optimize/minimizer.py line 183

"""