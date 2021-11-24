from pyscf import gto, scf, dft
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

#get atom pos:
from ase.build import molecule
from ase.collections import g2

import pymatgen.core.periodic_table as peri

#try paralaziation:
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

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
        self.basis = basis  # just str of basis
        self.atomstruc = atomstruc
        self.atomstruc_dqc = self._arr_int_conv()
        self.element_dict = self._generateElementdict()
        self.elements = self._get_element_arr()

        if rearrange == False:
            if requires_grad == False :
                basisparam  = self._loadbasis_dqc(requires_grad=False)  # loaded dqc basis
            else:
                basisparam  = self._loadbasis_dqc()
        else:
            if requires_grad == False :
                basisparam = self._rearrange_basis_dqc(requires_grad=False)  # loaded dqc basis
            else:
                basisparam = self._rearrange_basis_dqc()

        self.lbasis = basisparam  # basis with parameters in dqc format

        if scf == True:
            self.molbasis = None
            self.mol = self._create_scf_Mol()
            self.dft = self.scf_dft_calc()

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
        atomzs, atompos = parse_moldesc(self.atomstruc_dqc)
        atombases = [AtomCGTOBasis(atomz=atomzs[i], bases=basis[i], pos=atompos[i]) for i in range(len(basis))]
        wrap = dqc.hamilton.intor.LibcintWrapper(
            atombases)  # creates an wrapper object to pass informations on lower functions

        return intor.overlap(wrap)

    @property
    def get_atomstruc_dqc(self):
        return self.atomstruc_dqc

    @property
    def get_basis_dqc(self):
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
            bdict[str(self.element_dict[self.elements[i]])] = gto.basis.parse(file,optimize=False)
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
        mol.unit = 'Bohr'  # Angstrom Bohr
        mol.verbose = 6
        mol.output = 'scf.out'
        mol.symmetry = False
        mol.basis  = self.molbasis

        try:
            mol.spin = 0
            return mol.build()
        except:
            mol.spin = 1
            return mol.build()

    def scf_dft_calc(self):
        mf = dft.RKS(self.mol)
        mf.kernel()
        mf.xc = "GGA_X_B88 + GGA_C_LYP" #'b3lyp'
        return mf

    def _coeff_mat_scf(self):
        """
        just creates the coefficiency matrix for different input basis
        :return coefficient- matrix
        """
        return torch.tensor(self.dft.mo_coeff)

    def _get_occ(self):
        """
        get density matrix
        """
        return torch.tensor(self.dft.get_occ())

    def _get_ovlp_sfc(self):
        """
        create the overlap matrix of the pyscf system
        :return: torch.Tensor (torch.float64)
        """
        return torch.Tensor(self.mol.get_ovlp()).type(torch.float64)

    @property
    def molbasis(self):
        """
        getter for molbasis obj.
        """
        return self._molbasis

    @molbasis.setter
    def molbasis(self, var = None):
        """
        setter for molbasis
        :param var: is none basis is going to be created from NWCHem file
        :return: basis dict
        """
        if var == None:
            self._molbasis = self._get_molbasis_fparser_scf()
        else:
            self._molbasis = var
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
    @property
    def get_tot_energy_scf(self):
        return self.scf_dft_calc().energy_tot()
    ################################
    #extra staff to configure class:
    ################################

    def _get_element_arr(self):
        """
        create array with all elements in the system
        """
        elements_arr  = [self.atomstruc[i][0] for i in range(len(self.atomstruc))]
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
    def _num_gauss(self,refsystem):
        """
        calc the number of primitive Gaussian's in a basis set so that the elements of an overlap matrix can be defined.
        :param basis: list
            basis to get optimized
        :param restbasis: list
            optimized basis
        :return: int
            number of elements of each basis set

        """
        n_basis = 0
        n_restbasis = 0

        for i in range(len(self.atomstruc)):
            for el in self.lbasis[self.atomstruc[i][0]]:
                n_basis += 2 * el.angmom + 1

            for el in refsystem.lbasis[self.atomstruc[i][0]]:
                n_restbasis += 2 * el.angmom + 1
        return torch.tensor([n_basis, n_restbasis])

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
        ref_basis = bpacker_ref.construct_from_tensor(bparams_ref)
        return {"bparams": bparams,
                "bpacker": bpacker,
                "ref_basis": ref_basis,
                "atomstruc_dqc" : self.atomstruc_dqc,
                "atomstruc" : self.atomstruc,
                "coeffM" : ref_system.get_coeff_scf,
                "occ_scf" : ref_system._get_occ(),
                "num_gauss" :  self._num_gauss(ref_system)}

class system_ase(dft_system):

    def __init__(self, basis :str, atomstrucstr : str, scf = True, requires_grad = False, rearrange = True ):
        self.atomstrucstr = atomstrucstr
        self.atomstruc = self.create_atomstruc_from_ase()

        super().__init__(basis,self.atomstruc,  scf, requires_grad, rearrange)

    def create_atomstruc_from_ase(self):
        """
        creates atomstruc from ase database.
        :param atomstruc: molecule string
        :return: array like [element, [x,y,z], ...]
        """
        chem_symb = molecule(self.atomstrucstr).get_chemical_symbols()
        atompos =  molecule(self.atomstrucstr).get_positions()

        arr = []
        for i in range(len(chem_symb)):
            arr.append([chem_symb[i], list(atompos[i])])
        return arr

    # def _create_scf_Mol(self):
    #     """
    #     be aware here just basis as a string works
    #     :return: mol object
    #     """
    #
    #     """
    #     basis path from
    #     ~/anaconda3/envs/OptimizeBasisFunc/lib/python3.8/site-packages/pyscf/gto/basis
    #     """
    #     mol = gto.Mole()
    #     mol.atom = self.atomstruc
    #     mol.unit = 'Bohr'  # in Angstrom
    #     mol.verbose = 6
    #     mol.output = 'scf.out'
    #     mol.symmetry = False
    #     mol.basis = self._get_molbasis_fparser_scf()
    #     mol.build()
    #     # print(mol.nelec)
    #     # mol.charge = sum(molecule(self.atomstrucstr).get_initial_magnetic_moments())
    #     # mol.spin = sum(molecule(self.atomstrucstr).get_initial_charges())
    #     # print(mol.nelec)
    #     return mol

def system_init(atomstruc, basis1, basis2, **kwargs):
    """
    get system classes and dict for the optimization back.
    """

    if type(atomstruc) is list:
        system = dft_system(basis1, atomstruc, scf = False, **kwargs)
        system_ref = dft_system(basis2, atomstruc,**kwargs)
    elif type(atomstruc) is str:
        system = system_ase(basis1, atomstruc, scf = False,**kwargs)
        system_ref = system_ase(basis2, atomstruc,**kwargs)
    else:
        print("pls report")

    # print("Molecule structure:")
    #
    # [print(system.atomstruc[i]) for i in range(len(system.atomstruc))]

    # from collections import Counter
    #
    # print("Number of each Atom: ",Counter(molecule(atomstruc).get_chemical_symbols()))

    return system, system_ref,system.fcn_dict(system_ref)

def scf_basis_from_dqc(bparams, bpacker ):
    """
    creates a pyscf type basis dict out of an dqc basis input
     :param bparams: torch.tensor with coeff of basis set
    :param bpacker: xitorch._core.packer.Packer object to create the CGTOBasis out of the bparams
    :return: dict where each element gots his own basis arr
    """
    basis = bpacker.construct_from_tensor(bparams)
    bdict = {}

    for el in basis:
        arr = []
        for CGTOB in basis[el]:
            innerarr = [CGTOB.angmom]
            for al,co in zip(CGTOB.alphas, CGTOB.coeffs):
                innerarr.append([float(al), float(co)])
            arr.append(innerarr)
        bdict[el] = arr
    return bdict
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

def _cross_selcet(crossmat : torch.Tensor, num_gauss : torch.Tensor, direction : str ):
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

    colap = _crossoverlap(atomstruc_dqc, basis_cross)
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

    mf = scf.RHF(mol)
    mf.kernel()
    mf.xc = 'b3lyp'
    return mf.energy_tot()



if __name__ == "__main__":


    ####################################################################################################################
    # configure atomic system:
    ####################################################################################################################

    # atomstruc = [['H', [0.5, 0.0, 0.0]]]
    #              # ['O', [-0.5, 0.0, 0.0 ]],
    #              # ['H', [0.0, 1.0, 0.0]]]

    atomstruc = "CH4"

    ####################################################################################################################
    # configure basis to optimize:
    ####################################################################################################################

    basis = "STO-3G"
    #system = dft_system(basis, atomstruc)

    ####################################################################################################################
    # configure reference basis:
    ####################################################################################################################
    basis_ref = "cc-pvtz"
    #system_ref = dft_system(basis_ref, atomstruc)

    ####################################################################################################################

    bsys1, bsys2, func_dict = system_init(atomstruc,basis,basis_ref) #system.fcn_dict(system_ref)

    print("\n start optimization")

    min_bparams = xitorch.optimize.minimize(fcn,  func_dict["bparams"], (func_dict["bpacker"],
                                                                        func_dict["ref_basis"],
                                                                        func_dict["atomstruc_dqc"],
                                                                        func_dict["atomstruc"],
                                                                        func_dict["coeffM"],
                                                                        func_dict["occ_scf"],
                                                                        func_dict["num_gauss"], ),step = 2e-6,method = "Adam", maxiter = 100, verbose = True)# ,method = "Adam"

    testsystem = system_ase(basis, atomstruc)
    testsystem.get_ovlp_dqc
    initenergy = testsystem.get_tot_energy_scf
    print(f"total energy scf with {basis} as initial basis:\n",
          initenergy)
    print(f"total energy scf with {basis_ref} as reference basis:\n",
          bsys2.get_tot_energy_scf)
    # print("energy after optimization of basis as basis set:\n",
    #       scf_dft_energy(scf_basis_from_dqc(min_bparams, func_dict["bpacker"]), atomstruc))
    # print(dft_system(basis, atomstruc).get_basis_scf)
    # print(scf_basis_from_dqc(min_bparams, func_dict["bpacker"]))

    basist = func_dict["bpacker"].construct_from_tensor(min_bparams)

    atomzs, atomposs = dqc.parse_moldesc(system_ase(basis, atomstruc).atomstruc_dqc)
    mol = dqc.Mol((atomzs, atomposs), basis=basist)


    xcm = "GGA_X_B88 + GGA_C_LYP"

    qc = dqc.KS(mol, xc = xcm).run()
    ene = qc.energy()
    print("dqc ene: ", ene)

    mol2 = dqc.Mol((atomzs, atomposs), basis="STO-3G")

    qc2 = dqc.KS(mol2, xc=xcm).run()
    ene2 = qc2.energy()
    print("dqc ene STO-3G",ene2)


    mol3 = dqc.Mol((atomzs, atomposs), basis="cc-pvtz")

    qc2 = dqc.KS(mol3, xc=xcm).run()
    ene2 = qc2.energy()
    print("dqc ene cc-pvtz", ene2)
    # print(f"{basis}: \t", func_dict["bparams"],"len: ",len(func_dict["bparams"]))
    # print(f"{basis_ref}:\t", func_dict["bparams_ref"],"len: ",len(func_dict["bparams_ref"]))
    # print("Opt params:\t", min_bparams,"len: ",len(min_bparams))






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
"""
['PH3', 'P2', 'CH3CHO', 'H2COH', 'CS', 'OCHCHO', 'C3H9C', 'CH3COF', 
'CH3CH2OCH3', 'HCOOH', 'HCCl3', 'HOCl', 'H2', 'SH2','C2H2', 'C4H4NH', 
'CH3SCH3', 'SiH2_s3B1d', 'CH3SH','CH3CO', 'CO', 'ClF3', 'SiH4', 'C2H6CHOH',
'CH2NHCH2', 'isobutene','HCO', 'bicyclobutane', 'LiF', 'Si', 'C2H6', 'CN', 
'ClNO', 'S', 'SiF4', 'H3CNH2', 'methylenecyclopropane', 'CH3CH2OH','F', 'NaCl',
'CH3Cl', 'CH3SiH3', 'AlF3', 'C2H3', 'ClF', 'PF3', 'PH2', 'CH3CN', 'cyclobutene', 
'CH3ONO', 'SiH3', 'C3H6_D3h', 'CO2', 'NO', 'trans-butane', 'H2CCHCl', 'LiH', 'NH2',
'CH', 'CH2OCH2', 'C6H6', 'CH3CONH2', 'cyclobutane', 'H2CCHCN', 'butadiene', 'C', 
'H2CO', 'CH3COOH', 'HCF3', 'CH3S', 'CS2', 'SiH2_s1A1d', 'C4H4S', 'N2H4', 'OH', 'CH3OCH3',
'C5H5N', 'H2O', 'HCl', 'CH2_s1A1d', 'CH3CH2SH', 'CH3NO2', 'Cl', 'Be', 'BCl3', 'C4H4O', 'Al',
'CH3O', 'CH3OH', 'C3H7Cl', 'isobutane', 'Na', 'CCl4', 'CH3CH2O', 'H2CCHF', 'C3H7', 'CH3', 'O3', 
'P', 'C2H4', 'NCCN', 'S2', 'AlCl3', 'SiCl4', 'SiO', 'C3H4_D2d', 'H', 'COF2', '2-butyne', 
'C2H5', 'BF3', 'N2O', 'F2O', 'SO2','H2CCl2', 'CF3CN', 'HCN', 'C2H6NH', 'OCS', 'B', 'ClO', 
'C3H8', 'HF', 'O2', 'SO', 'NH', 'C2F4', 'NF3', 'CH2_s3B1d','CH3CH2Cl', 'CH3COCl', 'NH3', 
'C3H9N', 'CF4', 'C3H6_Cs', 'Si2H6', 'HCOOCH3', 'O', 'CCH', 'N', 'Si2', 'C2H6SO', 'C5H8', 
'H2CF2', 'Li2', 'CH2SCH2', 'C2Cl4', 'C3H4_C3v', 'CH3COCH3', 'F2', 'CH4', 'SH', 'H2CCO', 
'CH3CH2NH2', 'Li', 'N2', 'Cl2', 'H2O2', 'Na2', 'BeH', 'C3H4_C2v', 'NO2']
"""


def test(molecule):
    basis = "STO-3G"
    basis_ref = "cc-pvdz"
    sys1,sys2,func_dict = system_init(molecule, basis, basis_ref)
    min_bparams = xitorch.optimize.minimize(fcn, func_dict["bparams"], (func_dict["bpacker"],
                                                                        func_dict["bparams_ref"],
                                                                        func_dict["bpacker_ref"],
                                                                        func_dict["atomstruc_dqc"],
                                                                        func_dict["atomstruc"],
                                                                        func_dict["coeffM"],
                                                                        func_dict["occ_scf"],), step=2e-5,
                                            method="Adam", maxiter=1000, verbose=True)  # ,method = "Adam", verbose=True
    return func_dict, min_bparams

# elements = ["H2O", "CH4","SiH4", "N2O", "methylenecyclopropane"] #, "SiH4", "N2O", "methylenecyclopropane"
# testdict = {}
# for i in elements:
#     sysres,res = test(i)
#     testdict[f"sys_{i}"] = sysres
#     testdict[i] =  res
# #
# with ThreadPoolExecutor(max_workers=12) as executor:
#     result = executor.map(test,elements)
#     resopt1 = executor.submit(test, "H2O")
#     resopt2 =executor.submit(test, "CH4")
#     resopt3 =executor.submit(test, "SiH4")
#     resopt4 =executor.submit(test, "N2O")
#     resopt5 =executor.submit(test, "methylenecyclopropane")
#
# executor.shutdown(wait = True)
#
#
# basis = "STO-3G"
# basis_ref = "cc-pvdz"
#
# sys1 = system_ase(basis, "H2O").get_tot_energy_scf
# sys1_ref = system_ase(basis_ref, "H2O").get_tot_energy_scf
# print(f"{elements[0]}:\n\tenergy {basis}: {sys1}\n\tenergy {basis_ref}: {sys1_ref}\n\t energy opt:",
#       scf_dft_energy(scf_basis_from_dqc(resopt1.result()[1], resopt1.result()[0]["bpacker"]),"H2O"))
#
# sys2 = system_ase(basis, "CH4").get_tot_energy_scf
# sys2_ref = system_ase(basis_ref, "CH4").get_tot_energy_scf
# print(f"{elements[1]}:\n\tenergy {basis}: {sys2}\n\tenergy {basis_ref}: {sys2_ref}\n\t energy opt:",
#       scf_dft_energy(scf_basis_from_dqc(resopt2.result()[1], resopt2.result()[0]["bpacker"]),"CH4"))
#
# sys3 = system_ase(basis, "SiH4").get_tot_energy_scf
# sys3_ref = system_ase(basis_ref, "SiH4").get_tot_energy_scf
# print(f"{elements[2]}:\n\tenergy {basis}: {sys3}\n\tenergy {basis_ref}: {sys3_ref}\n\t energy opt:",
#       scf_dft_energy(scf_basis_from_dqc(resopt3.result()[1], resopt3.result()[0]["bpacker"]),"SiH4"))
#
# sys4 = system_ase(basis, "N2O").get_tot_energy_scf
# sys4_ref = system_ase(basis_ref, "N2O").get_tot_energy_scf
# print(f"{elements[3]}:\n\tenergy {basis}: {sys4}\n\tenergy {basis_ref}: {sys4_ref}\n\t energy opt:",
#       scf_dft_energy(scf_basis_from_dqc(resopt4.result()[1], resopt4.result()[0]["bpacker"]),"N2O"))
#
# sys5 = system_ase(basis, "methylenecyclopropane").get_tot_energy_scf
# sys5_ref = system_ase(basis_ref, "methylenecyclopropane").get_tot_energy_scf
# print(f"{elements[4]}:\n\tenergy {basis}: {sys5}\n\tenergy {basis_ref}: {sys5_ref}\n\tenergy opt:",
#       scf_dft_energy(scf_basis_from_dqc(resopt5.result()[1], resopt5.result()[0]["bpacker"]),"methylenecyclopropane"))
#
#
#
#
# basis = "STO-3G"
# basis_ref = "cc-pvtz"
#
# sys1 = system_ase(basis, "H2O").get_tot_energy_scf
# sys1_ref = system_ase(basis_ref, "H2O").get_tot_energy_scf
# print(f"{elements[0]}:\n\tenergy {basis}: {sys1}\n\tenergy {basis_ref}: {sys1_ref}\n\t energy opt:",
#       scf_dft_energy(scf_basis_from_dqc(testdict["H2O"], testdict["sys_H2O"]["bpacker"]),"H2O"))
#
# sys2 = system_ase(basis, "CH4").get_tot_energy_scf
# sys2_ref = system_ase(basis_ref, "CH4").get_tot_energy_scf
# print(f"{elements[1]}:\n\tenergy {basis}: {sys2}\n\tenergy {basis_ref}: {sys2_ref}\n\t energy opt:",
#       scf_dft_energy(scf_basis_from_dqc(testdict["CH4"], testdict["sys_CH4"]["bpacker"]),"CH4"))
#
# sys3 = system_ase(basis, "SiH4").get_tot_energy_scf
# sys3_ref = system_ase(basis_ref, "SiH4").get_tot_energy_scf
# print(f"{elements[2]}:\n\tenergy {basis}: {sys3}\n\tenergy {basis_ref}: {sys3_ref}\n\t energy opt:",
#       scf_dft_energy(scf_basis_from_dqc(testdict["SiH4"], testdict["sys_SiH4"]["bpacker"]),"SiH4"))
#
# sys4 = system_ase(basis, "N2O").get_tot_energy_scf
# sys4_ref = system_ase(basis_ref, "N2O").get_tot_energy_scf
# print(f"{elements[3]}:\n\tenergy {basis}: {sys4}\n\tenergy {basis_ref}: {sys4_ref}\n\t energy opt:",
#       scf_dft_energy(scf_basis_from_dqc(testdict["N2O"], testdict["sys_N2O"]["bpacker"]),"N2O"))
#
# sys5 = system_ase(basis, "methylenecyclopropane").get_tot_energy_scf
# sys5_ref = system_ase(basis_ref, "methylenecyclopropane").get_tot_energy_scf
# print(f"{elements[4]}:\n\tenergy {basis}: {sys5}\n\tenergy {basis_ref}: {sys5_ref}\n\tenergy opt:",
#       scf_dft_energy(scf_basis_from_dqc(testdict["methylenecyclopropane"], testdict["sys_methylenecyclopropane"]["bpacker"]),"methylenecyclopropane"))