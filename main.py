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


import pymatgen.core.periodic_table as peri

#try paralaziation:
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

from optb.Mole import *
from optb.projection import *
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
# first create class to configure optb
########################################################################################################################

def scf_basis_from_dqc(bparams, bpacker ):
    """
    creates a pyscf type basis dict out of an dqc basis input
     :param bparams: torch.tensor with coeff of basis set
    :param bpacker: xitorch._core.packer.Packer object to create the CGTOBasis out of the bparams
    :return: dict where each element gots his own basis arr
    """
    basis = bpacker.construct_from_tensor(bparams)
    def wfnormalize_(CGTOB):
        """
        Normalize coefficients from the unnormalized state from dqc
        :param CGTOB:
        :return:
        """
        # wavefunction normalization
        # the normalization is obtained from CINTgto_norm from
        # libcint/src/misc.c, or
        # https://github.com/sunqm/libcint/blob/b8594f1d27c3dad9034984a2a5befb9d607d4932/src/misc.c#L80

        # Please note that the square of normalized wavefunctions do not integrate
        # to 1, but e.g. for s: 4*pi, p: (4*pi/3)

        # if the basis has been normalized before, then do nothing

        # if self.normalized:
        #     return self

        coeffs = CGTOB.coeffs

        # normalize to have individual gaussian integral to be 1 (if coeff is 1)

        coeffs = coeffs * torch.sqrt(gaussian_int(2 * CGTOB.angmom + 2, 2 * CGTOB.alphas))
        # normalize the coefficients in the basis (because some basis such as
        # def2-svp-jkfit is not normalized to have 1 in overlap)
        ee = CGTOB.alphas.unsqueeze(-1) + CGTOB.alphas.unsqueeze(-2)  # (ngauss, ngauss)
        ee = gaussian_int(2 * CGTOB.angmom + 2, ee)
        s1 = 1 / torch.sqrt(torch.einsum("a,ab,b", coeffs, ee, coeffs))
        coeffs = coeffs * s1

        CGTOB.coeffs = coeffs
        CGTOB.normalized = True
        return CGTOB

    bdict = {}

    for el in basis:
        arr = []
        for CGTOB in basis[el]:
            CGTOB =  wfnormalize_(CGTOB)
            innerarr = [CGTOB.angmom]
            for al,co in zip(CGTOB.alphas, CGTOB.coeffs):
                innerarr.append([float(al), float(co)])
            arr.append(innerarr)
        bdict[el] = arr
    return bdict

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
    mf.xc = 'GGA_X_B88 + GGA_C_LYP'
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
    basis_ref = "cc-pvdz"
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
          scf_dft_energy(scf_basis_from_dqc(min_bparams, func_dict["bpacker"]), atomstruc))

    print("basis scf initial:", Mole_ase(basis, atomstruc).SCF.get_basis)
    print("basis after one step:", scf_basis_from_dqc(min_bparams, func_dict["bpacker"]))

    basist = func_dict["bpacker"].construct_from_tensor(min_bparams)

    #print(Mole_minimizer(basis,basis_ref, atomstruc))
    #
    # atomzs, atomposs = dqc.parse_moldesc(system_ase(basis, atomstruc).atomstruc_dqc)
    # mol = dqc.Mol((atomzs, atomposs), basis=basist)
    #
    #
    # xcm = "GGA_X_B88 + GGA_C_LYP"
    #
    # qc = dqc.KS(mol, xc = xcm).run()
    # ene = qc.energy()
    # print("dqc ene: ", ene)
    #
    # mol2 = dqc.Mol((atomzs, atomposs), basis="STO-3G")
    #
    # qc2 = dqc.KS(mol2, xc=xcm).run()
    # ene2 = qc2.energy()
    # print("dqc ene STO-3G",ene2)
    #
    #
    # mol3 = dqc.Mol((atomzs, atomposs), basis="cc-pvtz")
    #
    # qc2 = dqc.KS(mol3, xc=xcm).run()
    # ene2 = qc2.energy()
    # print("dqc ene cc-pvtz", ene2)
    #
    # print(f"{basis}: \t", func_dict["bparams"],"len: ",len(func_dict["bparams"]))
    # # print(f"{basis_ref}:\t", func_dict["ref_basis"],"len: ",len(func_dict["ref_basis"]))
    # print("Opt params:\t", min_bparams,"len: ",len(min_bparams))
    #
    #
    #
    #


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


#
# def test(molecule):
#     basis = "STO-3G"
#     basis_ref = "cc-pvtz"
#     sys1,sys2,func_dict = system_init(molecule, basis, basis_ref)
#     min_bparams = xitorch.optimize.minimize(fcn, func_dict["bparams"], (func_dict["bpacker"],
#                                                                         func_dict["bparams_ref"],
#                                                                         func_dict["bpacker_ref"],
#                                                                         func_dict["atomstruc_dqc"],
#                                                                         func_dict["atomstruc"],
#                                                                         func_dict["coeffM"],
#                                                                         func_dict["occ_scf"],), step=2e-5,
#                                             method="Adam", maxiter=1000, verbose=True)  # ,method = "Adam", verbose=True
#     return func_dict, min_bparams

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