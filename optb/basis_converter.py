"""
function to convert a given dqc Basis format into a scf basis format.
"""

import torch
from dqc.utils.misc import gaussian_int


def bconv(bparams, bpacker ):
    """
    creates a pyscf type basis dict out of a given dqc type basis input
    :param bparams: torch.Tensor with coeff of basis set
    :param bpacker: xitorch._core.packer.Packer object to create the CGTOBasis out of the bparams
    :return: dict where each element gets his own basis arr
    """
    basis = bpacker.construct_from_tensor(bparams)
    def wfnormalize_(CGTOB):
        """
        Normalize coefficients from the unnormalized state from dqc
        :param CGTOB:
        :return: basis dict
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
        return CGTOB

    bdict = {}

    for el in basis:
        arr = []
        for CGTOB in basis[el]:
            CGTOB = wfnormalize_(CGTOB)
            innerarr = [CGTOB.angmom]
            for al,co in zip(CGTOB.alphas, CGTOB.coeffs):
                innerarr.append([float(al), float(co)])
            arr.append(innerarr)
        bdict[el] = arr
    return bdict
