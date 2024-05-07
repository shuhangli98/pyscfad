import jax
from jax import numpy as np
from pyscf.lib import split_reshape
from pyscf.lib import logger, ops
from pyscfad import util
from pyscfad import lib
from pyscfad.lib import ops, logger
from pyscfad import ao2mo


def kernel(dsrg, s, mo_energy=None, mo_coeff=None, eris=None):
    if eris is None:
        eris = dsrg.ao2mo(mo_coeff)

    if mo_energy is None:
        mo_energy = eris.mo_energy

    nocc = dsrg.nocc
    nvir = dsrg.nmo - nocc
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
    edsrg = 0
    for i in range(nocc):
        gi = np.asarray(eris.ovov[i*nvir:(i+1)*nvir])
        gi = gi.reshape(nvir, nocc, nvir).transpose(1, 0, 2)
        delta = eia[:, :, None] + eia[i][None, None, :]
        t2i = (1 - np.exp(-2*s*(delta**2))) * gi.conj()/delta
        edsrg += np.einsum('jab,jab', t2i, gi) * 2
        edsrg -= np.einsum('jab,jba', t2i, gi)

    return edsrg.real


class DSRG():
    def __init__(self, mf, frozen=None, s=0.5):
        self.flow_param = s
        self.mol = mf.mol
        self._scf = mf
        self.verbose = mf.mol.verbose
        self.stdout = mf.mol.stdout
        self.max_memory = mf.max_memory
        self.frozen = frozen
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self.mo_energy = mf.mo_energy
        self.nocc = np.count_nonzero(self.mo_occ > 0)
        self.nmo = len(self.mo_energy)
        self.e_hf = 0.0
        self.corr = 0.0
        self.e_tot = 0.0

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        logger.note(self, 'E(%s) = %.15g  E_corr = %.15g',
                    self.__class__.__name__, self.e_tot, self.e_corr)
        return self

    def ao2mo(self, mo_coeff=None):
        eris = _ChemistsERIs()
        eris._common_init_(self, mo_coeff)
        mo_coeff = eris.mo_coeff

        nocc = self.nocc
        co = np.asarray(mo_coeff[:, :nocc])
        cv = np.asarray(mo_coeff[:, nocc:])
        eris.ovov = ao2mo.general(self._scf._eri, (co, cv, co, cv))
        return eris

    def kernel(self, s=0.5, mo_energy=None, mo_coeff=None, eris=None):
        self.e_hf = self.get_e_hf()
        if eris is None:
            eris = self.ao2mo(mo_coeff)

        if mo_energy is None:
            mo_energy = eris.mo_energy

        if self._scf.converged:
            nocc = self.nocc
            nvir = self.nmo - nocc
            eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
            edsrg = 0
            for i in range(nocc):
                gi = np.asarray(eris.ovov[i*nvir:(i+1)*nvir])
                gi = gi.reshape(nvir, nocc, nvir).transpose(1, 0, 2)
                delta = eia[:, :, None] + eia[i][None, None, :]
                t2i = (1 - np.exp(-2*s*(delta**2))) * gi.conj()/delta
                edsrg += np.einsum('jab,jab', t2i, gi) * 2
                edsrg -= np.einsum('jab,jba', t2i, gi)

        self.e_corr = edsrg.real
        self._finalize()
        self.e_tot = self.e_corr + self.e_hf

        return self.e_corr

    def get_e_hf(self):
        # Get HF energy.
        dm = self._scf.make_rdm1(self.mo_coeff, self.mo_occ)
        vhf = self._scf.get_veff(self._scf.mol, dm)
        return self._scf.energy_tot(dm=dm, vhf=vhf)


class _ChemistsERIs():
    def _common_init_(self, dsrg, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = dsrg.mo_coeff
        if mo_coeff is None:
            raise RuntimeError('mo_coeff, mo_energy are not initialized.')

        self.mo_coeff = mo_coeff
        self.mol = dsrg.mol
        self.mo_energy = dsrg.mo_energy
        self.fock = np.diag(self.mo_energy)

        return self


if __name__ == '__main__':
    import jax
    import pyscf
    from pyscfad import gto, scf, mp
    from pyscfad import config

    # # implicit differentiation of SCF iterations
    # config.update('pyscfad_scf_implicit_diff', True)

    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74'
    mol.basis = '631g'
    mol.verbose = 4
    mol.build()

    def run_dsrg(mol, dm0=None):
        mf = scf.RHF(mol)
        mf.kernel(dm0)
        dsrg = DSRG(mf)
        dsrg.kernel()
        return dsrg.e_tot
    jac = jax.grad(run_dsrg)(mol)
    print(f'Nuclaer gradient:\n{jac.coords}')
    print(f'Gradient wrt basis exponents:\n{jac.exp}')
    print(f'Gradient wrt basis contraction coefficients:\n{jac.ctr_coeff}')
