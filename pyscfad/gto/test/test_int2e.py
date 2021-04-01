import pytest
import numpy as np
import pyscf
import jax
from pyscfad.lib import numpy as jnp
from pyscfad import gto

@pytest.fixture
def get_mol0():
    mol = pyscf.M(
        atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587',
        basis = 'sto3g',
        verbose=0,
    )
    return mol

@pytest.fixture
def get_mol():
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = 'sto3g'
    mol.verbose=0
    mol.build()
    return mol

def int2e_grad_analyt(mol):
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    nao = mol.nao
    g = np.zeros((nao,nao,nao,nao,mol.natm,3))
    h1 = -mol.intor("int2e_ip1", comp=3)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        g[p0:p1,:,:,:,k] += h1[:,p0:p1].transpose(1,2,3,4,0)
        g[:,p0:p1,:,:,k] += h1[:,p0:p1].transpose(2,1,3,4,0)
        g[:,:,p0:p1,:,k] += h1[:,p0:p1].transpose(3,4,1,2,0)
        g[:,:,:,p0:p1,k] += h1[:,p0:p1].transpose(3,4,2,1,0)
    return g

def cs_grad_fd(mol, intor):
    disp = 1e-5 / 2.
    grad_fd = []
    cs, cs_of, _env_of = gto.mole.setup_ctr_coeff(mol)
    for i in range(len(cs)):
        ptr_ctr = _env_of[i]
        mol._env[ptr_ctr] += disp
        sp = mol.intor(intor)
        mol._env[ptr_ctr] -= disp *2.
        sm = mol.intor(intor)
        g = (sp-sm) / (disp*2.)
        grad_fd.append(g)
        mol._env[ptr_ctr] += disp
    grad_fd = np.asarray(grad_fd).transpose(1,2,3,4,0)
    return grad_fd

def exp_grad_fd(mol, intor):
    disp = 1e-5/2.
    grad_fd = []
    es, es_of, _env_of = gto.mole.setup_exp(mol)
    for i in range(len(es)):
        ptr_exp = _env_of[i]
        mol._env[ptr_exp] += disp
        sp = mol.intor(intor)
        mol._env[ptr_exp] -= disp *2.
        sm = mol.intor(intor)

        s = (sp-sm) / (disp*2.)
        grad_fd.append(s)
        mol._env[ptr_exp] += disp
    grad_fd = np.asarray(grad_fd).transpose(1,2,3,4,0)
    return grad_fd


def func(mol, intor):
    return mol.intor(intor)

def func1(mol, intor):
    return jnp.linalg.norm(mol.intor(intor))

def test_int2e(get_mol0, get_mol):
    mol0 = get_mol0
    eri0 = mol0.intor("int2e")
    mol1 = get_mol
    eri = mol1.intor("int2e")
    assert abs(eri-eri0).max() < 1e-10

    tmp_nuc = int2e_grad_analyt(mol0)
    tmp_cs = cs_grad_fd(mol0, "int2e")
    tmp_exp = exp_grad_fd(mol0, "int2e")

    g0_nuc = tmp_nuc
    g0_cs = tmp_cs
    g0_exp = tmp_exp
    jac = jax.jacfwd(func)(mol1, "int2e")
    g_nuc = jac.coords
    g_cs = jac.ctr_coeff
    g_exp = jac.exp
    assert abs(g_nuc-g0_nuc).max() < 1e-10
    assert abs(g_cs-g0_cs).max() < 1e-9
    assert abs(g_exp-g0_exp).max() < 1e-8

    g0_nuc = np.einsum("ijkl,ijklnx->nx", eri0, tmp_nuc) / np.linalg.norm(eri0)
    g0_cs = np.einsum("ijkl,ijklx->x", eri0, tmp_cs) / np.linalg.norm(eri0)
    g0_exp = np.einsum("ijkl,ijklx->x", eri0, tmp_exp) / np.linalg.norm(eri0)
    jac = jax.jacfwd(func1)(mol1, "int2e")
    g_nuc = jac.coords
    g_cs = jac.ctr_coeff
    g_exp = jac.exp
    assert abs(g_nuc-g0_nuc).max() < 1e-10
    assert abs(g_cs-g0_cs).max() < 1e-9
    assert abs(g_exp-g0_exp).max() < 1e-8
