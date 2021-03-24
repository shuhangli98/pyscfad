import pytest
import pyscf
from pyscfad import gto, scf

@pytest.fixture
def get_mol():
    mol = pyscf.M(
        atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587',
        basis = 'sto3g',
        verbose=0,
    )
    return mol

def test_nuc_grad(get_mol):
    mol0 = get_mol
    x = mol0.atom_coords()
    mol = gto.Mole(mol0, coords=x)
    mf = scf.RHF(mol)
    g = mf.nuc_grad_ad()

    mf0 = pyscf.scf.RHF(mol0)
    mf0.kernel()
    g0 = mf0.Gradients().grad()

    assert abs(g-g0).max() < 1e-6