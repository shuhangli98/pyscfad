import sys
from typing import Optional, Union, Any
import numpy

from pyscf import __config__
from pyscf import gto
from pyscf.lib import logger, param
from pyscf.gto.mole import NPRIM_OF, NCTR_OF, PTR_COEFF, PTR_ENV_START

from pyscfad import lib
from pyscfad.lib import numpy as jnp
from pyscfad.lib import ops
from pyscfad.gto import moleintor

def energy_nuc(mol, charges=None, coords=None):
    if charges is None: charges = mol.atom_charges()
    if len(charges) <= 1:
        return 0
    rr = inter_distance(mol, coords)
    rr = ops.index_update(rr, jnp.diag_indices_from(rr), 1.e200)
    e = jnp.einsum('i,ij,j->', charges, 1./rr, charges) * .5
    return e

def inter_distance(mol, coords=None):
    if coords is None: coords = mol.coords
    if coords is None: coords = mol.atom_coords()
    rr = jnp.linalg.norm(coords.reshape(-1,1,3) - coords, axis=2)
    rr = ops.index_update(rr, jnp.diag_indices_from(rr), 0.)
    return rr

def setup_ctr_coeff(mol):
    tmp = []
    cs = jnp.empty([0], dtype=float)
    _env_of = numpy.empty([0], dtype=numpy.int32)
    offset = 0
    cs_of = []
    for i in range(len(mol._bas)):
        nprim = mol._bas[i,NPRIM_OF]
        nctr = mol._bas[i,NCTR_OF]
        ptr_coeff = mol._bas[i,PTR_COEFF]
        if ptr_coeff not in tmp:
            tmp.append(ptr_coeff)
            cs = jnp.append(cs, mol._env[ptr_coeff : ptr_coeff+nprim*nctr])
            _env_of = numpy.append(_env_of, numpy.arange(ptr_coeff,ptr_coeff+nprim*nctr))
            cs_of.append(offset)
            offset += nprim*nctr
    tmp = numpy.asarray(tmp, dtype=numpy.int32)
    cs_of = numpy.asarray(cs_of, dtype=numpy.int32)
    ptr_coeff = mol._bas[:,PTR_COEFF]
    idx = []
    for ptr in ptr_coeff:
        idx.append(numpy.where(ptr == tmp)[0])
    idx = numpy.asarray(idx).flatten()
    cs_of = cs_of[idx]
    return cs, cs_of, _env_of

@lib.dataclass
class Mole(gto.Mole):
    # traced attributes
    # NOTE jax requires that at least one variable needs to be traced for AD
    coords: jnp.array = lib.field(pytree_node=True, default=jnp.zeros([1,3], dtype=float))
    exponents: Optional[jnp.array] = lib.field(pytree_node=True, default=None)
    ctr_coeff: Optional[jnp.array] = lib.field(pytree_node=True, default=None)

    # attributes of the base class
    verbose: int = getattr(__config__, 'VERBOSE', logger.NOTE)
    unit: str = getattr(__config__, 'UNIT', 'angstrom')
    incore_anyway: bool = getattr(__config__, 'INCORE_ANYWAY', False)
    cart: bool = getattr(__config__, 'gto_mole_Mole_cart', False)

    # attributes of the base class object
    output: Optional[str] = None
    max_memory: int = param.MAX_MEMORY
    charge: int = 0
    spin: int = 0
    symmetry: bool = False
    symmetry_subgroup: Optional[str] = None
    cart: bool = False
    atom: Union[list,str] = lib.field(default_factory = list)
    basis: Union[dict,str] = 'sto-3g'
    nucmod: Union[dict,str] = lib.field(default_factory = dict)
    ecp: Union[dict,str] = lib.field(default_factory = dict)
    nucprop: dict = lib.field(default_factory = dict)

    # private attributes
    _atm: numpy.ndarray = numpy.zeros((0,6), dtype=numpy.int32)
    _bas: numpy.ndarray = numpy.zeros((0,8), dtype=numpy.int32)
    _env: numpy.ndarray = numpy.zeros(PTR_ENV_START)
    _ecpbas: numpy.ndarray = numpy.zeros((0,8), dtype=numpy.int32)

    stdout: Any = sys.stdout
    groupname: str = 'C1'
    topgroup: str = 'C1'
    symm_orb: Optional[list] = None
    irrep_id: Optional[list] = None
    irrep_name: Optional[list] = None
    _symm_orig: Optional[numpy.ndarray] = None
    _symm_axes: Optional[numpy.ndarray] = None
    _nelectron: Optional[int] = None
    _nao: Optional[int] = None
    _enuc: Optional[float] = None
    _atom: list = lib.field(default_factory = list)
    _basis: dict = lib.field(default_factory = dict)
    _ecp: dict = lib.field(default_factory = dict)
    _built: bool = False
    _pseudo: dict = lib.field(default_factory = dict)

    def __post_init__(self):
        self._keys = set(self.__dict__.keys())

    def build(self, *args, **kwargs):
        gto.Mole.build(self, *args, **kwargs)
       
        self.coords = jnp.asarray(self.atom_coords())
        self.ctr_coeff, _, _ = setup_ctr_coeff(self)

    energy_nuc = energy_nuc

    def intor(self, intor, comp=None, hermi=0, aosym='s1', out=None,
              shls_slice=None):
        has_grad = ["int1e_ovlp", 
                    "int1e_kin",
                    "int1e_nuc",
                    "ECPscalar",
                    "int2e",]
        if not intor in has_grad:
            return gto.Mole.intor(self, intor, comp=comp, hermi=hermi, aosym=aosym, out=out, shls_slice=shls_slice)
        else:
            return moleintor.getints(self, intor)
