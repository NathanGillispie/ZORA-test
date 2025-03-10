import os
import numpy as np
from pyscf import dft
import time
import scipy.special
import constants
from pyscf.dft.gen_grid import sg1_prune

class ZORA():
  def __init__(self,wfn):
    self.molecule = wfn.mol
    self.wfn  = wfn
    self.options = wfn.options   

  def _read_basis(self,atoms):
    """Reads the model potential basis `modbas.2c`"""

    basis_file = []
    with open(os.path.abspath(os.path.dirname(__file__)+"/basis/modbas.2c"),'r') as f:
      for line in f.readlines():
        if (len(line) > 1):
          basis_file.append(line)
          
    self.c_a = []

    lower_atoms = list(map(lambda a: a.lower(), atoms))
    for atom in lower_atoms:
      position = [line for line, a in enumerate(basis_file) if a.split()[0] == atom][0]
      nbasis = int(basis_file[position][10:15])
      array = np.loadtxt(basis_file[position+2:position+2+nbasis]).transpose((1,0))
      self.c_a.append((np.array(array[1]),np.sqrt(np.array(array[0]))))

  def compute_Veff(self):
    #get grid
    mol = self.molecule
    atoms = self.options.atoms
    self.wfn.jk.grids.level = self.options.grid_level
    #self.wfn.jk.grids.prune = sg1_prune
    
    atomic_grid = self.wfn.jk.grids.gen_atomic_grids(mol)
    points, self.weights = self.wfn.jk.grids.get_partition(mol, atomic_grid)
    
    #get model potential
    self._read_basis(atoms)

    Vtot = np.zeros((len(points)))
    for Ci, C in enumerate(atoms):
        PA = points - mol.atom_coords()[Ci]
        RPA = np.sqrt(np.sum(PA**2, axis=1))
        c, a = self.c_a[Ci]
        outer = np.outer(a,RPA)
        Vtot += np.einsum("i,i,ip->p",c,a,scipy.special.erf(outer)/outer,optimize=True)
        Vtot -= constants.Z[C.upper()]/RPA
    self.kernel = np.asarray(Vtot)

    self.points  = np.asarray(points)
    self.weights = np.asarray(self.weights)

    print("   ZORA grid computed successfuly!",flush=True)

  
  def get_zora_correction(self):
    print("    Computing ZORA integrals.",flush=True)
    tic = time.time()

    self.compute_Veff()
    nbf = self.wfn.nbf
    self.eps_scal_ao = np.zeros((nbf,nbf))
    self.T = np.zeros((4,nbf,nbf))

    npoints = len(self.points)
    print("    Number of grid points: %i"%npoints)
    batch_size = self.options.batch_size
    excess = npoints%batch_size
    nbatches = (npoints-excess)//batch_size
    print("    Number of batches: %i"%(nbatches+1))
    print("    Maximum Batch Size: %i"%batch_size)
    print("    Memory estimation for ZORA build: %8.4f mb"%(batch_size*nbf*6*8/1024./1024.),flush=True)
    for batch in range(nbatches+1):
      low = batch*batch_size
      if batch < nbatches:
        high = low+batch_size
      else:
        high = low+excess

      bpoints  = self.points[low:high]
      bweights = self.weights[low:high]
      bVzora   = self.kernel[low:high]
      ao_val = dft.numint.eval_ao(self.molecule, bpoints, deriv=1)
      kernel = 1./(2.*(137.036**2) - bVzora)
      self.T[0] += np.einsum("xip,xiq,i->pq",ao_val[1:],ao_val[1:],bweights*kernel,optimize=True) * (137.036**2)
      self.eps_scal_ao += np.einsum("xip,xiq,i->pq",ao_val[1:],ao_val[1:],bweights*kernel**2,optimize=True) * (137.036**2)
      kernel = bVzora/(4.*(137.036**2) - 2.*bVzora)
      # x component
      self.T[1] += np.einsum("ip,iq,i->pq",ao_val[2],ao_val[3],bweights*kernel,optimize=True)
      self.T[1] -= np.einsum("ip,iq,i->pq",ao_val[3],ao_val[2],bweights*kernel,optimize=True)

      self.T[2] += np.einsum("ip,iq,i->pq",ao_val[3],ao_val[1],bweights*kernel,optimize=True)
      self.T[2] -= np.einsum("ip,iq,i->pq",ao_val[1],ao_val[3],bweights*kernel,optimize=True)

      self.T[3] += np.einsum("ip,iq,i->pq",ao_val[1],ao_val[2],bweights*kernel,optimize=True)
      self.T[3] -= np.einsum("ip,iq,i->pq",ao_val[2],ao_val[1],bweights*kernel,optimize=True)
    toc = time.time()

    print("    ZORA integrals computed in %5.2f seconds \n"%(toc-tic),flush=True)

    self.H_so = np.zeros((2*nbf,2*nbf),dtype=complex)
    Kx = 1j * self.T[1]
    Ky = 1j * self.T[2]
    Kz = 1j * self.T[3]

    self.H_so[:nbf,:nbf] =   Kz
    self.H_so[nbf:,nbf:] =  -Kz
    self.H_so[:nbf,nbf:] =  (Kx - 1j*Ky)
    self.H_so[nbf:,:nbf] =  (Kx + 1j*Ky)




