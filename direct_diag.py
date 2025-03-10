import numpy as np
import scipy as sp
import time
import xc_potential
from pyscf.scf import hf #jk
from pyscf.scf import jk #jk
from pyscf import dft, ao2mo
import sys
import tracemalloc as tmal

class RDIRECT_DIAG():

    def __init__(self, wfn):
        #tic = time.time()

        #begin Davidson
        self.wfn = wfn
        self.options = self.wfn.options
        self.mol = wfn.ints_factory
        self.ref = wfn.reference
        self.nbf   = int(self.wfn.nbf)
        self.no_a  = int(self.wfn.nel[0])
        self.nv_a  = int(self.nbf - self.no_a)
        self.nov   = int(self.no_a * self.nv_a)
        self.no_a_full  = int(self.wfn.nel[0])
        self.nv_a_full  = int(self.nbf - self.wfn.nel[0])
        self.nov_a_full   = int(self.no_a_full * self.nv_a_full)
        self.reduced_virtual = False

        self.S = self.wfn.S[0]
        self.nroots = self.nov

        self.Cmo = self.wfn.C[0]

        self.Co = self.Cmo[:,:self.no_a]
        self.Cv = self.Cmo[:,self.no_a:]

        eps_a = self.wfn.eps[0]
        self.F_occ  = np.diag(eps_a[:self.no_a])
        self.F_virt = np.diag(eps_a[self.no_a:])

    def compute(self):
        #print(self.cvs)
        #return
        print("\n")
        print("    Configuration Interaction Singles")
        print("    ---------------------------------")
        print("\n")

        self.tic = time.time()

        if (self.cvs is True):
          olow  = self.wfn.options.occupied[0] - 1
          ohigh = self.wfn.options.occupied[1] - 1
          print("    Restricting occupied orbitals in the window [%i,%i]"%(olow+1,ohigh+1),flush=True)
          self.no_a = int(ohigh-olow+1)
          self.F_occ = self.F_occ[olow:ohigh+1,olow:ohigh+1]
          self.Co      = self.Co[:,olow:ohigh+1]
          #if (self.reduced_virtual is True):
          #  vlow  = self.wfn.options.virtual[0] - 1 - self.no_a_full
          #  vhigh = self.wfn.options.virtual[1] - 1 - self.no_a_full
          #  print("    Restricting virtual orbitals in the window [%i,%i]"%(vlow+1+self.no_a_full,vhigh+1+self.no_a_full),flush=True)
          #  self.nv_a = int(vhigh-vlow+1)
          #  self.F_virt = self.F_virt[vlow:vhigh+1,vlow:vhigh+1]                                                                    self.Cv      = self.Cv[:,vlow:vhigh+1]
          #else:
          vlow  = 0
          vhigh = self.nv_a_full

          self.nov = self.no_a * self.nv_a
          self.nroots = self.nov

          #if self.nov < self.nroots:
          #  print("    Resizing number of roots to %i"%self.nov)
          #  self.nroots = self.nov
          #  self.maxdim = self.nov

        print("    Total number of configurations: %i"%(self.nov))
        mem_max = ((self.nov)**2)*8*8/1024/1024
        print("    Maximum memory required: %i mb"%mem_max)
        if mem_max > 32000:
          print("    This computation requires too much memory. Try the Davidson procedure.")
          exit("    Exiting computation.")

        A  = np.einsum("ab,ij->aibj",self.F_virt,np.eye(self.no_a),optimize=True)
        A -= np.einsum("ab,ij->aibj",np.eye(self.nv_a),self.F_occ,optimize=True)

        tic = time.time()

        nel_a = self.no_a
        nv_a = self.nv_a
        Co  = self.Co
        Cv  = self.Cv

        if self.wfn.triplet is False:
          jktic = time.time()
          G_aibj  = ao2mo.general(self.mol,[Cv,Co,Cv,Co],compact=False)
          G_aibj = G_aibj.reshape(nv_a,nel_a,nv_a,nel_a)
          jktoc = time.time()
          print("    (ai|bj) contributionn done in %5.2f seconds."%(jktoc-jktic),flush=True)

        jktic = time.time()
        G_ijba  = ao2mo.general(self.mol,[Co,Co,Cv,Cv],compact=False)
        G_abji = G_ijba.reshape(nel_a,nel_a,nv_a,nv_a).swapaxes(0,3).swapaxes(1,2)
        jktoc = time.time()
        print("    (ab|ji) contributionn done in %5.2f seconds."%(jktoc-jktic),flush=True)

        toc = time.time()
        print("    AO to MO transformation time: %5.2f seconds"%(toc -tic),flush=True)

        if self.wfn.triplet is False:
          A += 2.*G_aibj
          A -= G_abji.swapaxes(1,3).swapaxes(2,3)*self.options.xcalpha
  
        else:
          A -= G_abji.swapaxes(1,3).swapaxes(2,3)*self.options.xcalpha

        if self.options.nofxc is True:
          print("    WARNING!!! Excluding fxc.")
        else:
          tic = time.time()
          xc_functional = xc_potential.XC(self.wfn)
          if self.wfn.triplet is False:
            F_aibj = xc_functional.computeF([Co,Co],[Cv,Cv],spin=0)
            A += 2.*F_aibj[0]
          else:
            F_aibj = xc_functional.computeF([Co,Co],[Cv,Cv],spin=1)
            A += F_aibj[0] - F_aibj[1]
          toc = time.time()
          print("    Fxc evaluated in: %5.2f seconds"%(toc -tic),flush=True)

        A  = A.reshape(nel_a*nv_a,nel_a*nv_a)
        evals, X = sp.linalg.eigh(A)

        total_toc = time.time()
        diagon_time = total_toc - tic                                                    

        print("    Direct diagonalization performed in %8.5f seconds"%diagon_time,flush=True)

        if self.cvs is True:
          #expand original CI vectors to avoid problem with dimensions later
          olow  = self.wfn.options.occupied[0] - 1
          ohigh = self.wfn.options.occupied[1] - 1
          Xfull = np.zeros((self.nroots,self.nv_a_full,self.no_a_full))
          print("    Expanding occupied space from %i to %i"%(self.no_a,self.no_a_full))
          X = (X.T).reshape(self.nroots,self.nv_a,self.no_a)
          if self.reduced_virtual is True:
            print("    Expanding virtual space from %i to %i"%(self.nv_a,self.nv_a_full))
            vlow  = self.wfn.options.virtual[0] - 1 - self.no_a_full
            vhigh = self.wfn.options.virtual[1] - 1 - self.no_a_full
            Xfull[:,vlow:vhigh+1,olow:ohigh+1] = X
          else:
            Xfull[:,:,olow:ohigh+1] = X
          Xfull = Xfull.reshape(self.nroots,self.nov_a_full)
          self.Cmo = self.wfn.C[0]
          self.no_a = self.no_a_full
          self.nv_a = self.nv_a_full
          self.nov  = self.nov_a_full
          return evals[:self.nroots], Xfull.T
        else:
          return evals[:self.nroots], X

class UDIRECT_DIAG():

    def __init__(self, wfn):

        tic = time.time()
        self.wfn = wfn
        self.options = self.wfn.options
        self.mol = wfn.ints_factory
        self.ref = wfn.reference
        self.nbf   = int(self.wfn.nbf)
        self.no_a  = int(self.wfn.nel[0])
        self.no_a_full  = int(self.wfn.nel[0])
        self.nv_a  = int(self.nbf - self.no_a)
        self.nov_a = int(self.no_a * self.nv_a)
        self.nov_a_full = int(self.no_a * self.nv_a)
        self.no_b  = int(self.wfn.nel[1])
        self.no_b_full  = int(self.wfn.nel[1])
        self.nv_b  = int(self.nbf - self.no_b)
        self.nov_b = int(self.no_b * self.nv_b)
        self.nov_b_full = int(self.no_b * self.nv_b)

        self.S = self.wfn.S
        self.nroots = self.nov_a + self.nov_b

        self.Cmo = self.wfn.C.copy()

        self.Co_a = 1.*self.Cmo[0][:,:self.no_a]
        self.Cv_a = 1.*self.Cmo[0][:,self.no_a:]
        self.Co_b = 1.*self.Cmo[1][:,:self.no_b]
        self.Cv_b = 1.*self.Cmo[1][:,self.no_b:]

        eps_a = self.wfn.eps[0]
        eps_b = self.wfn.eps[1]

        self.F_occ_a  = np.diag(eps_a[:self.no_a])
        self.F_occ_b  = np.diag(eps_b[:self.no_b])
        self.F_virt_a = np.diag(eps_a[self.no_a:])
        self.F_virt_b = np.diag(eps_b[self.no_b:])

        toc = time.time() 
        print("Initialize Diagonalization",toc-tic) 


    def compute(self):
        #print(self.cvs)
        #return
        print("\n")
        print("    Configuration Interaction Singles") 
        print("    ---------------------------------") 
        print("\n")

        tic = time.time()
  
        #atomic L edge 
        #atomic K edge
        #low = 0
        #high = 1

        if (self.cvs is True):
          low  = self.wfn.options.occupied[0] - 1
          high = self.wfn.options.occupied[1] - 1
          print("    Restricting occupied orbitals in the window [%i,%i]"%(low+1,high+1),flush=True)
          self.no_a = int(high-low+1)
          self.no_b = int(high-low+1)
          self.F_occ_a = self.F_occ_a[low:high+1,low:high+1]
          self.F_occ_b = self.F_occ_b[low:high+1,low:high+1]
          self.Co_a      = self.Co_a[:,low:high+1]
          self.Co_b      = self.Co_b[:,low:high+1]
          self.nov_a = self.no_a * self.nv_a
          self.nov_b = self.no_b * self.nv_b
          self.nroots = self.nov_a + self.nov_b

          self.Cmo_a = np.zeros((self.nbf,self.nv_a+self.no_a))
          self.Cmo_b = np.zeros((self.nbf,self.nv_b+self.no_b))
          self.Cmo_a[:,:self.no_a] = self.Co_a 
          self.Cmo_a[:,self.no_a:] = self.Cv_a 
          self.Cmo_b[:,:self.no_b] = self.Co_b 
          self.Cmo_b[:,self.no_b:] = self.Cv_b 

        print("    Total number of configurations: %i"%(self.nov_a+self.nov_b))
        mem_max = ((self.nov_a+self.nov_b)**2)*8*8/1024/1024
        print("    Maximum memory required: %i mb"%mem_max)
        if mem_max > 32000:
          print("    This computation requires too much memory. Try the Davidson procedure.")
          exit("    Exiting computation.")

        A_aa  = np.einsum("ab,ij->aibj",self.F_virt_a,np.eye(self.no_a),optimize=True)
        A_aa -= np.einsum("ab,ij->aibj",np.eye(self.nv_a),self.F_occ_a,optimize=True)
        A_bb  = np.einsum("ab,ij->aibj",self.F_virt_b,np.eye(self.no_b),optimize=True)
        A_bb -= np.einsum("ab,ij->aibj",np.eye(self.nv_b),self.F_occ_b,optimize=True)

        A_ab = np.zeros((self.nv_a,self.no_a,self.nv_b,self.no_b))
        A_ba = np.zeros((self.nv_b,self.no_b,self.nv_a,self.no_a))
          
        self.ints = self.wfn.ints_factory

        # J-type integrals
        if (self.options.akonly is True) or (self.options.fonly is True):
          print("    Excluding J-type integrals",flush=True)
        elif(self.options.atomic_J is True):
          print("    Enabling Atomic J build for AOS in the list:",self.options.atomic_J_list,flush=True)

          Co_a_red = np.zeros(self.Co_a.shape)
          Co_b_red = np.zeros(self.Co_b.shape)
          Cv_a_red = np.zeros(self.Cv_a.shape)
          Cv_b_red = np.zeros(self.Cv_b.shape)

          for mu in self.options.atomic_J_list:
            Co_a_red[mu] = self.Co_a[mu]    
            Co_b_red[mu] = self.Co_b[mu]    
            Cv_a_red[mu] = self.Cv_a[mu]    
            Cv_b_red[mu] = self.Cv_b[mu]    

          jktic = time.time()
          G_aibj  = ao2mo.general(self.ints,[Cv_a_red,Co_a_red,Cv_a_red,Co_a_red],compact=False)
          jktoc = time.time()
          print("    (ai|bj) contributionn done in %8.5f seconds."%(jktoc-jktic),flush=True)

          jktic = time.time()
          G_aiBJ  = ao2mo.general(self.ints,[Cv_a_red,Co_a_red,Cv_b_red,Co_b_red],compact=False)
          jktoc = time.time()
          print("    (ai|BJ) contributionn done in %8.5f seconds."%(jktoc-jktic),flush=True)

          jktic = time.time()
          G_AIbj  = ao2mo.general(self.ints,[Cv_b_red,Co_b_red,Cv_a_red,Co_a_red],compact=False)
          jktoc = time.time()
          print("    (AI|bj) contributionn done in %8.5f seconds."%(jktoc-jktic),flush=True)

          jktic = time.time()
          G_AIBJ  = ao2mo.general(self.ints,[Cv_b_red,Co_b_red,Cv_b_red,Co_b_red],compact=False)
          jktoc = time.time()
          print("    (AI|BJ) contributionn done in %8.5f seconds."%(jktoc-jktic),flush=True)

          self.G_aibj = G_aibj.reshape(self.nv_a,self.no_a,self.nv_a,self.no_a)
          self.G_aiBJ = G_aiBJ.reshape(self.nv_a,self.no_a,self.nv_b,self.no_b)
          self.G_AIbj = G_AIbj.reshape(self.nv_b,self.no_b,self.nv_a,self.no_a)
          self.G_AIBJ = G_AIBJ.reshape(self.nv_b,self.no_b,self.nv_b,self.no_b)

          A_aa += self.G_aibj
          A_bb += self.G_AIBJ
          A_ab += self.G_aiBJ
          A_ba += self.G_AIbj

        elif self.options.perturb_J:
          print("    Computing J to do perturbative inclusion",flush=True)
 
          jktic = time.time()
          G_aibj  = ao2mo.general(self.ints,[self.Cv_a,self.Co_a,self.Cv_a,self.Co_a],compact=False)
          jktoc = time.time()
          print("    (ai|bj) contributionn done in %8.5f seconds."%(jktoc-jktic),flush=True)

          jktic = time.time()
          G_aiBJ  = ao2mo.general(self.ints,[self.Cv_a,self.Co_a,self.Cv_b,self.Co_b],compact=False)
          jktoc = time.time()
          print("    (ai|BJ) contributionn done in %8.5f seconds."%(jktoc-jktic),flush=True)

          jktic = time.time()
          G_AIbj  = ao2mo.general(self.ints,[self.Cv_b,self.Co_b,self.Cv_a,self.Co_a],compact=False)
          jktoc = time.time()
          print("    (AI|bj) contributionn done in %8.5f seconds."%(jktoc-jktic),flush=True)

          jktic = time.time()
          G_AIBJ  = ao2mo.general(self.ints,[self.Cv_b,self.Co_b,self.Cv_b,self.Co_b],compact=False)
          jktoc = time.time()
          print("    (AI|BJ) contributionn done in %8.5f seconds."%(jktoc-jktic),flush=True)

          self.G_aibj = G_aibj.reshape(self.nv_a,self.no_a,self.nv_a,self.no_a)
          self.G_aiBJ = G_aiBJ.reshape(self.nv_a,self.no_a,self.nv_b,self.no_b)
          self.G_AIbj = G_AIbj.reshape(self.nv_b,self.no_b,self.nv_a,self.no_a)
          self.G_AIBJ = G_AIBJ.reshape(self.nv_b,self.no_b,self.nv_b,self.no_b)

        else:
          jktic = time.time()
          G_aibj  = ao2mo.general(self.ints,[self.Cv_a,self.Co_a,self.Cv_a,self.Co_a],compact=False)
          jktoc = time.time()
          print("    (ai|bj) contributionn done in %8.5f seconds."%(jktoc-jktic),flush=True)

          jktic = time.time()
          G_aiBJ  = ao2mo.general(self.ints,[self.Cv_a,self.Co_a,self.Cv_b,self.Co_b],compact=False)
          jktoc = time.time()
          print("    (ai|BJ) contributionn done in %8.5f seconds."%(jktoc-jktic),flush=True)

          jktic = time.time()
          G_AIbj  = ao2mo.general(self.ints,[self.Cv_b,self.Co_b,self.Cv_a,self.Co_a],compact=False)
          jktoc = time.time()
          print("    (AI|bj) contributionn done in %8.5f seconds."%(jktoc-jktic),flush=True)

          jktic = time.time()
          G_AIBJ  = ao2mo.general(self.ints,[self.Cv_b,self.Co_b,self.Cv_b,self.Co_b],compact=False)
          jktoc = time.time()
          print("    (AI|BJ) contributionn done in %8.5f seconds."%(jktoc-jktic),flush=True)
        
          self.G_aibj = G_aibj.reshape(self.nv_a,self.no_a,self.nv_a,self.no_a)
          self.G_aiBJ = G_aiBJ.reshape(self.nv_a,self.no_a,self.nv_b,self.no_b)
          self.G_AIbj = G_AIbj.reshape(self.nv_b,self.no_b,self.nv_a,self.no_a)
          self.G_AIBJ = G_AIBJ.reshape(self.nv_b,self.no_b,self.nv_b,self.no_b)
        
          A_aa += self.G_aibj
          A_bb += self.G_AIBJ
          A_ab += self.G_aiBJ
          A_ba += self.G_AIbj


        # K-type integrals
        if self.options.fonly is True:
          print("    Excluding K-type integrals",flush=True)
        else:
          jktic = time.time()
          #DRN: For some reason, computing G_abji takes more than 10x longer. Possibly contiguous memory issues?!
          #G_abji  = ao2mo.general(self.ints,[self.Cv_a,self.Cv_a,self.Co_a,self.Co_a],compact=False)
          G_ijba  = ao2mo.general(self.ints,[self.Co_a,self.Co_a,self.Cv_a,self.Cv_a],compact=False)
          jktoc = time.time()
          print("    (ab|ji) contributionn done in %8.5f seconds."%(jktoc-jktic),flush=True)

          jktic = time.time()
          #G_ABJI  = ao2mo.general(self.ints,[self.Cv_b,self.Cv_b,self.Co_b,self.Co_b],compact=False)
          G_IJBA = ao2mo.general(self.ints,[self.Co_b,self.Co_b,self.Cv_b,self.Cv_b],compact=False)
          jktoc = time.time()
          print("    (AB|JI) contributionn done in %8.5f seconds."%(jktoc-jktic),flush=True)

          #self.G_abji = G_abji.reshape(self.nv_a,self.nv_a,self.no_a,self.no_a)
          self.G_abji = G_ijba.reshape(self.no_a,self.no_a,self.nv_a,self.nv_a).swapaxes(0,3).swapaxes(1,2)
          #self.G_ABJI = G_ABJI.reshape(self.nv_b,self.nv_b,self.no_b,self.no_b)
          self.G_ABJI = G_IJBA.reshape(self.no_b,self.no_b,self.nv_b,self.nv_b).swapaxes(0,3).swapaxes(1,2)

          A_aa -= self.G_abji.swapaxes(1,3).swapaxes(2,3)*self.options.xcalpha 
          A_bb -= self.G_ABJI.swapaxes(1,3).swapaxes(2,3)*self.options.xcalpha



        if (self.options.nofxc is True) or (self.options.akonly is True) or (self.options.fonly is True):
          print("    Skipping fxc build!")
        else:
          if self.options.xctype != "HF":
            xctic = time.time()
            xc_functional = xc_potential.XC(self.wfn)
            F_aibj, F_aiBJ, F_AIbj, F_AIBJ = xc_functional.computeF([self.Co_a,self.Co_b],[self.Cv_a,self.Cv_b],spin=1)
 
            A_aa += F_aibj #* (1.-self.options.xcalpha)
            A_ab += F_aiBJ #* (1.-self.options.xcalpha)
            A_ba += F_AIbj #* (1.-self.options.xcalpha)
            A_bb += F_AIBJ #* (1.-self.options.xcalpha)
             
            xctoc = time.time()
            print("    fxc evaluated in %f seconds"%(xctoc-xctic),flush=True)


        A_aa  = A_aa.reshape(self.no_a*self.nv_a,self.no_a*self.nv_a) 
        A_ab  = A_ab.reshape(self.no_a*self.nv_a,self.no_b*self.nv_b) 
        A_ba  = A_ba.reshape(self.no_b*self.nv_b,self.no_a*self.nv_a) 
        A_bb  = A_bb.reshape(self.no_b*self.nv_b,self.no_b*self.nv_b) 

        #self.G_aibj = self.G_aibj.reshape(self.no_a*self.nv_a,self.no_a*self.nv_a) 
        #self.G_AIBJ = self.G_AIBJ.reshape(self.no_b*self.nv_b,self.no_b*self.nv_b)
        #self.G_aiBJ = self.G_aiBJ.reshape(self.no_a*self.nv_a,self.no_b*self.nv_b) 
        #self.G_AIbj = self.G_AIbj.reshape(self.no_b*self.nv_b,self.no_a*self.nv_a)

        #A_aa += np.einsum("pq,pq->pq",np.eye(self.no_a*self.nv_a),self.G_aibj,optimize=True) 
        #A_bb += np.einsum("pq,pq->pq",np.eye(self.no_b*self.nv_b),self.G_AIBJ,optimize=True) 

#        F_aiBJ = F_aiBJ.reshape(self.no_a*self.nv_a,self.no_b*self.nv_b) 
#        F_AIbj = F_AIbj.reshape(self.no_b*self.nv_b,self.no_a*self.nv_a)
#
#        A_aa += np.einsum("pq,p->pq",np.eye(self.no_a*self.nv_a),F_ai,optimize=True) 
#        A_ab += np.einsum("pq,pq->pq",np.eye(self.no_a*self.nv_a),F_aiBJ,optimize=True) 
#        A_bb += np.einsum("pq,p->pq",np.eye(self.no_b*self.nv_b),F_AI,optimize=True) 
#        A_ba += np.einsum("pq,pq->pq",np.eye(self.no_b*self.nv_b),F_AIbj,optimize=True) 


        A = np.block([[A_aa,A_ab],[A_ba,A_bb]])             

        evals, X = sp.linalg.eigh(A)

        if self.options.perturb_J:
          print("   Doing perturbative treatment",flush=True)  
          G_aibj = self.G_aibj.reshape(self.nv_a*self.no_a,self.nv_a*self.no_a) 
          G_aiBJ = self.G_aiBJ.reshape(self.nv_a*self.no_a,self.nv_b*self.no_b) 
          G_AIbj = self.G_AIbj.reshape(self.nv_b*self.no_b,self.nv_a*self.no_a) 
          G_AIBJ = self.G_AIBJ.reshape(self.nv_b*self.no_b,self.nv_b*self.no_b) 

          Xa = X[:,:self.nv_a*self.no_a] 
          Xb = X[:,self.nv_a*self.no_a:] 

          sigma_ai  = np.einsum("pq,Nq->pN",G_aibj,Xa,optimize=True)
          sigma_ai += np.einsum("pq,Nq->pN",G_aiBJ,Xb,optimize=True)

          sigma_AI  = np.einsum("pq,Nq->pN",G_AIbj,Xa,optimize=True)
          sigma_AI += np.einsum("pq,Nq->pN",G_AIBJ,Xb,optimize=True)

          Heff  = np.einsum("Mp,pN->MN",Xa,sigma_ai,optimize=True)
          Heff += np.einsum("Mp,pN->MN",Xb,sigma_AI,optimize=True)

          evals , evecs = sp.linalg.eigh(Heff + evals*np.eye(len(evals)))

          #reconstruct X
          newXa = np.einsum("MN,Np->Mp",evecs,Xa,optimize=True)
          newXb = np.einsum("MN,Np->Mp",evecs,Xb,optimize=True)
          
          X[:,:self.nv_a*self.no_a] = newXa
          X[:,self.nv_a*self.no_a:] = newXb

        total_toc = time.time()
        diagon_time = total_toc - tic
        
        print("    Direct diagonalization performed in %8.5f seconds"%diagon_time)

        if self.cvs is True:
          #expand original CI vectors to avoid problem with dimensions later
          low  = self.wfn.options.occupied[0] - 1
          high = self.wfn.options.occupied[1] - 1
          Xafull = np.zeros((self.nroots,self.nv_a,self.no_a_full),dtype=complex)
          Xbfull = np.zeros((self.nroots,self.nv_b,self.no_b_full),dtype=complex)
          print("    Expanding occupied space from %i to %i"%(self.no_a,self.no_a_full))
          Xa = X[:self.nov_a,:]
          Xb = X[self.nov_a:,:]
          Xa = (Xa.T).reshape(self.nroots,self.nv_a,self.no_a)
          Xb = (Xb.T).reshape(self.nroots,self.nv_b,self.no_b)
          Xafull[:,:,low:high+1] = Xa
          Xbfull[:,:,low:high+1] = Xb
          Xafull = Xafull.reshape(self.nroots,self.nov_a_full)
          Xbfull = Xbfull.reshape(self.nroots,self.nov_b_full)
          Xfull = np.zeros((self.nroots,self.nov_a_full+self.nov_b_full),dtype=complex)
          Xfull[:,:self.nov_a_full] = Xafull
          Xfull[:,self.nov_a_full:] = Xbfull
          self.Cmo = self.wfn.C
          self.no_a = self.no_a_full
          self.no_b = self.no_b_full
          self.nov_a  = self.nov_a_full
          self.nov_b  = self.nov_b_full
          return evals[:self.nroots], Xfull.T
        else:
          return evals[:self.nroots], X

class SFDIRECT_DIAG():

    def __init__(self, wfn):

        tic = time.time()
        self.wfn = wfn
        self.options = self.wfn.options
        self.mol = wfn.ints_factory
        self.ref = wfn.reference
        self.nbf   = int(self.wfn.nbf)
        self.no_a  = int(self.wfn.nel[0])
        self.no_a_full  = int(self.wfn.nel[0])
        self.nv_a  = int(self.nbf - self.no_a)
        self.nov_a = int(self.no_a * self.nv_a)
        self.nov_a_full = int(self.no_a * self.nv_a)
        self.no_b  = int(self.wfn.nel[1])
        self.no_b_full  = int(self.wfn.nel[1])
        self.nv_b  = int(self.nbf - self.no_b)
        self.nov_b = int(self.no_b * self.nv_b)
        self.nov_b_full = int(self.no_b * self.nv_b)

        self.S = self.wfn.S
        self.nroots = self.nov_a + self.nov_b

        self.Cmo = self.wfn.C.copy()

        self.Co_a = 1.*self.Cmo[0][:,:self.no_a]
        self.Cv_a = 1.*self.Cmo[0][:,self.no_a:]
        self.Co_b = 1.*self.Cmo[1][:,:self.no_b]
        self.Cv_b = 1.*self.Cmo[1][:,self.no_b:]

        eps_a = self.wfn.eps[0]
        eps_b = self.wfn.eps[1]

        self.F_occ_a  = np.diag(eps_a[:self.no_a])
        self.F_occ_b  = np.diag(eps_b[:self.no_b])
        self.F_virt_a = np.diag(eps_a[self.no_a:])
        self.F_virt_b = np.diag(eps_b[self.no_b:])

        toc = time.time() 
        print("Initialize Diagonalization",toc-tic) 


    def compute(self):
        #print(self.cvs)
        #return
        print("\n")
        print("    Configuration Interaction Singles") 
        print("    ---------------------------------") 
        print("\n")

        tic = time.time()
  
        #atomic L edge 
        #atomic K edge
        #low = 0
        #high = 1

        if (self.cvs is True):
          low  = self.wfn.options.occupied[0] - 1
          high = self.wfn.options.occupied[1] - 1
          print("    Restricting occupied orbitals in the window [%i,%i]"%(low+1,high+1),flush=True)
          self.no_a = int(high-low+1)
          self.no_b = int(high-low+1)
          self.F_occ_a = self.F_occ_a[low:high+1,low:high+1]
          self.F_occ_b = self.F_occ_b[low:high+1,low:high+1]
          self.Co_a      = self.Co_a[:,low:high+1]
          self.Co_b      = self.Co_b[:,low:high+1]
          self.nov_a = self.no_a * self.nv_a
          self.nov_b = self.no_b * self.nv_b
          self.nroots = self.nov_a + self.nov_b

          self.Cmo_a = np.zeros((self.nbf,self.nv_a+self.no_a))
          self.Cmo_b = np.zeros((self.nbf,self.nv_b+self.no_b))
          self.Cmo_a[:,:self.no_a] = self.Co_a 
          self.Cmo_a[:,self.no_a:] = self.Cv_a 
          self.Cmo_b[:,:self.no_b] = self.Co_b 
          self.Cmo_b[:,self.no_b:] = self.Cv_b 

        print("    Total number of configurations: %i"%(self.nov_a+self.nov_b))
        mem_max = ((self.nov_a+self.nov_b)**2)*8*8/1024/1024
        print("    Maximum memory required: %i mb"%mem_max)
        if mem_max > 32000:
          print("    This computation requires too much memory. Try the Davidson procedure.")
          exit("    Exiting computation.")

        A_ab  = np.einsum("ab,ij->aibj",self.F_virt_b,np.eye(self.no_a),optimize=True)
        A_ab -= np.einsum("ab,ij->aibj",np.eye(self.nv_b),self.F_occ_a,optimize=True)
#        A_bb  = np.einsum("ab,ij->aibj",self.F_virt_b,np.eye(self.no_b),optimize=True)
#        A_bb -= np.einsum("ab,ij->aibj",np.eye(self.nv_b),self.F_occ_b,optimize=True)
          
        self.ints = self.wfn.ints_factory

        jktic = time.time()
        #DRN: For some reason, computing G_abji takes more than 10x longer. Possibly contiguous memory issues?!
        #G_abji  = ao2mo.general(self.ints,[self.Cv_a,self.Cv_a,self.Co_a,self.Co_a],compact=False)
        G_ijBA  = ao2mo.general(self.ints,[self.Co_a,self.Co_a,self.Cv_b,self.Cv_b],compact=False)
        jktoc = time.time()
        print("    (ab|ji) contributionn done in %8.5f seconds."%(jktoc-jktic),flush=True)

        self.G_ABji = G_ijBA.reshape(self.no_a,self.no_a,self.nv_b,self.nv_b).swapaxes(0,3).swapaxes(1,2)

        A_ab -= self.G_ABji.swapaxes(1,3).swapaxes(2,3)*self.options.xcalpha 

        if self.options.nofxc is True:
          print("    Skipping fxc build!")
        else:
          if self.options.xctype != "HF":
            xctic = time.time()
            xc_functional = xc_potential.XC(self.wfn)
            F_aibj, F_aiBJ, F_AIbj, F_AIBJ = xc_functional.computeF([self.Co_a,self.Co_b],[self.Cv_a,self.Cv_b],spin=1)
 
            A_aa += F_aibj
            A_ab += F_aiBJ
            A_ba += F_AIbj
            A_bb += F_AIBJ
             
            xctoc = time.time()
            print("    fxc evaluated in %f seconds"%(xctoc-xctic),flush=True)


        A_ab  = A_ab.reshape(self.no_a*self.nv_b,self.no_a*self.nv_b) 

        evals, X = sp.linalg.eigh(A_ab)
        print(evals*27.21138)

        total_toc = time.time()
        diagon_time = total_toc - tic
        
        print("    Direct diagonalization performed in %8.5f seconds"%diagon_time)
        # exit(0) 
        # why????????????????

        if self.cvs is True:
          #expand original CI vectors to avoid problem with dimensions later
          low  = self.wfn.options.occupied[0] - 1
          high = self.wfn.options.occupied[1] - 1
          Xafull = np.zeros((self.nroots,self.nv_a,self.no_a_full),dtype=complex)
          Xbfull = np.zeros((self.nroots,self.nv_b,self.no_b_full),dtype=complex)
          print("    Expanding occupied space from %i to %i"%(self.no_a,self.no_a_full))
          Xa = X[:self.nov_a,:]
          Xb = X[self.nov_a:,:]
          Xa = (Xa.T).reshape(self.nroots,self.nv_a,self.no_a)
          Xb = (Xb.T).reshape(self.nroots,self.nv_b,self.no_b)
          Xafull[:,:,low:high+1] = Xa
          Xbfull[:,:,low:high+1] = Xb
          Xafull = Xafull.reshape(self.nroots,self.nov_a_full)
          Xbfull = Xbfull.reshape(self.nroots,self.nov_b_full)
          Xfull = np.zeros((self.nroots,self.nov_a_full+self.nov_b_full),dtype=complex)
          Xfull[:,:self.nov_a_full] = Xafull
          Xfull[:,self.nov_a_full:] = Xbfull
          self.Cmo = self.wfn.C
          self.no_a = self.no_a_full
          self.no_b = self.no_b_full
          self.nov_a  = self.nov_a_full
          self.nov_b  = self.nov_b_full
          return evals[:self.nroots], Xfull.T
        else:
          return evals[:self.nroots], X

class RGDIRECT_DIAG():

    def __init__(self, wfn):

        #begin Davidson
        self.wfn = wfn
        self.options = self.wfn.options
        self.mol = wfn.ints_factory
        self.ref = wfn.reference
        self.nbf   = int(2*self.wfn.nbf)
        self.no_a  = int(self.wfn.nel[0])
        self.nv_a  = int(self.nbf - self.no_a)
        self.nov   = int(self.no_a * self.nv_a)
        self.no_a_full  = int(self.wfn.nel[0])
        self.nov_a_full   = int(self.no_a * self.nv_a)
        #print(self.nbf, self.no_a, self.nv_a, self.nov)

        self.S = self.wfn.S
        self.nroots = self.options.nroots
#        self.maxdim = 50 * self.nroots
#        self.maxiter = 50

        self.Cmo = 1.*self.wfn.C[0]

        self.Co = self.Cmo[:,:self.no_a]
        self.Cv = self.Cmo[:,self.no_a:]

        eps_a = self.wfn.eps[0]

        self.F_occ  = np.diag(eps_a[:self.no_a])
        self.F_virt = np.diag(eps_a[self.no_a:])


    def compute(self):
        #print(self.cvs)
        #return
        print("\n")
        print("    Configuration Interaction Singles") 
        print("    ---------------------------------") 
        print("\n")

        tic = time.time()
  
        #atomic L edge 
        #atomic K edge
        #low = 0
        #high = 1

        if (self.cvs is True):
          low  = self.wfn.options.occupied[0] - 1
          high = self.wfn.options.occupied[1] - 1
          print("    Restricting occupied orbitals in the window [%i,%i]"%(low+1,high+1),flush=True)
          self.no_a = int(high-low+1)
          self.F_occ = self.F_occ[low:high+1,low:high+1]
          self.Co      = self.Co[:,low:high+1]
          self.nov = self.no_a * self.nv_a
          print("    Reducing the number of nov pairs from %i to %i"%(self.nov_a_full,self.nov))

          self.Cmo = np.zeros((self.nbf,self.nv_a+self.no_a))
          self.Cmo[:,:self.no_a] = self.Co
          self.Cmo[:,self.no_a:] = self.Cv

          if self.nroots > self.nov:
            print("    WARNING: Too many roots requested. Setting nroots to %i"%self.nov)
            self.nroots = self.nov

        print("    Total number of configurations: %i"%(self.nov))
        mem_max = ((self.nov)**2)*8*4/1024/1024
        print("    Estimated memory required: %i mb"%mem_max,flush=True)
        #if mem_max > 60000:
        #  print("    This computation requires too much memory. Try the Davidson procedure.")
        #  print("    Exiting computation.")
        #  exit(0)

        A  = np.einsum("ab,ij->aibj",self.F_virt,np.eye(self.no_a),optimize=True)
        A -= np.einsum("ab,ij->aibj",np.eye(self.nv_a),self.F_occ,optimize=True)
          
        self.ints = self.wfn.ints_factory

        tic = time.time()

        nbf = self.wfn.nbf
        nel = self.no_a
        Co_a_real = self.Cmo[:nbf,:nel]
        Cv_a_real = self.Cmo[:nbf,nel:]
        Co_b_real = self.Cmo[nbf:,:nel]
        Cv_b_real = self.Cmo[nbf:,nel:]

        #G_voov integrals
        #AA
        G_aijb_real  = ao2mo.general(self.ints,[Cv_a_real,Co_a_real,Co_a_real,Cv_a_real],compact=False)
        print("    (ai|jb) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

        #AB
        G_aijb_real += ao2mo.general(self.ints,[Cv_a_real,Co_a_real,Co_b_real,Cv_b_real],compact=False)
        print("    (ai|JB) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

        #BA
        G_aijb_real += ao2mo.general(self.ints,[Cv_b_real,Co_b_real,Co_a_real,Cv_a_real],compact=False)
        print("    (AI|jb) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

        #BB
        G_aijb_real += ao2mo.general(self.ints,[Cv_b_real,Co_b_real,Co_b_real,Cv_b_real],compact=False)
        print("    (AI|JB) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

        G_aijb = G_aijb_real 
        
        #G_vvoo integrals

        #AA
        #G_abji_real  = ao2mo.general(self.ints,[Cv_a_real,Cv_a_real,Co_a_real,Co_a_real],compact=False)
        G_ijba_real  = ao2mo.general(self.ints,[Co_a_real,Co_a_real,Cv_a_real,Cv_a_real],compact=False)
        print("    (ab|ji) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

        #AB
        G_ijba_real += ao2mo.general(self.ints,[Co_a_real,Co_a_real,Cv_b_real,Cv_b_real],compact=False)
        print("    (ab|JI) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)


        #BA
        G_ijba_real += ao2mo.general(self.ints,[Co_b_real,Co_b_real,Cv_a_real,Cv_a_real],compact=False)
        print("    (AB|ji) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

        G_ijba_real += ao2mo.general(self.ints,[Co_b_real,Co_b_real,Cv_b_real,Cv_b_real],compact=False)
        print("    (AB|JI) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

#        G_abji = G_abji_real
        G_abji = G_ijba_real.reshape(self.no_a,self.no_a,self.nv_a,self.nv_a).swapaxes(0,3).swapaxes(1,2)

        toc = time.time()
        print("    AO to MO transformation time: %5.2f seconds"%(toc -tic),flush=True)

        nv = self.nv_a
        G_aibj = G_aijb.reshape(nv,nel,nel,nv).swapaxes(2,3)
        G_abji = G_abji.reshape(nv,nv,nel,nel)


        A += G_aibj
        A -= G_abji.swapaxes(1,3).swapaxes(2,3)*self.options.xcalpha 

        A  = A.reshape(nel*nv,nel*nv) 

        evals, X = sp.linalg.eigh(A)

        total_toc = time.time()
        diagon_time = total_toc - tic
        
        print("    Direct diagonalization performed in %8.5f seconds"%diagon_time,flush=True)

        if self.cvs is True:
          #expand original CI vectors to avoid problem with dimensions later
          low  = self.wfn.options.occupied[0] - 1
          high = self.wfn.options.occupied[1] - 1
          if self.nov_a_full**2 * 8 *1e-6 > 20000: 
            print("  WARNING: Storing full vector would take %f mb of RAM!!"%(self.nov_a_full**2 * 8 * 1e-6),flush=True)
            print("    Keeping only %i roots"%self.nroots,flush=True)
            print("    New RAM requirement is %f mb ;)"%(self.nov_a_full*self.nroots * 8 * 1e-6),flush=True)
          else:
            print("    Storing full vector will require %f mb of RAM! Keeping full vector"%(self.nov_a_full**2 * 8 * 1e-6),flush=True)
            self.nroots = self.nov

          Xfull = np.zeros((self.nroots,self.nv_a,self.no_a_full))
          print("    Expanding occupied space from %i to %i"%(self.no_a,self.no_a_full),flush=True)
          X = (X.T)[:self.nroots,:].reshape(self.nroots,self.nv_a,self.no_a)
          Xfull[:,:,low:high+1] = X
          Xfull = Xfull.reshape(self.nroots,self.nov_a_full)
   #       print(Xfull.shape)  
          self.Cmo = self.wfn.C[0]
          self.no_a = self.no_a_full
          self.nov = self.nov_a_full
          return evals[:self.nroots], Xfull.T
        else:
          return evals, X

class GDIRECT_DIAG():

    def __init__(self, wfn):

        #begin Davidson
        self.wfn = wfn
        self.options = self.wfn.options
        self.mol = wfn.ints_factory
        self.ref = wfn.reference
        self.nbf   = int(2*self.wfn.nbf)
        self.no_a  = int(self.wfn.nel[0])
        self.nv_a  = int(self.nbf - self.no_a)
        self.nov   = int(self.no_a * self.nv_a)
        self.no_a_full  = int(self.wfn.nel[0])
        self.nov_a_full   = int(self.no_a * self.nv_a)
        #print(self.nbf, self.no_a, self.nv_a, self.nov)

        self.S = self.wfn.S
        self.nroots = self.options.nroots
#        self.maxdim = 50 * self.nroots
#        self.maxiter = 50

        self.Cmo = 1.*self.wfn.C[0]

        self.Co = self.Cmo[:,:self.no_a]
        self.Cv = self.Cmo[:,self.no_a:]

        eps_a = self.wfn.eps[0]

        self.F_occ  = np.diag(eps_a[:self.no_a])
        self.F_virt = np.diag(eps_a[self.no_a:])


    def compute(self):
        #print(self.cvs)
        #return
        print("\n")
        print("    Configuration Interaction Singles") 
        print("    ---------------------------------") 
        print("\n")

        tic = time.time()
  
        #atomic L edge 
        #atomic K edge
        #low = 0
        #high = 1

        if (self.cvs is True):
          low  = self.wfn.options.occupied[0] - 1
          high = self.wfn.options.occupied[1] - 1
          print("    Restricting occupied orbitals in the window [%i,%i]"%(low+1,high+1),flush=True)
          self.no_a = int(high-low+1)
          self.F_occ = self.F_occ[low:high+1,low:high+1]
          self.Co      = self.Co[:,low:high+1]
          self.nov = self.no_a * self.nv_a
          print("    Reducing the number of nov pairs from %i to %i"%(self.nov_a_full,self.nov))

          self.Cmo = np.zeros((self.nbf,self.nv_a+self.no_a),dtype=complex)
          self.Cmo[:,:self.no_a] = self.Co
          self.Cmo[:,self.no_a:] = self.Cv

          if self.nroots > self.nov:
            print("    WARNING: Too many roots requested. Setting nroots to %i"%self.nov)
            self.nroots = self.nov

        print("    Total number of configurations: %i"%(self.nov))
        mem_max = ((self.nov)**2)*8*4/1024/1024
        print("    Estimated memory required: %i mb"%mem_max,flush=True)
        #if mem_max > 60000:
        #  print("    This computation requires too much memory. Try the Davidson procedure.")
        #  print("    Exiting computation.")
        #  exit(0)

        A = np.zeros((self.nv_a,self.no_a,self.nv_a,self.no_a),dtype=complex)

        A += np.einsum("ab,ij->aibj",self.F_virt,np.eye(self.no_a),optimize=True)
        A -= np.einsum("ab,ij->aibj",np.eye(self.nv_a),self.F_occ,optimize=True)
          
        self.ints = self.wfn.ints_factory

        tic = time.time()

        nbf = self.wfn.nbf
        nel = self.no_a
        nv = self.nv_a
        Co_a_real = self.Cmo[:nbf,:nel].real
        Cv_a_real = self.Cmo[:nbf,nel:].real
        Co_b_real = self.Cmo[nbf:,:nel].real
        Cv_b_real = self.Cmo[nbf:,nel:].real

        Co_a_imag = self.Cmo[:nbf,:nel].imag
        Cv_a_imag = self.Cmo[:nbf,nel:].imag
        Co_b_imag = self.Cmo[nbf:,:nel].imag
        Cv_b_imag = self.Cmo[nbf:,nel:].imag

        if (self.options.akonly is True) or (self.options.fonly is True):
          print("    Excluding J-type integrals",flush=True)
        else:
          #G_voov integrals
          #AA
          G_aijb_real  = ao2mo.general(self.ints,[Cv_a_real,Co_a_real,Co_a_real,Cv_a_real],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_a_real,Co_a_real,Co_a_imag,Cv_a_imag],compact=False)
          G_aijb_real -= ao2mo.general(self.ints,[Cv_a_real,Co_a_imag,Co_a_real,Cv_a_imag],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_a_imag,Co_a_real,Co_a_real,Cv_a_imag],compact=False)
          G_aijb_real -= ao2mo.general(self.ints,[Cv_a_imag,Co_a_real,Co_a_imag,Cv_a_real],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_a_imag,Co_a_imag,Co_a_real,Cv_a_real],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_a_real,Co_a_imag,Co_a_imag,Cv_a_real],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_a_imag,Co_a_imag,Co_a_imag,Cv_a_imag],compact=False)

          G_aijb_imag  = ao2mo.general(self.ints,[Cv_a_real,Co_a_real,Co_a_real,Cv_a_imag],compact=False)
          G_aijb_imag -= ao2mo.general(self.ints,[Cv_a_real,Co_a_real,Co_a_imag,Cv_a_real],compact=False)
          G_aijb_imag += ao2mo.general(self.ints,[Cv_a_real,Co_a_imag,Co_a_real,Cv_a_real],compact=False)
          G_aijb_imag += ao2mo.general(self.ints,[Cv_a_real,Co_a_imag,Co_a_imag,Cv_a_imag],compact=False)
          G_aijb_imag -= ao2mo.general(self.ints,[Cv_a_imag,Co_a_real,Co_a_real,Cv_a_real],compact=False)
          G_aijb_imag -= ao2mo.general(self.ints,[Cv_a_imag,Co_a_real,Co_a_imag,Cv_a_imag],compact=False)
          G_aijb_imag += ao2mo.general(self.ints,[Cv_a_imag,Co_a_imag,Co_a_real,Cv_a_imag],compact=False)
          G_aijb_imag -= ao2mo.general(self.ints,[Cv_a_imag,Co_a_imag,Co_a_imag,Cv_a_real],compact=False)
          print("    (ai|jb) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

          #AB
          G_aijb_real += ao2mo.general(self.ints,[Cv_a_real,Co_a_real,Co_b_real,Cv_b_real],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_a_real,Co_a_real,Co_b_imag,Cv_b_imag],compact=False)
          G_aijb_real -= ao2mo.general(self.ints,[Cv_a_real,Co_a_imag,Co_b_real,Cv_b_imag],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_a_imag,Co_a_real,Co_b_real,Cv_b_imag],compact=False)
          G_aijb_real -= ao2mo.general(self.ints,[Cv_a_imag,Co_a_real,Co_b_imag,Cv_b_real],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_a_imag,Co_a_imag,Co_b_real,Cv_b_real],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_a_real,Co_a_imag,Co_b_imag,Cv_b_real],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_a_imag,Co_a_imag,Co_b_imag,Cv_b_imag],compact=False)

          G_aijb_imag += ao2mo.general(self.ints,[Cv_a_real,Co_a_real,Co_b_real,Cv_b_imag],compact=False)
          G_aijb_imag -= ao2mo.general(self.ints,[Cv_a_real,Co_a_real,Co_b_imag,Cv_b_real],compact=False)
          G_aijb_imag += ao2mo.general(self.ints,[Cv_a_real,Co_a_imag,Co_b_real,Cv_b_real],compact=False)
          G_aijb_imag += ao2mo.general(self.ints,[Cv_a_real,Co_a_imag,Co_b_imag,Cv_b_imag],compact=False)
          G_aijb_imag -= ao2mo.general(self.ints,[Cv_a_imag,Co_a_real,Co_b_real,Cv_b_real],compact=False)
          G_aijb_imag -= ao2mo.general(self.ints,[Cv_a_imag,Co_a_real,Co_b_imag,Cv_b_imag],compact=False)
          G_aijb_imag += ao2mo.general(self.ints,[Cv_a_imag,Co_a_imag,Co_b_real,Cv_b_imag],compact=False)
          G_aijb_imag -= ao2mo.general(self.ints,[Cv_a_imag,Co_a_imag,Co_b_imag,Cv_b_real],compact=False)
          print("    (ai|JB) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

          #BA
          G_aijb_real += ao2mo.general(self.ints,[Cv_b_real,Co_b_real,Co_a_real,Cv_a_real],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_b_real,Co_b_real,Co_a_imag,Cv_a_imag],compact=False)
          G_aijb_real -= ao2mo.general(self.ints,[Cv_b_real,Co_b_imag,Co_a_real,Cv_a_imag],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_b_imag,Co_b_real,Co_a_real,Cv_a_imag],compact=False)
          G_aijb_real -= ao2mo.general(self.ints,[Cv_b_imag,Co_b_real,Co_a_imag,Cv_a_real],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_b_imag,Co_b_imag,Co_a_real,Cv_a_real],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_b_real,Co_b_imag,Co_a_imag,Cv_a_real],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_b_imag,Co_b_imag,Co_a_imag,Cv_a_imag],compact=False)

          G_aijb_imag += ao2mo.general(self.ints,[Cv_b_real,Co_b_real,Co_a_real,Cv_a_imag],compact=False)
          G_aijb_imag -= ao2mo.general(self.ints,[Cv_b_real,Co_b_real,Co_a_imag,Cv_a_real],compact=False)
          G_aijb_imag += ao2mo.general(self.ints,[Cv_b_real,Co_b_imag,Co_a_real,Cv_a_real],compact=False)
          G_aijb_imag += ao2mo.general(self.ints,[Cv_b_real,Co_b_imag,Co_a_imag,Cv_a_imag],compact=False)
          G_aijb_imag -= ao2mo.general(self.ints,[Cv_b_imag,Co_b_real,Co_a_real,Cv_a_real],compact=False)
          G_aijb_imag -= ao2mo.general(self.ints,[Cv_b_imag,Co_b_real,Co_a_imag,Cv_a_imag],compact=False)
          G_aijb_imag += ao2mo.general(self.ints,[Cv_b_imag,Co_b_imag,Co_a_real,Cv_a_imag],compact=False)
          G_aijb_imag -= ao2mo.general(self.ints,[Cv_b_imag,Co_b_imag,Co_a_imag,Cv_a_real],compact=False)
          print("    (AI|jb) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

          #BB
          G_aijb_real += ao2mo.general(self.ints,[Cv_b_real,Co_b_real,Co_b_real,Cv_b_real],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_b_real,Co_b_real,Co_b_imag,Cv_b_imag],compact=False)
          G_aijb_real -= ao2mo.general(self.ints,[Cv_b_real,Co_b_imag,Co_b_real,Cv_b_imag],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_b_imag,Co_b_real,Co_b_real,Cv_b_imag],compact=False)
          G_aijb_real -= ao2mo.general(self.ints,[Cv_b_imag,Co_b_real,Co_b_imag,Cv_b_real],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_b_imag,Co_b_imag,Co_b_real,Cv_b_real],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_b_real,Co_b_imag,Co_b_imag,Cv_b_real],compact=False)
          G_aijb_real += ao2mo.general(self.ints,[Cv_b_imag,Co_b_imag,Co_b_imag,Cv_b_imag],compact=False)

          G_aijb_imag += ao2mo.general(self.ints,[Cv_b_real,Co_b_real,Co_b_real,Cv_b_imag],compact=False)
          G_aijb_imag -= ao2mo.general(self.ints,[Cv_b_real,Co_b_real,Co_b_imag,Cv_b_real],compact=False)
          G_aijb_imag += ao2mo.general(self.ints,[Cv_b_real,Co_b_imag,Co_b_real,Cv_b_real],compact=False)
          G_aijb_imag += ao2mo.general(self.ints,[Cv_b_real,Co_b_imag,Co_b_imag,Cv_b_imag],compact=False)
          G_aijb_imag -= ao2mo.general(self.ints,[Cv_b_imag,Co_b_real,Co_b_real,Cv_b_real],compact=False)
          G_aijb_imag -= ao2mo.general(self.ints,[Cv_b_imag,Co_b_real,Co_b_imag,Cv_b_imag],compact=False)
          G_aijb_imag += ao2mo.general(self.ints,[Cv_b_imag,Co_b_imag,Co_b_real,Cv_b_imag],compact=False)
          G_aijb_imag -= ao2mo.general(self.ints,[Cv_b_imag,Co_b_imag,Co_b_imag,Cv_b_real],compact=False)
          print("    (AI|JB) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

          G_aijb = G_aijb_real + 1j*G_aijb_imag
          G_aibj = G_aijb.reshape(nv,nel,nel,nv).swapaxes(2,3)
          A += G_aibj

        if (self.options.fonly is True) or (self.options.jonly):
          print("    Excluding K-type integrals",flush=True)
        else:
          #G_vvoo integrals

          #AA
          G_ijba_real  = ao2mo.general(self.ints,[Co_a_real,Co_a_real,Cv_a_real,Cv_a_real],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_a_real,Co_a_real,Cv_a_imag,Cv_a_imag],compact=False)
          G_ijba_real -= ao2mo.general(self.ints,[Co_a_real,Co_a_imag,Cv_a_real,Cv_a_imag],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_a_imag,Co_a_real,Cv_a_real,Cv_a_imag],compact=False)
          G_ijba_real -= ao2mo.general(self.ints,[Co_a_imag,Co_a_real,Cv_a_imag,Cv_a_real],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_a_imag,Co_a_imag,Cv_a_real,Cv_a_real],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_a_real,Co_a_imag,Cv_a_imag,Cv_a_real],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_a_imag,Co_a_imag,Cv_a_imag,Cv_a_imag],compact=False)

          G_ijba_imag  = ao2mo.general(self.ints,[Co_a_real,Co_a_real,Cv_a_real,Cv_a_imag],compact=False)
          G_ijba_imag -= ao2mo.general(self.ints,[Co_a_real,Co_a_real,Cv_a_imag,Cv_a_real],compact=False)
          G_ijba_imag += ao2mo.general(self.ints,[Co_a_real,Co_a_imag,Cv_a_real,Cv_a_real],compact=False)
          G_ijba_imag += ao2mo.general(self.ints,[Co_a_real,Co_a_imag,Cv_a_imag,Cv_a_imag],compact=False)
          G_ijba_imag -= ao2mo.general(self.ints,[Co_a_imag,Co_a_real,Cv_a_real,Cv_a_real],compact=False)
          G_ijba_imag -= ao2mo.general(self.ints,[Co_a_imag,Co_a_real,Cv_a_imag,Cv_a_imag],compact=False)
          G_ijba_imag += ao2mo.general(self.ints,[Co_a_imag,Co_a_imag,Cv_a_real,Cv_a_imag],compact=False)
          G_ijba_imag -= ao2mo.general(self.ints,[Co_a_imag,Co_a_imag,Cv_a_imag,Cv_a_real],compact=False)
          print("    (ab|ji) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

          #AB
          G_ijba_real += ao2mo.general(self.ints,[Co_a_real,Co_a_real,Cv_b_real,Cv_b_real],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_a_real,Co_a_real,Cv_b_imag,Cv_b_imag],compact=False)
          G_ijba_real -= ao2mo.general(self.ints,[Co_a_real,Co_a_imag,Cv_b_real,Cv_b_imag],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_a_imag,Co_a_real,Cv_b_real,Cv_b_imag],compact=False)
          G_ijba_real -= ao2mo.general(self.ints,[Co_a_imag,Co_a_real,Cv_b_imag,Cv_b_real],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_a_imag,Co_a_imag,Cv_b_real,Cv_b_real],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_a_real,Co_a_imag,Cv_b_imag,Cv_b_real],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_a_imag,Co_a_imag,Cv_b_imag,Cv_b_imag],compact=False)

          G_ijba_imag += ao2mo.general(self.ints,[Co_a_real,Co_a_real,Cv_b_real,Cv_b_imag],compact=False)
          G_ijba_imag -= ao2mo.general(self.ints,[Co_a_real,Co_a_real,Cv_b_imag,Cv_b_real],compact=False)
          G_ijba_imag += ao2mo.general(self.ints,[Co_a_real,Co_a_imag,Cv_b_real,Cv_b_real],compact=False)
          G_ijba_imag += ao2mo.general(self.ints,[Co_a_real,Co_a_imag,Cv_b_imag,Cv_b_imag],compact=False)
          G_ijba_imag -= ao2mo.general(self.ints,[Co_a_imag,Co_a_real,Cv_b_real,Cv_b_real],compact=False)
          G_ijba_imag -= ao2mo.general(self.ints,[Co_a_imag,Co_a_real,Cv_b_imag,Cv_b_imag],compact=False)
          G_ijba_imag += ao2mo.general(self.ints,[Co_a_imag,Co_a_imag,Cv_b_real,Cv_b_imag],compact=False)
          G_ijba_imag -= ao2mo.general(self.ints,[Co_a_imag,Co_a_imag,Cv_b_imag,Cv_b_real],compact=False)
          print("    (ab|JI) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

          #BA
          G_ijba_real += ao2mo.general(self.ints,[Co_b_real,Co_b_real,Cv_a_real,Cv_a_real],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_b_real,Co_b_real,Cv_a_imag,Cv_a_imag],compact=False)
          G_ijba_real -= ao2mo.general(self.ints,[Co_b_real,Co_b_imag,Cv_a_real,Cv_a_imag],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_b_imag,Co_b_real,Cv_a_real,Cv_a_imag],compact=False)
          G_ijba_real -= ao2mo.general(self.ints,[Co_b_imag,Co_b_real,Cv_a_imag,Cv_a_real],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_b_imag,Co_b_imag,Cv_a_real,Cv_a_real],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_b_real,Co_b_imag,Cv_a_imag,Cv_a_real],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_b_imag,Co_b_imag,Cv_a_imag,Cv_a_imag],compact=False)

          G_ijba_imag += ao2mo.general(self.ints,[Co_b_real,Co_b_real,Cv_a_real,Cv_a_imag],compact=False)
          G_ijba_imag -= ao2mo.general(self.ints,[Co_b_real,Co_b_real,Cv_a_imag,Cv_a_real],compact=False)
          G_ijba_imag += ao2mo.general(self.ints,[Co_b_real,Co_b_imag,Cv_a_real,Cv_a_real],compact=False)
          G_ijba_imag += ao2mo.general(self.ints,[Co_b_real,Co_b_imag,Cv_a_imag,Cv_a_imag],compact=False)
          G_ijba_imag -= ao2mo.general(self.ints,[Co_b_imag,Co_b_real,Cv_a_real,Cv_a_real],compact=False)
          G_ijba_imag -= ao2mo.general(self.ints,[Co_b_imag,Co_b_real,Cv_a_imag,Cv_a_imag],compact=False)
          G_ijba_imag += ao2mo.general(self.ints,[Co_b_imag,Co_b_imag,Cv_a_real,Cv_a_imag],compact=False)
          G_ijba_imag -= ao2mo.general(self.ints,[Co_b_imag,Co_b_imag,Cv_a_imag,Cv_a_real],compact=False)
          print("    (AB|ji) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

          G_ijba_real += ao2mo.general(self.ints,[Co_b_real,Co_b_real,Cv_b_real,Cv_b_real],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_b_real,Co_b_real,Cv_b_imag,Cv_b_imag],compact=False)
          G_ijba_real -= ao2mo.general(self.ints,[Co_b_real,Co_b_imag,Cv_b_real,Cv_b_imag],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_b_imag,Co_b_real,Cv_b_real,Cv_b_imag],compact=False)
          G_ijba_real -= ao2mo.general(self.ints,[Co_b_imag,Co_b_real,Cv_b_imag,Cv_b_real],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_b_imag,Co_b_imag,Cv_b_real,Cv_b_real],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_b_real,Co_b_imag,Cv_b_imag,Cv_b_real],compact=False)
          G_ijba_real += ao2mo.general(self.ints,[Co_b_imag,Co_b_imag,Cv_b_imag,Cv_b_imag],compact=False)

          G_ijba_imag += ao2mo.general(self.ints,[Co_b_real,Co_b_real,Cv_b_real,Cv_b_imag],compact=False)
          G_ijba_imag -= ao2mo.general(self.ints,[Co_b_real,Co_b_real,Cv_b_imag,Cv_b_real],compact=False)
          G_ijba_imag += ao2mo.general(self.ints,[Co_b_real,Co_b_imag,Cv_b_real,Cv_b_real],compact=False)
          G_ijba_imag += ao2mo.general(self.ints,[Co_b_real,Co_b_imag,Cv_b_imag,Cv_b_imag],compact=False)
          G_ijba_imag -= ao2mo.general(self.ints,[Co_b_imag,Co_b_real,Cv_b_real,Cv_b_real],compact=False)
          G_ijba_imag -= ao2mo.general(self.ints,[Co_b_imag,Co_b_real,Cv_b_imag,Cv_b_imag],compact=False)
          G_ijba_imag += ao2mo.general(self.ints,[Co_b_imag,Co_b_imag,Cv_b_real,Cv_b_imag],compact=False)
          G_ijba_imag -= ao2mo.general(self.ints,[Co_b_imag,Co_b_imag,Cv_b_imag,Cv_b_real],compact=False)

          G_abji_real = G_ijba_real.reshape(self.no_a,self.no_a,self.nv_a,self.nv_a).swapaxes(0,3).swapaxes(1,2)
          G_abji_imag = G_ijba_imag.reshape(self.no_a,self.no_a,self.nv_a,self.nv_a).swapaxes(0,3).swapaxes(1,2)
          print("    (AB|JI) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

          G_abji = G_abji_real + 1j*G_abji_imag

          G_abji = G_abji.reshape(nv,nv,nel,nel)

          A -= G_abji.swapaxes(1,3).swapaxes(2,3)*self.options.xcalpha 

        A  = A.reshape(nel*nv,nel*nv) 

        evals, X = sp.linalg.eigh(A)

        total_toc = time.time()
        diagon_time = total_toc - tic
        
        print("    Direct diagonalization performed in %8.5f seconds"%diagon_time,flush=True)

        if self.cvs is True:
          #expand original CI vectors to avoid problem with dimensions later
          low  = self.wfn.options.occupied[0] - 1
          high = self.wfn.options.occupied[1] - 1
          if self.nov_a_full**2 * 8 *1e-6 > 20000: 
            print("  WARNING: Storing full vector would take %f mb of RAM!!"%(self.nov_a_full**2 * 8 * 1e-6),flush=True)
            print("    Keeping only %i roots"%self.nroots,flush=True)
            print("    New RAM requirement is %f mb ;)"%(self.nov_a_full*self.nroots * 8 * 1e-6),flush=True)
          else:
            print("    Storing full vector will require %f mb of RAM! Keeping full vector"%(self.nov_a_full**2 * 8 * 1e-6),flush=True)
            self.nroots = self.nov

          Xfull = np.zeros((self.nroots,self.nv_a,self.no_a_full),dtype=complex)
          print("    Expanding occupied space from %i to %i"%(self.no_a,self.no_a_full),flush=True)
          X = (X.T)[:self.nroots,:].reshape(self.nroots,self.nv_a,self.no_a)
          Xfull[:,:,low:high+1] = X
          Xfull = Xfull.reshape(self.nroots,self.nov_a_full)
   #       print(Xfull.shape)  
          self.Cmo = self.wfn.C[0]
          self.no_a = self.no_a_full
          self.nov = self.nov_a_full
          return evals[:self.nroots], Xfull.T
        else:
          return evals, X
