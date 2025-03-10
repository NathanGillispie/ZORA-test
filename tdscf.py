import numpy as np
import sys
import os
sys.path.insert(1,os.path.abspath(os.path.dirname(__file__)+"/cpp"))
import scipy as sp
import diis_routine
import time
#import matplotlib.pyplot as plt
from pyscf import dft, scf, ao2mo
import utils
import davidson
import direct_diag
import xc_potential
np.set_printoptions(precision=5, linewidth=200, suppress=False,threshold=sys.maxsize)

class TDSCF():

    def __init__(self,wfn):
      self.wfn = wfn 
      self.options = wfn.options
      self.ints = wfn.ints_factory
      
      print("\n    Initializing TDSCF Class")

    def tda(self):
      print("    Starting TDA Algorithm")    
  
      if self.options.tdscf_in_core is True: 
          print("\n    In-core algorithm requested. Computing two-electron integrals",flush=True)
          print("    This may take a while...\n",flush=True)
  
          if (self.wfn.reference == "rks"):
            tic = time.time()
 
            nel_a = self.wfn.nel[0]
            Co_a  = self.wfn.C[0][:,:nel_a]
            Cv_a  = self.wfn.C[0][:,nel_a:]
            G_AO = self.ints.intor('int2e')
  
            G_aibj  = ao2mo.general(G_AO,[Cv_a,Co_a,Cv_a,Co_a],compact=False)
            print("    (ai|bj) contributionn done.",flush=True)
  
            G_abji  = ao2mo.general(G_AO,[Cv_a,Cv_a,Co_a,Co_a],compact=False)
            print("    (ab|ji) contributionn done.",flush=True)
  
            toc = time.time()
            print("    AO to MO transformation time: %5.2f seconds"%(toc -tic),flush=True)
            
            nv_a = self.wfn.nbf - nel_a
            G_aibj = G_aibj.reshape(nv_a,nel_a,nv_a,nel_a)
            G_abji = G_abji.reshape(nv_a,nv_a,nel_a,nel_a)

            if self.options.xctype == "HF":
              self.wfn.G_array = [2.*G_aibj , G_abji]
            elif self.options.plus_tb is True:
              print("    Building exchange-correlation kernel from TB model",flush=True)
              Gamma = self.options.tb_gamma
              F_aibj = np.einsum("Aia,AB,Bjb->aibj",self.wfn.QAia,Gamma,self.wfn.QAia,optimize=True)
              #self.wfn.G_array = [0.*G_aibj + 0.*F_aibj, self.options.xcalpha * G_abji]
              self.wfn.G_array = [2.*G_aibj + 2.*F_aibj, self.options.xcalpha * G_abji]
            else:
              xc_functional = xc_potential.XC(self.wfn)
              F_aibj = xc_functional.computeF(Co_a,Cv_a,spin=0) 
              self.wfn.G_array = [2.*G_aibj + 2.*F_aibj[0], self.options.xcalpha * G_abji]
            singlet = davidson.RDAVIDSON(self.wfn)
            singlet.reduced_virtual = self.options.reduced_virtual
            singlet.cvs = self.options.cvs
            w_s, X_s = singlet.compute()
            

            Co  = [Co_a, Co_a]
            Cv  = [Cv_a, Cv_a]

            ##DRN COMMENTED: (uncomment this part)
            if self.options.xctype == "HF":
              self.wfn.G_array = [G_aibj*0., G_abji]
            elif self.options.plus_tb is True:
              print("    Building exchange-correlation kernel from TB model",flush=True)
              W = self.options.molecule["tb_w"]
              F_aibj = np.einsum("Aia,AB,Bjb->aibj",self.wfn.QAia,W,self.wfn.QAia,optimize=True)
              self.wfn.G_array = [F_aibj, self.options.xcalpha * G_abji]
            else:
              #For restricted triplets, need to re-evaluate the xc kernel as an open-shell system
              xc_functional = xc_potential.XC(self.wfn)
              F_aibj = xc_functional.computeF(Co,Cv,spin=1) 
              self.wfn.G_array = [F_aibj[0] - F_aibj[1], self.options.xcalpha*G_abji]

            triplet = davidson.RDAVIDSON(self.wfn)
            triplet.cvs = self.options.cvs
            triplet.reduced_virtual = self.options.reduced_virtual
            w_t, X_t = triplet.compute()
            self.tda_analyze_restricted(w_s,np.sqrt(2.)*X_s,singlet,w_t,np.sqrt(2.)*X_t,triplet)
            ##END HERE

          elif (self.wfn.reference == "uks"):
            tic = time.time()
 
            nel_a = self.wfn.nel[0]
            Co_a  = self.wfn.C[0][:,:nel_a]
            Cv_a  = self.wfn.C[0][:,nel_a:]
  
            G_aibj  = ao2mo.general(self.ints,[Cv_a,Co_a,Cv_a,Co_a],compact=False)
            print("    (ai|bj) contributionn done.",flush=True)
  
            G_abji  = ao2mo.general(self.ints,[Cv_a,Cv_a,Co_a,Co_a],compact=False)
            print("    (ab|ji) contributionn done.",flush=True)
  
            
            nv_a = self.wfn.nbf - nel_a
            G_aibj = G_aibj.reshape(nv_a,nel_a,nv_a,nel_a)
            G_abji = G_abji.reshape(nv_a,nv_a,nel_a,nel_a)

            nel_b = self.wfn.nel[1]
            nv_b = self.wfn.nbf - nel_b
            Co_b = self.wfn.C[1][:,:nel_b].real
            Cv_b = self.wfn.C[1][:,nel_b:].real

            Co  = [Co_a, Co_b]
            Cv  = [Cv_a, Cv_b]

            G_aiBJ  = ao2mo.general(self.ints,[Cv_a,Co_a,Cv_b,Co_b],compact=False)
            print("    (ai|BJ) contributionn done.",flush=True)
 
            G_AIbj  = ao2mo.general(self.ints,[Cv_b,Co_b,Cv_a,Co_a],compact=False)
            print("    (ai|BJ) contributionn done.",flush=True)
  
            G_AIBJ  = ao2mo.general(self.ints,[Cv_b,Co_b,Cv_b,Co_b],compact=False)
            print("    (AI|BJ) contributionn done.",flush=True)

            G_ABJI  = ao2mo.general(self.ints,[Cv_b,Cv_b,Co_b,Co_b],compact=False)
            print("    (ab|ji) contributionn done.",flush=True)
            toc = time.time()
            print("    AO to MO transformation time: %5.2f seconds"%(toc -tic),flush=True)
          
            G_aiBJ = G_aiBJ.reshape(nv_a,nel_a,nv_b,nel_b)
            G_AIbj = G_AIbj.reshape(nv_b,nel_b,nv_a,nel_a)

            G_AIBJ = G_AIBJ.reshape(nv_b,nel_b,nv_b,nel_b)
            G_ABJI = G_ABJI.reshape(nv_b,nv_b,nel_b,nel_b)
        
            if self.options.xctype == "HF": 
              K_aibj = [G_aibj ,G_aiBJ ,G_AIbj,G_AIBJ]
              K_abji = [G_abji, G_ABJI]
              self.wfn.G_array = [K_aibj, K_abji]
            else:
              xc_functional = xc_potential.XC(self.wfn)
              F_aibj, F_aiBJ, F_AIbj, F_AIBJ = xc_functional.computeF(Co,Cv,spin=1) 
              K_aibj = [G_aibj + F_aibj,G_aiBJ + F_aiBJ,G_AIbj + F_AIbj,G_AIBJ + F_AIBJ]
              K_abji = [self.options.xcalpha*G_abji,self.options.xcalpha*G_ABJI]
              self.wfn.G_array = [K_aibj, K_abji]

            sol = davidson.UDAVIDSON(self.wfn)
            sol.cvs = self.options.cvs
            w, X = sol.compute()
            self.tda_analyze_unrestricted(w,X,sol)

          #elif (self.wfn.reference == "gks"):
          #  exit("    ERROR: TD-GKS not implemented.")

          elif (self.wfn.reference == "rgks"):
            tic = time.time()

            nbf = self.wfn.nbf 
            nel = self.wfn.nel[0]
            Co_a_real = self.wfn.C[0][:nbf,:nel]
            Cv_a_real = self.wfn.C[0][:nbf,nel:]
            Co_b_real = self.wfn.C[0][nbf:,:nel]
            Cv_b_real = self.wfn.C[0][nbf:,nel:]

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
            G_abji_real  = ao2mo.general(self.ints,[Cv_a_real,Cv_a_real,Co_a_real,Co_a_real],compact=False)
            print("    (ab|ji) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

            #AB
            G_abji_real += ao2mo.general(self.ints,[Cv_a_real,Cv_a_real,Co_b_real,Co_b_real],compact=False)
            print("    (ab|JI) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)


            #BA
            G_abji_real += ao2mo.general(self.ints,[Cv_b_real,Cv_b_real,Co_a_real,Co_a_real],compact=False)
            print("    (AB|ji) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

            G_abji_real += ao2mo.general(self.ints,[Cv_b_real,Cv_b_real,Co_b_real,Co_b_real],compact=False)
            print("    (AB|JI) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

            G_abji = G_abji_real

            toc = time.time()
            print("    AO to MO transformation time: %5.2f seconds"%(toc -tic),flush=True)
          
            nv = 2*self.wfn.nbf - nel
            G_aibj = G_aijb.reshape(nv,nel,nel,nv).swapaxes(2,3)
            G_abji = G_abji.reshape(nv,nv,nel,nel)
            if self.options.xctype == "HF":
              self.wfn.G_array = [G_aibj, G_abji]
            else:
             # #Redefine some of the wavefunction quantities to adopt a open-shell-like structure
             # Co_a = self.wfn.C[0][:nbf,:nel]
             # Cv_a = self.wfn.C[0][:nbf,nel:]
             # Co_b = self.wfn.C[0][nbf:,:nel]
             # Cv_b = self.wfn.C[0][nbf:,nel:]

             # Daa = self.wfn.D[0][:nbf,:nbf]
             # Dbb = self.wfn.D[0][nbf:,nbf:]

             # #noncollinear densities also saved but won't be used because fxc is collinear
             # Dab = self.wfn.D[0][:nbf,nbf:]
             # Dba = self.wfn.D[0][nbf:,:nbf]
             # 
             # # Real density
             # self.wfn.D = [Daa,Dbb]
             # self.wfn.nbf *= 2
             # xc_functional = xc_potential.XC(self.wfn)
             # F_aibj = xc_functional.computeF([Co_a,Co_b],[Cv_a,Cv_b],spin=1) 
             # #K_aibj = G_aibj + F_aibj[0] + F_aibj[1] + F_aibj[2] + F_aibj[3]
             # K_aibj = G_aibj #- 2.*F_aibj[0] - 2.*F_aibj[3] + F_aibj[2] + F_aibj[1]
             # self.wfn.G_array = [K_aibj, self.options.xcalpha * G_abji]
              self.wfn.G_array = [G_aibj, self.options.xcalpha * G_abji]
             # #self.wfn.G_array = [K_aibj, G_abji]
             # self.wfn.nbf //= 2
             # self.wfn.D = np.asarray([[Daa,Dab],[Dba,Dbb]])  
           
            cis = davidson.RGDAVIDSON(self.wfn)
            cis.cvs = self.options.cvs
            w, X = cis.compute()
            self.tda_analyze_generalized(w,X,cis)

          elif (self.wfn.reference == "uhf"):
            tic = time.time()
 
            nel_a = self.wfn.nel[0]
            Co_a  = self.wfn.C[0][:,:nel_a]
            Cv_a  = self.wfn.C[0][:,nel_a:]
  
            G_aibj  = ao2mo.general(self.ints,[Cv_a,Co_a,Cv_a,Co_a],compact=False)
            print("    (ai|bj) contributionn done.",flush=True)
  
            G_abji  = ao2mo.general(self.ints,[Cv_a,Cv_a,Co_a,Co_a],compact=False)
            print("    (ab|ji) contributionn done.",flush=True)
            
            nv_a = self.wfn.nbf - nel_a
            G_aibj = G_aibj.reshape(nv_a,nel_a,nv_a,nel_a)
            G_abji = G_abji.reshape(nv_a,nv_a,nel_a,nel_a)

            nel_b = self.wfn.nel[1]
            nv_b = self.wfn.nbf - nel_b
            Co_b = self.wfn.C[1][:,:nel_b]
            Cv_b = self.wfn.C[1][:,nel_b:]

            Co  = [Co_a, Co_b]
            Cv  = [Cv_a, Cv_b]

            G_aiBJ  = ao2mo.general(self.ints,[Cv_a,Co_a,Cv_b,Co_b],compact=False)
            print("    (ai|BJ) contributionn done.",flush=True)
  
            G_AIBJ  = ao2mo.general(self.ints,[Cv_b,Co_b,Cv_b,Co_b],compact=False)
            print("    (AI|BJ) contributionn done.",flush=True)

            G_ABJI  = ao2mo.general(self.ints,[Cv_b,Cv_b,Co_b,Co_b],compact=False)
            print("    (ab|ji) contributionn done.",flush=True)
            toc = time.time()
            print("    AO to MO transformation time: %5.2f seconds"%(toc -tic),flush=True)
          
            G_aiBJ = G_aiBJ.reshape(nv_a,nel_a,nv_b,nel_b)
            G_AIbj = (G_aiBJ.swapaxes(0,2).swapaxes(1,3)).reshape(nv_b,nel_b,nv_a,nel_a)

            G_AIBJ = G_AIBJ.reshape(nv_b,nel_b,nv_b,nel_b)
            G_ABJI = G_ABJI.reshape(nv_b,nv_b,nel_b,nel_b)

            K_aibj = [G_aibj,G_aiBJ,G_AIbj,G_AIBJ]
            K_abji = [G_abji,G_ABJI]
 
            self.wfn.G_array = [K_aibj, K_abji]
            sol = davidson.UDAVIDSON(self.wfn)
            sol.cvs = self.options.cvs
            w, X = sol.compute()
            self.tda_analyze_unrestricted(w,X,sol)

          elif (self.wfn.reference == "ghf" or self.wfn.reference == "gks"):
            tic = time.time()
            G_AO = self.ints.intor('int2e')

            nbf = self.wfn.nbf 
            nel = self.wfn.nel[0]
            Co_a_real = self.wfn.C[0][:nbf,:nel].real
            Cv_a_real = self.wfn.C[0][:nbf,nel:].real
            Co_b_real = self.wfn.C[0][nbf:,:nel].real
            Cv_b_real = self.wfn.C[0][nbf:,nel:].real

            Co_a_imag = self.wfn.C[0][:nbf,:nel].imag
            Cv_a_imag = self.wfn.C[0][:nbf,nel:].imag
            Co_b_imag = self.wfn.C[0][nbf:,:nel].imag
            Cv_b_imag = self.wfn.C[0][nbf:,nel:].imag


            #G_voov integrals
            #AA
            G_aijb_real  = ao2mo.general(G_AO,[Cv_a_real,Co_a_real,Co_a_real,Cv_a_real],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_a_real,Co_a_real,Co_a_imag,Cv_a_imag],compact=False)
            G_aijb_real -= ao2mo.general(G_AO,[Cv_a_real,Co_a_imag,Co_a_real,Cv_a_imag],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_a_imag,Co_a_real,Co_a_real,Cv_a_imag],compact=False)
            G_aijb_real -= ao2mo.general(G_AO,[Cv_a_imag,Co_a_real,Co_a_imag,Cv_a_real],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_a_imag,Co_a_imag,Co_a_real,Cv_a_real],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_a_real,Co_a_imag,Co_a_imag,Cv_a_real],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_a_imag,Co_a_imag,Co_a_imag,Cv_a_imag],compact=False)

            G_aijb_imag  = ao2mo.general(G_AO,[Cv_a_real,Co_a_real,Co_a_real,Cv_a_imag],compact=False)
            G_aijb_imag -= ao2mo.general(G_AO,[Cv_a_real,Co_a_real,Co_a_imag,Cv_a_real],compact=False)
            G_aijb_imag += ao2mo.general(G_AO,[Cv_a_real,Co_a_imag,Co_a_real,Cv_a_real],compact=False)
            G_aijb_imag += ao2mo.general(G_AO,[Cv_a_real,Co_a_imag,Co_a_imag,Cv_a_imag],compact=False)
            G_aijb_imag -= ao2mo.general(G_AO,[Cv_a_imag,Co_a_real,Co_a_real,Cv_a_real],compact=False)
            G_aijb_imag -= ao2mo.general(G_AO,[Cv_a_imag,Co_a_real,Co_a_imag,Cv_a_imag],compact=False)
            G_aijb_imag += ao2mo.general(G_AO,[Cv_a_imag,Co_a_imag,Co_a_real,Cv_a_imag],compact=False)
            G_aijb_imag -= ao2mo.general(G_AO,[Cv_a_imag,Co_a_imag,Co_a_imag,Cv_a_real],compact=False)
            print("    (ai|jb) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

            #AB
            G_aijb_real += ao2mo.general(G_AO,[Cv_a_real,Co_a_real,Co_b_real,Cv_b_real],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_a_real,Co_a_real,Co_b_imag,Cv_b_imag],compact=False)
            G_aijb_real -= ao2mo.general(G_AO,[Cv_a_real,Co_a_imag,Co_b_real,Cv_b_imag],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_a_imag,Co_a_real,Co_b_real,Cv_b_imag],compact=False)
            G_aijb_real -= ao2mo.general(G_AO,[Cv_a_imag,Co_a_real,Co_b_imag,Cv_b_real],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_a_imag,Co_a_imag,Co_b_real,Cv_b_real],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_a_real,Co_a_imag,Co_b_imag,Cv_b_real],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_a_imag,Co_a_imag,Co_b_imag,Cv_b_imag],compact=False)

            G_aijb_imag += ao2mo.general(G_AO,[Cv_a_real,Co_a_real,Co_b_real,Cv_b_imag],compact=False)
            G_aijb_imag -= ao2mo.general(G_AO,[Cv_a_real,Co_a_real,Co_b_imag,Cv_b_real],compact=False)
            G_aijb_imag += ao2mo.general(G_AO,[Cv_a_real,Co_a_imag,Co_b_real,Cv_b_real],compact=False)
            G_aijb_imag += ao2mo.general(G_AO,[Cv_a_real,Co_a_imag,Co_b_imag,Cv_b_imag],compact=False)
            G_aijb_imag -= ao2mo.general(G_AO,[Cv_a_imag,Co_a_real,Co_b_real,Cv_b_real],compact=False)
            G_aijb_imag -= ao2mo.general(G_AO,[Cv_a_imag,Co_a_real,Co_b_imag,Cv_b_imag],compact=False)
            G_aijb_imag += ao2mo.general(G_AO,[Cv_a_imag,Co_a_imag,Co_b_real,Cv_b_imag],compact=False)
            G_aijb_imag -= ao2mo.general(G_AO,[Cv_a_imag,Co_a_imag,Co_b_imag,Cv_b_real],compact=False)
            print("    (ai|JB) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

            #BA
            G_aijb_real += ao2mo.general(G_AO,[Cv_b_real,Co_b_real,Co_a_real,Cv_a_real],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_b_real,Co_b_real,Co_a_imag,Cv_a_imag],compact=False)
            G_aijb_real -= ao2mo.general(G_AO,[Cv_b_real,Co_b_imag,Co_a_real,Cv_a_imag],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_b_imag,Co_b_real,Co_a_real,Cv_a_imag],compact=False)
            G_aijb_real -= ao2mo.general(G_AO,[Cv_b_imag,Co_b_real,Co_a_imag,Cv_a_real],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_b_imag,Co_b_imag,Co_a_real,Cv_a_real],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_b_real,Co_b_imag,Co_a_imag,Cv_a_real],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_b_imag,Co_b_imag,Co_a_imag,Cv_a_imag],compact=False)

            G_aijb_imag += ao2mo.general(G_AO,[Cv_b_real,Co_b_real,Co_a_real,Cv_a_imag],compact=False)
            G_aijb_imag -= ao2mo.general(G_AO,[Cv_b_real,Co_b_real,Co_a_imag,Cv_a_real],compact=False)
            G_aijb_imag += ao2mo.general(G_AO,[Cv_b_real,Co_b_imag,Co_a_real,Cv_a_real],compact=False)
            G_aijb_imag += ao2mo.general(G_AO,[Cv_b_real,Co_b_imag,Co_a_imag,Cv_a_imag],compact=False)
            G_aijb_imag -= ao2mo.general(G_AO,[Cv_b_imag,Co_b_real,Co_a_real,Cv_a_real],compact=False)
            G_aijb_imag -= ao2mo.general(G_AO,[Cv_b_imag,Co_b_real,Co_a_imag,Cv_a_imag],compact=False)
            G_aijb_imag += ao2mo.general(G_AO,[Cv_b_imag,Co_b_imag,Co_a_real,Cv_a_imag],compact=False)
            G_aijb_imag -= ao2mo.general(G_AO,[Cv_b_imag,Co_b_imag,Co_a_imag,Cv_a_real],compact=False)
            print("    (AI|jb) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

            #BB
            G_aijb_real += ao2mo.general(G_AO,[Cv_b_real,Co_b_real,Co_b_real,Cv_b_real],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_b_real,Co_b_real,Co_b_imag,Cv_b_imag],compact=False)
            G_aijb_real -= ao2mo.general(G_AO,[Cv_b_real,Co_b_imag,Co_b_real,Cv_b_imag],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_b_imag,Co_b_real,Co_b_real,Cv_b_imag],compact=False)
            G_aijb_real -= ao2mo.general(G_AO,[Cv_b_imag,Co_b_real,Co_b_imag,Cv_b_real],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_b_imag,Co_b_imag,Co_b_real,Cv_b_real],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_b_real,Co_b_imag,Co_b_imag,Cv_b_real],compact=False)
            G_aijb_real += ao2mo.general(G_AO,[Cv_b_imag,Co_b_imag,Co_b_imag,Cv_b_imag],compact=False)

            G_aijb_imag += ao2mo.general(G_AO,[Cv_b_real,Co_b_real,Co_b_real,Cv_b_imag],compact=False)
            G_aijb_imag -= ao2mo.general(G_AO,[Cv_b_real,Co_b_real,Co_b_imag,Cv_b_real],compact=False)
            G_aijb_imag += ao2mo.general(G_AO,[Cv_b_real,Co_b_imag,Co_b_real,Cv_b_real],compact=False)
            G_aijb_imag += ao2mo.general(G_AO,[Cv_b_real,Co_b_imag,Co_b_imag,Cv_b_imag],compact=False)
            G_aijb_imag -= ao2mo.general(G_AO,[Cv_b_imag,Co_b_real,Co_b_real,Cv_b_real],compact=False)
            G_aijb_imag -= ao2mo.general(G_AO,[Cv_b_imag,Co_b_real,Co_b_imag,Cv_b_imag],compact=False)
            G_aijb_imag += ao2mo.general(G_AO,[Cv_b_imag,Co_b_imag,Co_b_real,Cv_b_imag],compact=False)
            G_aijb_imag -= ao2mo.general(G_AO,[Cv_b_imag,Co_b_imag,Co_b_imag,Cv_b_real],compact=False)
            print("    (AI|JB) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

            G_aijb = G_aijb_real + 1j*G_aijb_imag

            
            #G_vvoo integrals

            #AA
            G_abji_real  = ao2mo.general(G_AO,[Cv_a_real,Cv_a_real,Co_a_real,Co_a_real],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_a_real,Cv_a_real,Co_a_imag,Co_a_imag],compact=False)
            G_abji_real -= ao2mo.general(G_AO,[Cv_a_real,Cv_a_imag,Co_a_real,Co_a_imag],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_a_imag,Cv_a_real,Co_a_real,Co_a_imag],compact=False)
            G_abji_real -= ao2mo.general(G_AO,[Cv_a_imag,Cv_a_real,Co_a_imag,Co_a_real],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_a_imag,Cv_a_imag,Co_a_real,Co_a_real],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_a_real,Cv_a_imag,Co_a_imag,Co_a_real],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_a_imag,Cv_a_imag,Co_a_imag,Co_a_imag],compact=False)

            G_abji_imag  = ao2mo.general(G_AO,[Cv_a_real,Cv_a_real,Co_a_real,Co_a_imag],compact=False)
            G_abji_imag -= ao2mo.general(G_AO,[Cv_a_real,Cv_a_real,Co_a_imag,Co_a_real],compact=False)
            G_abji_imag += ao2mo.general(G_AO,[Cv_a_real,Cv_a_imag,Co_a_real,Co_a_real],compact=False)
            G_abji_imag += ao2mo.general(G_AO,[Cv_a_real,Cv_a_imag,Co_a_imag,Co_a_imag],compact=False)
            G_abji_imag -= ao2mo.general(G_AO,[Cv_a_imag,Cv_a_real,Co_a_real,Co_a_real],compact=False)
            G_abji_imag -= ao2mo.general(G_AO,[Cv_a_imag,Cv_a_real,Co_a_imag,Co_a_imag],compact=False)
            G_abji_imag += ao2mo.general(G_AO,[Cv_a_imag,Cv_a_imag,Co_a_real,Co_a_imag],compact=False)
            G_abji_imag -= ao2mo.general(G_AO,[Cv_a_imag,Cv_a_imag,Co_a_imag,Co_a_real],compact=False)
            print("    (ab|ji) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

            #AB
            G_abji_real += ao2mo.general(G_AO,[Cv_a_real,Cv_a_real,Co_b_real,Co_b_real],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_a_real,Cv_a_real,Co_b_imag,Co_b_imag],compact=False)
            G_abji_real -= ao2mo.general(G_AO,[Cv_a_real,Cv_a_imag,Co_b_real,Co_b_imag],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_a_imag,Cv_a_real,Co_b_real,Co_b_imag],compact=False)
            G_abji_real -= ao2mo.general(G_AO,[Cv_a_imag,Cv_a_real,Co_b_imag,Co_b_real],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_a_imag,Cv_a_imag,Co_b_real,Co_b_real],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_a_real,Cv_a_imag,Co_b_imag,Co_b_real],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_a_imag,Cv_a_imag,Co_b_imag,Co_b_imag],compact=False)

            G_abji_imag += ao2mo.general(G_AO,[Cv_a_real,Cv_a_real,Co_b_real,Co_b_imag],compact=False)
            G_abji_imag -= ao2mo.general(G_AO,[Cv_a_real,Cv_a_real,Co_b_imag,Co_b_real],compact=False)
            G_abji_imag += ao2mo.general(G_AO,[Cv_a_real,Cv_a_imag,Co_b_real,Co_b_real],compact=False)
            G_abji_imag += ao2mo.general(G_AO,[Cv_a_real,Cv_a_imag,Co_b_imag,Co_b_imag],compact=False)
            G_abji_imag -= ao2mo.general(G_AO,[Cv_a_imag,Cv_a_real,Co_b_real,Co_b_real],compact=False)
            G_abji_imag -= ao2mo.general(G_AO,[Cv_a_imag,Cv_a_real,Co_b_imag,Co_b_imag],compact=False)
            G_abji_imag += ao2mo.general(G_AO,[Cv_a_imag,Cv_a_imag,Co_b_real,Co_b_imag],compact=False)
            G_abji_imag -= ao2mo.general(G_AO,[Cv_a_imag,Cv_a_imag,Co_b_imag,Co_b_real],compact=False)
            print("    (ab|JI) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)


            #BA
            G_abji_real += ao2mo.general(G_AO,[Cv_b_real,Cv_b_real,Co_a_real,Co_a_real],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_b_real,Cv_b_real,Co_a_imag,Co_a_imag],compact=False)
            G_abji_real -= ao2mo.general(G_AO,[Cv_b_real,Cv_b_imag,Co_a_real,Co_a_imag],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_b_imag,Cv_b_real,Co_a_real,Co_a_imag],compact=False)
            G_abji_real -= ao2mo.general(G_AO,[Cv_b_imag,Cv_b_real,Co_a_imag,Co_a_real],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_b_imag,Cv_b_imag,Co_a_real,Co_a_real],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_b_real,Cv_b_imag,Co_a_imag,Co_a_real],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_b_imag,Cv_b_imag,Co_a_imag,Co_a_imag],compact=False)

            G_abji_imag += ao2mo.general(G_AO,[Cv_b_real,Cv_b_real,Co_a_real,Co_a_imag],compact=False)
            G_abji_imag -= ao2mo.general(G_AO,[Cv_b_real,Cv_b_real,Co_a_imag,Co_a_real],compact=False)
            G_abji_imag += ao2mo.general(G_AO,[Cv_b_real,Cv_b_imag,Co_a_real,Co_a_real],compact=False)
            G_abji_imag += ao2mo.general(G_AO,[Cv_b_real,Cv_b_imag,Co_a_imag,Co_a_imag],compact=False)
            G_abji_imag -= ao2mo.general(G_AO,[Cv_b_imag,Cv_b_real,Co_a_real,Co_a_real],compact=False)
            G_abji_imag -= ao2mo.general(G_AO,[Cv_b_imag,Cv_b_real,Co_a_imag,Co_a_imag],compact=False)
            G_abji_imag += ao2mo.general(G_AO,[Cv_b_imag,Cv_b_imag,Co_a_real,Co_a_imag],compact=False)
            G_abji_imag -= ao2mo.general(G_AO,[Cv_b_imag,Cv_b_imag,Co_a_imag,Co_a_real],compact=False)
            print("    (AB|ji) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

            G_abji_real += ao2mo.general(G_AO,[Cv_b_real,Cv_b_real,Co_b_real,Co_b_real],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_b_real,Cv_b_real,Co_b_imag,Co_b_imag],compact=False)
            G_abji_real -= ao2mo.general(G_AO,[Cv_b_real,Cv_b_imag,Co_b_real,Co_b_imag],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_b_imag,Cv_b_real,Co_b_real,Co_b_imag],compact=False)
            G_abji_real -= ao2mo.general(G_AO,[Cv_b_imag,Cv_b_real,Co_b_imag,Co_b_real],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_b_imag,Cv_b_imag,Co_b_real,Co_b_real],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_b_real,Cv_b_imag,Co_b_imag,Co_b_real],compact=False)
            G_abji_real += ao2mo.general(G_AO,[Cv_b_imag,Cv_b_imag,Co_b_imag,Co_b_imag],compact=False)

            G_abji_imag += ao2mo.general(G_AO,[Cv_b_real,Cv_b_real,Co_b_real,Co_b_imag],compact=False)
            G_abji_imag -= ao2mo.general(G_AO,[Cv_b_real,Cv_b_real,Co_b_imag,Co_b_real],compact=False)
            G_abji_imag += ao2mo.general(G_AO,[Cv_b_real,Cv_b_imag,Co_b_real,Co_b_real],compact=False)
            G_abji_imag += ao2mo.general(G_AO,[Cv_b_real,Cv_b_imag,Co_b_imag,Co_b_imag],compact=False)
            G_abji_imag -= ao2mo.general(G_AO,[Cv_b_imag,Cv_b_real,Co_b_real,Co_b_real],compact=False)
            G_abji_imag -= ao2mo.general(G_AO,[Cv_b_imag,Cv_b_real,Co_b_imag,Co_b_imag],compact=False)
            G_abji_imag += ao2mo.general(G_AO,[Cv_b_imag,Cv_b_imag,Co_b_real,Co_b_imag],compact=False)
            G_abji_imag -= ao2mo.general(G_AO,[Cv_b_imag,Cv_b_imag,Co_b_imag,Co_b_real],compact=False)
            print("    (AB|JI) contributionn done in %5.2f seconds."%(time.time()-tic),flush=True)

            G_abji = G_abji_real + 1j*G_abji_imag

            toc = time.time()
            print("    AO to MO transformation time: %5.2f seconds"%(toc -tic),flush=True)
          
            nv = 2*self.wfn.nbf - nel
            G_aibj = G_aijb.reshape(nv,nel,nel,nv).swapaxes(2,3)
            G_abji = G_abji.reshape(nv,nv,nel,nel)
          
            if self.options.xctype == "HF":
              self.wfn.G_array = [G_aibj, G_abji]
            else:
              self.wfn.G_array = [G_aibj, G_abji*self.options.xcalpha]
            cis = davidson.GDAVIDSON(self.wfn)
            cis.cvs = self.options.cvs
            w, X = cis.compute()
            self.tda_analyze_generalized(w,X,cis)

          elif self.wfn.reference == "rhf":
            tic = time.time()
 
            nel_a = self.wfn.nel[0]
            Co_a  = self.wfn.C[0][:,:nel_a]
            Cv_a  = self.wfn.C[0][:,nel_a:]
  
            G_aibj  = ao2mo.general(self.ints,[Cv_a,Co_a,Cv_a,Co_a],compact=False)
            print("    (ai|bj) contributionn done.",flush=True)
  
            G_abji  = ao2mo.general(self.ints,[Cv_a,Cv_a,Co_a,Co_a],compact=False)
            print("    (ab|ji) contributionn done.",flush=True)
  
            toc = time.time()
            print("    AO to MO transformation time: %5.2f seconds"%(toc -tic),flush=True)
            
            nv_a = self.wfn.nbf - nel_a
            G_aibj = G_aibj.reshape(nv_a,nel_a,nv_a,nel_a)
            G_abji = G_abji.reshape(nv_a,nv_a,nel_a,nel_a)

            self.wfn.G_array = [2.*G_aibj, G_abji]
            singlet = davidson.RDAVIDSON(self.wfn)
            singlet.cvs = self.options.cvs
            w_s, X_s = singlet.compute()

            self.wfn.G_array = [0.*G_aibj, G_abji]
            triplet = davidson.RDAVIDSON(self.wfn)
            triplet.cvs = self.options.cvs
            w_t, X_t = triplet.compute()
            self.tda_analyze_restricted(w_s,np.sqrt(2.)*X_s,singlet,w_t,np.sqrt(2.)*X_t,triplet)
          else:
            exit("    ERROR: Reference %s not implemented."%(self.wfn.reference))  

      elif self.options.direct_diagonalization is True: 
          print("\n    Direct diagonalization algorithm requested. This algorithm should only be used for relatively small systems",flush=True)
         
          if (self.wfn.reference == "uks"):
            if self.options.spin_flip is True:
              print("    Spin-Flip Algorithm enabled",flush=True)
              sol = direct_diag.SFDIRECT_DIAG(self.wfn)
            else:
              sol = direct_diag.UDIRECT_DIAG(self.wfn)
            sol.cvs = self.options.cvs
            w, X = sol.compute()
            self.tda_analyze_unrestricted(w,X,sol)
          elif (self.wfn.reference == "rks"):
            s1 = direct_diag.RDIRECT_DIAG(self.wfn)
            s1.cvs = self.options.cvs
            #self.options.occupied = self.options.occupied1 
            w1, X1 = s1.compute()

            if self.options.do_triplet is True:
              self.wfn.triplet = True
              s2 = direct_diag.RDIRECT_DIAG(self.wfn)
              s2.cvs = self.options.cvs
              #self.options.occupied = self.options.occupied2 
              w2, X2 = s2.compute()
              self.wfn.triplet = False
              self.tda_analyze_restricted(w1,np.sqrt(2.)*X1,s1,w2,np.sqrt(2.)*X2,s2)

            else:
              self.tda_analyze_restricted(w1,np.sqrt(2.)*X1,s1,w1,np.sqrt(2.)*X1,s1)

          elif (self.wfn.reference == "rgks"):
            if self.options.couple_states is True:
              s1 = direct_diag.RGDIRECT_DIAG(self.wfn)
              s1.cvs = True
              self.options.occupied = self.options.occupied1 
              w1, X1 = s1.compute()
              s1.Cmo = self.wfn.C[0]
              s1.no_a = s1.no_a_full
              self.tda_analyze_generalized(w1,X1,s1)

              s2 = direct_diag.RGDIRECT_DIAG(self.wfn)
              s2.cvs = True
              self.options.occupied = self.options.occupied2 
              w2, X2 = s2.compute()
              self.tda_analyze_generalized(w2,X2,s2)
              self.tda_couple_generalized(w1,w2,X1,X2,s1,s2)
            else:
              sol = direct_diag.RGDIRECT_DIAG(self.wfn)
              sol.cvs = self.options.cvs
              w, X = sol.compute()
              self.tda_analyze_generalized(w,X,sol)
          elif (self.wfn.reference == "gks"):
            sol = direct_diag.GDIRECT_DIAG(self.wfn)
            sol.cvs = self.options.cvs
            w, X = sol.compute()
            self.tda_analyze_generalized(w,X,sol)


 
      else: #don't use in-core algorithm (do not store two-electron integrals)
          if (self.wfn.reference == "rks"):
            singlet = davidson.RDAVIDSON(self.wfn)
            singlet.cvs = self.options.cvs
            singlet.reduced_virtual = self.options.reduced_virtual
            w_s, X_s = singlet.compute()

            self.wfn.triplet = True
            triplet = davidson.RDAVIDSON(self.wfn)
            triplet.cvs = self.options.cvs
            triplet.reduced_virtual = self.options.reduced_virtual
            w_t, X_t = triplet.compute()
            self.wfn.triplet = False
            self.tda_analyze_restricted(w_s,np.sqrt(2.)*X_s,singlet,w_t,np.sqrt(2.)*X_t,triplet)
          elif (self.wfn.reference == "rhf"):
            sol = davidson.RDAVIDSON(self.wfn)
            sol.cvs = self.options.cvs
            w, X = sol.compute()
          elif (self.wfn.reference == "uhf"):
            sol = davidson.UDAVIDSON(self.wfn)
            sol.cvs = self.options.cvs
            w, X = sol.compute()
         # elif (self.wfn.reference == "rks"):
         #   sol = davidson.RDAVIDSON(self.wfn)
         #   sol.cvs = self.options.cvs
         #   w, X = sol.compute()
          elif (self.wfn.reference == "uks"):
            sol = davidson.UDAVIDSON(self.wfn)
            sol.cvs = self.options.cvs
            w, X = sol.compute()
            self.tda_analyze_unrestricted(w,X,sol)
          elif (self.wfn.reference == "rgks"):
            if self.options.couple_states is True:
              s1 = davidson.RGDAVIDSON(self.wfn)
              s1.cvs = True
              self.options.occupied = self.options.occupied1 
              w1, X1 = s1.compute()
              s1.Cmo = self.wfn.C[0]
              s1.no_a = s1.no_a_full
              self.tda_analyze_generalized(w1,X1,s1)

              s2 = davidson.RGDAVIDSON(self.wfn)
              s2.cvs = True
              self.options.occupied = self.options.occupied2 
              w2, X2 = s2.compute()
              self.tda_analyze_generalized(w2,X2,s2)
              self.tda_couple_generalized(w1,w2,X1,X2,s1,s2)
            else:
              sol = davidson.RGDAVIDSON(self.wfn)
              sol.cvs = self.options.cvs
              w, X = sol.compute()
              self.tda_analyze_generalized(w,X,sol)
          elif (self.wfn.reference == "gks"):
            sol = davidson.GDAVIDSON(self.wfn)
            sol.cvs = self.options.cvs
            w, X = sol.compute()
            self.tda_analyze_generalized(w,X,sol)
        
    def XhX(self,Xl,h,Xr,na_virt,na_docc,nroots,phase):
        H = np.zeros((nroots,nroots),dtype=complex)
        X1 = Xr.T
        X2 = Xl.T
        for n in range(nroots):
          Xn   = X1[n].reshape(na_virt,na_docc) #X_ai
          hXr  = np.matmul(h[na_docc:,na_docc:],Xn) #h_ab * X_bi
          hXr -= phase*np.matmul(Xn,h[:na_docc,:na_docc]) #X_aj * h_ji
          for m in range(nroots):
            Xm = X2[m].reshape(na_virt,na_docc) #X_ai
            H[m][n] = np.trace(np.matmul(Xm.T,hXr)) #X_ai * X_ai
        return H

    def lorentzian(self,wrange,dump,roots,os):
        nw = int((float(wrange[1]) - float(wrange[0]))/float(wrange[2]))
        w = np.zeros(nw)
        S = np.zeros(nw)
        for windex in range(nw):
            w[windex] = (float(wrange[0]) + float(wrange[2]) * windex)
            for root in range(len(roots)):
                S[windex] += os[root]*dump/((w[windex]-roots[root])**2 + dump**2)
            S[windex] /= (1.0 * np.pi)
        return w, S

    def get_pair_map(self,no,nv):
      pair_map = np.zeros((nv*no,2),dtype=int)
      for i in range(no):
        for a in range(nv):
          pair_map[a*no+i][0] = a+no+1
          pair_map[a*no+i][1] = i+1
      return pair_map   

    def tda_analyze_restricted(self,w_s,X_s,singlet,w_t,X_t,triplet):

      if (self.wfn.reference != "uks" and self.wfn.reference != "uhf"):
      
          mux = np.matmul(np.conj(singlet.Cmo.T),np.matmul(self.wfn.mu[0],singlet.Cmo))
          muy = np.matmul(np.conj(singlet.Cmo.T),np.matmul(self.wfn.mu[1],singlet.Cmo))
          muz = np.matmul(np.conj(singlet.Cmo.T),np.matmul(self.wfn.mu[2],singlet.Cmo))
    
          dx = np.einsum("p,pN->N",mux[singlet.no_a:,:singlet.no_a].reshape(singlet.nv_a*singlet.no_a),np.conj(X_s),optimize=True)
          dy = np.einsum("p,pN->N",muy[singlet.no_a:,:singlet.no_a].reshape(singlet.nv_a*singlet.no_a),np.conj(X_s),optimize=True)
          dz = np.einsum("p,pN->N",muz[singlet.no_a:,:singlet.no_a].reshape(singlet.nv_a*singlet.no_a),np.conj(X_s),optimize=True)
          f  = 2. * w_s * (np.abs(dx)**2 + np.abs(dy)**2 + np.abs(dz)**2) /3.
    
          print(" ")
          print("            Spectral Analysis ")
          print("    ----------------------------------- ")
          print(" ")
          print("    Root     E(eV)       <0|x|n>        <0|y|n>        <0|z|n>          f",flush=True)
          for n in range(len(w_s)):
            print("   %5i %10.4f %14.10f %14.10f %14.10f %14.10f"%(n+1,w_s[n]*27.21138,dx[n].real,dy[n].real,dz[n].real,f[n]),flush=True)

          print(" ")
          print("       CI Vectors Contributions (S=0) ")
          print("    ----------------------------------- ")
          print(" ")
          
          pair_map = self.get_pair_map(singlet.no_a,singlet.nv_a)
          
          for n in range(singlet.nroots):
            Xn = X_s.T[n]/np.sqrt(2.)
            print("    Root %i, E(eV): %f, f: %f "%(n+1,w_s[n]*27.21138,f[n]),flush=True)
            for ai in range(singlet.nov):
              if (Xn[ai]**2 > 0.05):
                a = pair_map[ai][0]
                i = pair_map[ai][1]
                print("        Occ. %i alpha --- Virt. %i alpha: %f"%(i,a,Xn[ai]**2),flush=True)
            print("")

          print(" ")
          print("       CI Vectors Contributions (S=1) ")
          print("    ----------------------------------- ")
          print(" ")

          for n in range(triplet.nroots):
            Xn = X_t.T[n]/np.sqrt(2.)
            print("    Root %i, E(eV): %f, f: %f "%(n+1,w_t[n]*27.21138,0.),flush=True)
            for ai in range(triplet.nov):
              if (Xn[ai]**2 > 0.05):
                a = pair_map[ai][0]
                i = pair_map[ai][1]
                print("        Occ. %i alpha --- Virt. %i alpha: %f"%(i,a,Xn[ai]**2),flush=True)
            print("")

          if self.options.fourier_transform is True:
            if self.options.frequencies is not None:
              ifreq, ffreq, dfreq, gamma = self.options.frequencies
            else:
              ifreq = w_s[0]*27.21138 - 10.
              ffreq = w_s[-1]*27.21138 + 10.
              dfreq = 0.01
              gamma = self.options.gamma
            frange = [ifreq, ffreq, dfreq]
            #w, S_sr      = lorentzian(frange,gamma,w_sr[:nroots]*27.21138,f_sr)
            E, S      = self.lorentzian(frange,gamma,w_s*27.21138,f)               
            sp_filename  = self.options.inputfile.split(".")[0]+".spec"
            sp_file = open(sp_filename,"w")
            sp_file.write("# E[eV] f \n")
            for n in range(len(E)):
                sp_file.write("%12.8f %12.8f \n"%(E[n],S[n]))
            sp_file.close()

          if (self.options.relativistic == "zora") and (self.wfn.reference == "rhf" or self.wfn.reference == "rks"):
            nroots = singlet.nroots
            
            h_z_ao =  self.wfn.H_so[:self.wfn.nbf,:self.wfn.nbf]
            h_m_ao =  self.wfn.H_so[:self.wfn.nbf,self.wfn.nbf:]
            h_p_ao =  self.wfn.H_so[self.wfn.nbf:,:self.wfn.nbf]
 
            h_z = np.matmul(singlet.Cmo.T,np.matmul(h_z_ao,singlet.Cmo))
            h_m = np.matmul(singlet.Cmo.T,np.matmul(h_m_ao,singlet.Cmo))
            h_p = np.matmul(singlet.Cmo.T,np.matmul(h_p_ao,singlet.Cmo))
 
            na_virt = singlet.nv_a
            na_docc = singlet.no_a
 
            #Add diagonal energies to the Hamiltonian
            H_so = np.zeros((4*nroots,4*nroots),dtype=complex)
            w_sr = np.zeros((4*nroots))
            X_s /= np.sqrt(2.)
            X_t /= np.sqrt(2.)
            for n in range(nroots):
              H_so[0*nroots+n][0*nroots+n] = w_s[n]
              H_so[1*nroots+n][1*nroots+n] = w_t[n]
              H_so[2*nroots+n][2*nroots+n] = w_t[n]
              H_so[3*nroots+n][3*nroots+n] = w_t[n]
 
              w_sr[0*nroots+n] = w_s[n]
              w_sr[1*nroots+n] = w_t[n]
              w_sr[2*nroots+n] = w_t[n]
              w_sr[3*nroots+n] = w_t[n]
 
            #Add off-diagonal blocks to the Hamiltonian
 
            fac = 1. #1./np.sqrt(3.) #C-G coefficient 
            H_so[0*nroots:1*nroots,1*nroots:2*nroots] += np.sqrt(2.) * fac*self.XhX(X_s,h_p,X_t,na_virt,na_docc,nroots,1.)/2.
            H_so[0*nroots:1*nroots,2*nroots:3*nroots] += 1           * fac*self.XhX(X_s,h_z,X_t,na_virt,na_docc,nroots,1.)
            H_so[0*nroots:1*nroots,3*nroots:4*nroots] -= np.sqrt(2.) * fac*self.XhX(X_s,h_m,X_t,na_virt,na_docc,nroots,1.)/2.
 
            fac = 1. #C-G coefficient 
            H_so[1*nroots:2*nroots,0*nroots:1*nroots] += np.sqrt(2.) * fac*self.XhX(X_t,h_m,X_s,na_virt,na_docc,nroots,1.)/2.
            H_so[2*nroots:3*nroots,0*nroots:1*nroots] += 1.          * fac*self.XhX(X_t,h_z,X_s,na_virt,na_docc,nroots,1.)
            H_so[3*nroots:4*nroots,0*nroots:1*nroots] -= np.sqrt(2.) * fac*self.XhX(X_t,h_p,X_s,na_virt,na_docc,nroots,1.)/2.
 
            fac = 1. #/np.sqrt(2.) #C-G coefficient
            H_so[1*nroots:2*nroots,1*nroots:2*nroots] += 1           * fac*self.XhX(X_t,h_z,X_t,na_virt,na_docc,nroots,-1.)
            H_so[1*nroots:2*nroots,2*nroots:3*nroots] -= np.sqrt(2.) * fac*self.XhX(X_t,h_m,X_t,na_virt,na_docc,nroots,-1.)/2.
 
            H_so[2*nroots:3*nroots,1*nroots:2*nroots] -= np.sqrt(2.) * fac*self.XhX(X_t,h_p,X_t,na_virt,na_docc,nroots,-1.)/2.
            H_so[2*nroots:3*nroots,3*nroots:4*nroots] -= np.sqrt(2.) * fac*self.XhX(X_t,h_m,X_t,na_virt,na_docc,nroots,-1.)/2.
 
            H_so[3*nroots:4*nroots,2*nroots:3*nroots] -= np.sqrt(2.) * fac*self.XhX(X_t,h_p,X_t,na_virt,na_docc,nroots,-1.)/2.
            H_so[3*nroots:4*nroots,3*nroots:4*nroots] -= 1           * fac*self.XhX(X_t,h_z,X_t,na_virt,na_docc,nroots,-1.)

            w_so, X_so = np.linalg.eigh(H_so)
            dx_so = np.matmul(dx,X_so[:nroots,:])
            dy_so = np.matmul(dy,X_so[:nroots,:])
            dz_so = np.matmul(dz,X_so[:nroots,:])
 
            f_so  = 2.*w_so * (np.abs(dx_so)**2 + np.abs(dy_so)**2 + np.abs(dz_so)**2) /3.
 
            print(" ")
            print("    Spin-Orbit Coupled Spectra Analysis ")
            print("    ----------------------------------- ")
            print(" ")
            for n in range(len(w_sr)):
              print("   %5i %8.5f %14.10f %14.10f %14.10f %14.10f %14.10f %14.10f %14.10f "%(n+1,w_so[n]*27.21138,dx_so[n].real,dx_so[n].imag,dy_so[n].real,dy_so[n].imag,dz_so[n].real,dz_so[n].imag,f_so[n]))
            if self.options.fourier_transform is True:
              if self.options.frequencies is not None:
                ifreq, ffreq, dfreq, gamma = self.options.frequencies
              else:
                ifreq = w_so[0]*27.21138 - 10.
                ffreq = w_so[-1]*27.21138 + 10.
                dfreq = 0.01
                gamma = self.options.gamma
              frange = [ifreq, ffreq, dfreq]
              #w, S_sr      = lorentzian(frange,gamma,w_sr[:nroots]*27.21138,f_sr)
              E, S      = self.lorentzian(frange,gamma,w_so*27.21138,f_so)               
              sp_filename  = self.options.inputfile.split(".")[0]+".spec_so"
              sp_file = open(sp_filename,"w")
              sp_file.write("# E[eV] f \n")
              for n in range(len(E)):
                  sp_file.write("%12.8f %12.8f \n"%(E[n],S[n]))
              sp_file.close()

            print(" ")
            print("        SO-CI Vectors Contributions     ")
            print("    ----------------------------------- ")
            print(" ")

            for n in range(len(w_sr)):
              Xn = X_so.T[n]
              print("    SO Vector %i, E(eV): %f, f: %f "%(n+1,w_so[n]*27.21138,f_so[n]),flush=True)
              for ai in range(4*nroots):
                if (np.abs(Xn[ai])**2 > 0.05):
                  if ai < nroots:
                    print("        Singlet (0, 0) Root  %i: %f"%(ai+1,np.abs(Xn[ai])**2),flush=True)
                  elif (ai > nroots-1 and ai < 2*nroots):
                    print("        Triplet (1,-1) Root  %i: %f"%(ai-nroots+1,np.abs(Xn[ai])**2),flush=True)
                  elif (ai > 2*nroots-1 and ai < 3*nroots):
                    print("        Triplet (1, 0) Root  %i: %f"%(ai-2*nroots+1,np.abs(Xn[ai])**2),flush=True)
                  elif (ai > 3*nroots-1 and ai < 4*nroots):
                    print("        Triplet (1,+1) Root  %i: %f"%(ai-3*nroots+1,np.abs(Xn[ai])**2),flush=True)
              print("")
    #  else:
    def tda_analyze_unrestricted(self,w,X,cis):
             
       mux_a = np.matmul(np.conj(cis.Cmo[0].T),np.matmul(self.wfn.mu[0],cis.Cmo[0]))
       muy_a = np.matmul(np.conj(cis.Cmo[0].T),np.matmul(self.wfn.mu[1],cis.Cmo[0]))
       muz_a = np.matmul(np.conj(cis.Cmo[0].T),np.matmul(self.wfn.mu[2],cis.Cmo[0]))

       mux_b = np.matmul(np.conj(cis.Cmo[1].T),np.matmul(self.wfn.mu[0],cis.Cmo[1]))
       muy_b = np.matmul(np.conj(cis.Cmo[1].T),np.matmul(self.wfn.mu[1],cis.Cmo[1]))
       muz_b = np.matmul(np.conj(cis.Cmo[1].T),np.matmul(self.wfn.mu[2],cis.Cmo[1]))
    
       dx  = np.einsum("p,pN->N",mux_a[cis.no_a:,:cis.no_a].reshape(cis.nv_a*cis.no_a),np.conj(X[:cis.nv_a*cis.no_a,:]),optimize=True)
       dy  = np.einsum("p,pN->N",muy_a[cis.no_a:,:cis.no_a].reshape(cis.nv_a*cis.no_a),np.conj(X[:cis.nv_a*cis.no_a,:]),optimize=True)
       dz  = np.einsum("p,pN->N",muz_a[cis.no_a:,:cis.no_a].reshape(cis.nv_a*cis.no_a),np.conj(X[:cis.nv_a*cis.no_a,:]),optimize=True)
       dx += np.einsum("p,pN->N",mux_b[cis.no_b:,:cis.no_b].reshape(cis.nv_b*cis.no_b),np.conj(X[cis.nv_a*cis.no_a:,:]),optimize=True)
       dy += np.einsum("p,pN->N",muy_b[cis.no_b:,:cis.no_b].reshape(cis.nv_b*cis.no_b),np.conj(X[cis.nv_a*cis.no_a:,:]),optimize=True)
       dz += np.einsum("p,pN->N",muz_b[cis.no_b:,:cis.no_b].reshape(cis.nv_b*cis.no_b),np.conj(X[cis.nv_a*cis.no_a:,:]),optimize=True)
       f  = 2. * w * (np.abs(dx)**2 + np.abs(dy)**2 + np.abs(dz)**2) /3.
    
       print(" ")
       print("            Spectral Analysis ")
       print("    ----------------------------------- ")
       print(" ")
       print("    Root     E(eV)       <0|x|n>        <0|y|n>        <0|z|n>          f",flush=True)
       for n in range(len(w)):
         print("   %5i %10.4f %14.10f %14.10f %14.10f %14.10f"%(n+1,w[n]*27.21138,dx[n].real,dy[n].real,dz[n].real,f[n]),flush=True)

       print(" ")
       print("       CI Vectors Contributions ")
       print("    ----------------------------------- ")
       print(" ")

       pair_map_a = self.get_pair_map(cis.no_a,cis.nv_a)
       pair_map_b = self.get_pair_map(cis.no_b,cis.nv_b)

       for n in range(cis.nroots):
         Xn_a = X.T[n][:cis.nov_a]
         Xn_b = X.T[n][cis.nov_a:]
         print("    Root %i, E(eV): %f, f: %f "%(n+1,w[n]*27.21138,f[n]),flush=True)
         for ai in range(cis.nov_a):
           if (Xn_a[ai]**2 > 0.05):
             a = pair_map_a[ai][0]
             i = pair_map_a[ai][1]
             print("        Occ. %i alpha --- Virt. %i alpha: %f"%(i,a,np.abs(Xn_a[ai])**2),flush=True)
         for ai in range(cis.nov_b):
           if (Xn_b[ai]**2 > 0.05):
             a = pair_map_b[ai][0]
             i = pair_map_b[ai][1]
             print("        Occ. %i beta  --- Virt. %i beta : %f"%(i,a,np.abs(Xn_b[ai])**2),flush=True)
         print("")

       if self.options.fourier_transform is True:
         if self.options.frequencies is not None:
           ifreq, ffreq, dfreq, gamma = self.options.frequencies
           print("    Fourier Transform parameters:",flush=True)
           print("    ifreq = %8.5f eV, ffreq = %8.5f eV, dfreq = %8.5f eV, gamma = %8.5f eV"%(ifreq,ffreq,dfreq,gamma),flush=True)
         else:
           ifreq = w[0]*27.21138 - 10.
           ffreq = w[-1]*27.21138 + 10.
           dfreq = 0.01
           gamma = self.options.gamma
         frange = [ifreq, ffreq, dfreq]
         #w, S_sr      = lorentzian(frange,gamma,w_sr[:nroots]*27.21138,f_sr)
         E, S      = self.lorentzian(frange,gamma,w*27.21138,f)               
         sp_filename  = self.options.inputfile.split(".")[0]+".spec"
         sp_file = open(sp_filename,"w")
         sp_file.write("# E[eV] f \n")
         for n in range(len(E)):
             sp_file.write("%12.8f %12.8f \n"%(E[n],S[n]))
         sp_file.close()

    def tda_analyze_generalized(self,w,X,cis):

          mux = np.matmul(np.conj(cis.Cmo.T),np.matmul(self.wfn.mu[0],cis.Cmo))
          muy = np.matmul(np.conj(cis.Cmo.T),np.matmul(self.wfn.mu[1],cis.Cmo))
          muz = np.matmul(np.conj(cis.Cmo.T),np.matmul(self.wfn.mu[2],cis.Cmo))
          print(cis.Cmo.shape)
    
          dx = np.einsum("p,pN->N",mux[cis.no_a:,:cis.no_a].reshape(cis.nv_a*cis.no_a),np.conj(X),optimize=True)
          dy = np.einsum("p,pN->N",muy[cis.no_a:,:cis.no_a].reshape(cis.nv_a*cis.no_a),np.conj(X),optimize=True)
          dz = np.einsum("p,pN->N",muz[cis.no_a:,:cis.no_a].reshape(cis.nv_a*cis.no_a),np.conj(X),optimize=True)
          f  = 2. * w * (np.abs(dx)**2 + np.abs(dy)**2 + np.abs(dz)**2) /3.
    
          print(" ")
          print("            Spectral Analysis ")
          print("    ----------------------------------- ")
          print(" ")
          print("    Root     E(eV)       <0|x|n>        <0|y|n>        <0|z|n>          f",flush=True)
          for n in range(len(w)):
            print("   %5i %10.4f %14.10f %14.10f %14.10f %14.10f"%(n+1,w[n]*27.21138,dx[n].real,dy[n].real,dz[n].real,f[n]),flush=True)

          print(" ")
          print("          CI Vectors Contributions      ")
          print("    ----------------------------------- ")
          print(" ")

          pair_map = self.get_pair_map(cis.no_a,cis.nv_a)
          for n in range(cis.nroots):
            Xn = X.T[n]
            print("    Root %i, E(eV): %f, f: %f "%(n+1,w[n]*27.21138,f[n]),flush=True)
            for ai in range(cis.nov):
              if (np.abs(Xn[ai])**2 > 0.05):
                a = pair_map[ai][0]
                i = pair_map[ai][1]
                #print("        Occ. %i --- Virt. %i : %f"%(i,a,np.abs(Xn[ai])**2),flush=True)
                print("        Occ. %i --- Virt. %i : %f"%(i,a,Xn[ai].real),flush=True)
            print("")


          if self.options.fourier_transform is True:
            if self.options.frequencies is not None:
              ifreq, ffreq, dfreq, gamma = self.options.frequencies
            else:
              ifreq = w[0]*27.21138 - 10.
              ffreq = w[-1]*27.21138 + 10.
              dfreq = 0.01
              gamma = self.options.gamma
            frange = [ifreq, ffreq, dfreq]
            #w, S_sr      = lorentzian(frange,gamma,w_sr[:nroots]*27.21138,f_sr)
            E, S      = self.lorentzian(frange,gamma,w*27.21138,f)               
            if self.options.so_scf is True:
              sp_filename  = self.options.inputfile.split(".")[0]+".spec_so"
            else:
              sp_filename  = self.options.inputfile.split(".")[0]+".spec"
            sp_file = open(sp_filename,"w")
            sp_file.write("# E[eV] f \n")
            for n in range(len(E)):
                sp_file.write("%12.8f %12.8f \n"%(E[n],S[n]))
            sp_file.close()

          if (self.options.relativistic == "zora" and self.options.so_scf is not True):
            nroots = cis.nroots
 
            H_so_mo = np.matmul(np.conj(cis.Cmo.T),np.matmul(self.wfn.H_so,cis.Cmo))
            #print(H_so_mo.shape)
            #print(X.shape)
            #exit(0)
 
            #Add diagonal energies to the Hamiltonian
            H_so = np.zeros((nroots,nroots),dtype=complex)
            for n in range(nroots):
              H_so[n][n] = w[n]
            #Add off-diagonal blocks to the Hamiltonian
            H_so += np.einsum("mai,ab,nbi->mn",np.conj(X.T).reshape(nroots,cis.nv_a,cis.no_a),H_so_mo[cis.no_a:,cis.no_a:],(X.T).reshape(nroots,cis.nv_a,cis.no_a),optimize=True)
            H_so -= np.einsum("mai,ji,naj->mn",np.conj(X.T).reshape(nroots,cis.nv_a,cis.no_a),H_so_mo[:cis.no_a,:cis.no_a],(X.T).reshape(nroots,cis.nv_a,cis.no_a),optimize=True)

            w_so, X_so = np.linalg.eigh(H_so)
            dx_so = np.matmul(dx,np.conj(X_so[:nroots,:]))
            dy_so = np.matmul(dy,np.conj(X_so[:nroots,:]))
            dz_so = np.matmul(dz,np.conj(X_so[:nroots,:]))
 
            f_so  = 2.*w_so * (np.abs(dx_so)**2 + np.abs(dy_so)**2 + np.abs(dz_so)**2) /3.
 
            print(" ")
            print("    Spin-Orbit Coupled Spectra Analysis ")
            print("    ----------------------------------- ")
            print(" ")
            for n in range(len(w)):
              print("   %5i %8.5f %14.10f %14.10f %14.10f %14.10f %14.10f %14.10f %14.10f "%(n+1,w_so[n]*27.21138,dx_so[n].real,dx_so[n].imag,dy_so[n].real,dy_so[n].imag,dz_so[n].real,dz_so[n].imag,f_so[n]))
 
            if self.options.fourier_transform is True:
              if self.options.frequencies is not None:
                ifreq, ffreq, dfreq, gamma = self.options.frequencies
              else:
                ifreq = w_so[0]*27.21138 - 10.
                ffreq = w_so[-1]*27.21138 + 10.
                dfreq = 0.01
                gamma = self.options.gamma
              frange = [ifreq, ffreq, dfreq]
              #w, S_sr      = lorentzian(frange,gamma,w_sr[:nroots]*27.21138,f_sr)
              w, S_so      = self.lorentzian(frange,gamma,w_so*27.21138,f_so)               
              so_filename  = self.options.inputfile.split(".")[0]+".spec_so"
              so_file = open(so_filename,"w")
              so_file.write("# E[eV] f \n")
              for n in range(len(w)):
                  so_file.write("%12.8f %12.8f \n"%(w[n],S_so[n]))
              so_file.close()

            print(" ")
            print("        SO-CI Vectors Contributions     ")
            print("    ----------------------------------- ")
            print(" ")

            for n in range(len(w_so)):
              Xn = X_so.T[n]
              print("    SO Vector %i, E(eV): %f, f: %f "%(n+1,w_so[n]*27.21138,f_so[n]),flush=True)
              for ai in range(nroots):
                if (np.abs(Xn[ai])**2 > 0.05):
                  print("        Uncoupled Root  %i: %f"%(ai+1,np.abs(Xn[ai])**2),flush=True)
              print("")

    def tda_couple_generalized(self,w1,w2,X1,X2,cis1,cis2):
          no_a = cis1.no_a_full
          nv_a = cis1.nv_a
          #nov  = cis1.nov_a_full
          Cmo = self.wfn.C[0]
          Cv  = Cmo[:,no_a:]
          Co  = Cmo[:,:no_a]

          mux = np.matmul(np.conj(Cmo.T),np.matmul(self.wfn.mu[0],Cmo))
          muy = np.matmul(np.conj(Cmo.T),np.matmul(self.wfn.mu[1],Cmo))
          muz = np.matmul(np.conj(Cmo.T),np.matmul(self.wfn.mu[2],Cmo))
          print(no_a *nv_a, X1.shape,mux[no_a:,:no_a].reshape(nv_a*no_a).shape)
    
          dx1 = np.einsum("p,pN->N",mux[no_a:,:no_a].reshape(nv_a*no_a),np.conj(X1),optimize=True)
          dy1 = np.einsum("p,pN->N",muy[no_a:,:no_a].reshape(nv_a*no_a),np.conj(X1),optimize=True)
          dz1 = np.einsum("p,pN->N",muz[no_a:,:no_a].reshape(nv_a*no_a),np.conj(X1),optimize=True)
          f1  = 2. * w1 * (np.abs(dx1)**2 + np.abs(dy1)**2 + np.abs(dz1)**2) /3.

          fp = open(str(self.options.inputfile.split(".")[0])+".couplings_noso","w")
          for n in range(len(w1)):
            fp.write(" 0 %i %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f \n"%(n+1,w1[n],\
            dx1[n].real,dx1[n].imag,dy1[n].real,dy1[n].imag,dz1[n].real,dz1[n].imag))

          X1 = (X1.T).reshape(cis1.nroots,nv_a,no_a)
          X2 = (X2.T).reshape(cis2.nroots,nv_a,no_a)
          dx12  = np.einsum("Nai,ab,Mbi->NM",np.conj(X1),mux[no_a:,no_a:],X2,optimize=True)
          dx12 -= np.einsum("Naj,ji,Mai->NM",np.conj(X1),mux[:no_a,:no_a],X2,optimize=True)
          dy12  = np.einsum("Nai,ab,Mbi->NM",np.conj(X1),muy[no_a:,no_a:],X2,optimize=True)
          dy12 -= np.einsum("Naj,ji,Mai->NM",np.conj(X1),muy[:no_a,:no_a],X2,optimize=True)
          dz12  = np.einsum("Nai,ab,Mbi->NM",np.conj(X1),muz[no_a:,no_a:],X2,optimize=True)
          dz12 -= np.einsum("Naj,ji,Mai->NM",np.conj(X1),muz[:no_a,:no_a],X2,optimize=True)

          for n in range(cis1.nroots):
            for f in range(cis2.nroots):
              fp.write(" %i %i %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f \n"%(n+1,f+1,-w1[n]+w2[f],\
                   dx12[n][f].real,dx12[n][f].imag,dy12[n][f].real,dy12[n][f].imag,dz12[n][f].real,dz12[n][f].imag))
        
          fp.close() 
           
          if (self.options.relativistic == "zora" and self.options.so_scf is not True):
            H_so_mo = np.matmul(np.conj(Cmo.T),np.matmul(self.wfn.H_so,Cmo))

            #compute couplings for initial states  
            H_so1 = np.zeros((cis1.nroots,cis1.nroots),dtype=complex)
            for n in range(cis1.nroots):
              H_so1[n][n] = w1[n]
            #Add off-diagonal blocks to the Hamiltonian
            H_so1 += np.einsum("mai,ab,nbi->mn",np.conj(X1),H_so_mo[no_a:,no_a:],X1,optimize=True)
            H_so1 -= np.einsum("mai,ji,naj->mn",np.conj(X1),H_so_mo[:no_a,:no_a],X1,optimize=True)

            w_n, X_n = np.linalg.eigh(H_so1)

            dx_gn = np.matmul(dx1,X_n[:cis1.nroots,:])
            dy_gn = np.matmul(dy1,X_n[:cis1.nroots,:])
            dz_gn = np.matmul(dz1,X_n[:cis1.nroots,:])
 
            fp = open(str(self.options.inputfile.split(".")[0])+".couplings_so","w")
            for n in range(len(w_n)):
              fp.write(" 0 %i %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f \n"%(n+1,w_n[n],\
              dx_gn[n].real,dx_gn[n].imag,dy_gn[n].real,dy_gn[n].imag,dz_gn[n].real,dz_gn[n].imag))

            #compute couplings for final states  
            H_so2 = np.zeros((cis2.nroots,cis2.nroots),dtype=complex)
            for n in range(cis2.nroots):
              H_so2[n][n] = w2[n]
            #Add off-diagonal blocks to the Hamiltonian
            H_so2 += np.einsum("mai,ab,nbi->mn",np.conj(X2),H_so_mo[no_a:,no_a:],X2,optimize=True)
            H_so2 -= np.einsum("mai,ji,naj->mn",np.conj(X2),H_so_mo[:no_a,:no_a],X2,optimize=True)

            w_f, X_f = np.linalg.eigh(H_so2)

            dx_nf = np.matmul(np.conj(X_n.T),np.matmul(dx12,X_f))
            dy_nf = np.matmul(np.conj(X_n.T),np.matmul(dy12,X_f))
            dz_nf = np.matmul(np.conj(X_n.T),np.matmul(dz12,X_f))

            for n in range(cis1.nroots):
              for f in range(cis2.nroots):
                fp.write(" %i %i %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f \n"%(n+1,f+1,-w_n[n]+w_f[f],\
                     dx_nf[n][f].real,dx_nf[n][f].imag,dy_nf[n][f].real,dy_nf[n][f].imag,dz_nf[n][f].real,dz_nf[n][f].imag))
        
            fp.close() 

