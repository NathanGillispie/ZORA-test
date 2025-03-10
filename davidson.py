import numpy as np
import scipy as sp
import time
import xc_potential
import sys
import tracemalloc as tmal

class RDAVIDSON():

    def conv_test(self,q, evals,nroots):
      de = np.zeros(nroots)
      for n in range(nroots):
        de[n] = np.abs(np.sum(q.T[n]))
        if de[n] < 1e-4:
          self.is_converged[n] = 1
        else:
          self.is_converged[n] = 0
#        if np.sum(self.is_converged) == self.nroots:
#          if n == 0:
#            print("    Root  Subspace  Energy   dE  converged? ",flush=True)
#          print("  %5i %5i %8.5f %8.5f %s"%(n+1, len(evals), 27.21138*evals[n],de[n],("Y" if (self.is_converged[n] == 1) else "N")),flush=True)
      print("    Iter %5i:  %5i roots converged. Subspace size: %8i. Elapsed time: %8.2f seconds"%(self.itter+1,np.sum(self.is_converged), len(evals), time.time()-self.tic),flush=True) 
      if np.sum(self.is_converged) == self.nroots:
        print("\n    Root  Subspace  Energy   dE  converged? ",flush=True)
        for n in range(nroots):
          print("  %5i %5i    %8.4f   %8.5f %s"%(n+1, len(evals), 27.21138*evals[n],de[n],("Y" if (self.is_converged[n] == 1) else "N")),flush=True)
      return
    
    def sigma(self, Xt):
        #s = np.zeros((Xt.shape),dtype=complex)
        ndim = len(Xt[0])
           
        Xt = Xt.reshape(self.nv_a,self.no_a,ndim)

        s  = np.einsum("ab,bin->ain",self.F_virt,Xt,optimize=True)
        s -= np.einsum("ji,ajn->ain",self.F_occ,Xt,optimize=True)

        if self.in_core is False:
            if (self.wfn.reference == "rks"):
              if self.options.nofxc is True:
                print("WARNING!!! Excluding fxc",flush=True)
              else:
                xc_functional = xc_potential.XC(self.wfn)
                if self.wfn.triplet is True:
                  F = xc_functional.computeVx([self.Cmo[:,:self.no_a],self.Cmo[:,:self.no_a]],[self.Cmo[:,self.no_a:],self.Cmo[:,self.no_a:]],[Xt,-Xt],spin=1)
                  s += F[0]
                else:
                  F = xc_functional.computeVx(self.Cmo[:,:self.no_a],self.Cmo[:,self.no_a:],Xt,spin=0)
                  s += 2.*F[0] 
            for root in range(ndim):
              Xn = Xt[:,:,root]
              tmp  = np.einsum("ma,ai->mi",self.Cmo[:,self.no_a:],Xn,optimize=True)
              D    = np.einsum("mi,ni->mn",tmp,np.conjugate(self.Cmo[:,:self.no_a]),optimize=True)
              jao, kao = self.wfn.jk.get_jk(mol=self.ints,dm=D,hermi=0)
        
              tmp = np.einsum("ma,mn->an",np.conjugate(self.Cmo[:,self.no_a:]),jao,optimize=True)
              J   = np.einsum("an,ni->ai",tmp,self.Cmo[:,:self.no_a],optimize=True)
              tmp = np.einsum("ma,mn->an",np.conjugate(self.Cmo[:,self.no_a:]),kao,optimize=True)
              K   = np.einsum("an,ni->ai",tmp,self.Cmo[:,:self.no_a],optimize=True)

              if (self.wfn.reference == "rks"):
                if self.wfn.triplet is True:
                  s[:,:,root] -= self.options.xcalpha*K
                else:
                  s[:,:,root] += 2.*J - self.options.xcalpha*K
              else:
                if self.wfn.triplet is True:
                  s[:,:,root] -= K
                else:
                  s[:,:,root] += 2.*J - K
        else:
            s += np.einsum("aibj,bjN->aiN",self.G_aibj,Xt,optimize=True)           
            s -= np.einsum("abji,bjN->aiN",self.G_abji,Xt,optimize=True)           

        return s.reshape(self.nov,ndim)
    
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

        self.S = self.wfn.S[0]
        self.nroots = self.options.nroots
        self.maxdim = 50 * self.nroots
        self.maxiter = 50

        self.Cmo = self.wfn.C[0]

        self.Co = self.Cmo[:,:self.no_a]
        self.Cv = self.Cmo[:,self.no_a:]

        eps_a = self.wfn.eps[0]
        self.F_occ  = np.diag(eps_a[:self.no_a]) 
        self.F_virt = np.diag(eps_a[self.no_a:]) 

        self.in_core = self.options.tdscf_in_core
        if self.in_core is True:
          self.G_aibj = self.wfn.G_array[0]
          self.G_abji = self.wfn.G_array[1]

        self.itter = -1
        #toc = time.time()
        #print(" Davidson Initialization",toc-tic)
       


    def compute(self):
        #print(self.cvs)
        #return
        print("\n")
        print("    Configuration Interaction Singles") 
        print("    ---------------------------------") 
        print("\n")

        self.tic = time.time()
  
        #atomic L edge 
        #atomic K edge
        #low = 0
        #high = 1

        if (self.cvs is True):
          olow  = self.wfn.options.occupied[0] - 1
          ohigh = self.wfn.options.occupied[1] - 1
          print("    Restricting occupied orbitals in the window [%i,%i]"%(olow+1,ohigh+1),flush=True)
          self.no_a = int(ohigh-olow+1)
          self.F_occ = self.F_occ[olow:ohigh+1,olow:ohigh+1]
          self.Co      = self.Co[:,olow:ohigh+1]
          if (self.reduced_virtual is True):
            vlow  = self.wfn.options.virtual[0] - 1 - self.no_a_full
            vhigh = self.wfn.options.virtual[1] - 1 - self.no_a_full
            print("    Restricting virtual orbitals in the window [%i,%i]"%(vlow+1+self.no_a_full,vhigh+1+self.no_a_full),flush=True)
            self.nv_a = int(vhigh-vlow+1)
            self.F_virt = self.F_virt[vlow:vhigh+1,vlow:vhigh+1]
            self.Cv      = self.Cv[:,vlow:vhigh+1]
          else:
            vlow  = 0
            vhigh = self.nv_a_full
          
          self.nov = self.no_a * self.nv_a

          #if self.nov < self.nroots:
          #  print("    Resizing number of roots to %i"%self.nov)
          #  self.nroots = self.nov 
          #  self.maxdim = self.nov 

          self.Cmo = np.zeros((self.nbf,self.nv_a+self.no_a))
          self.Cmo[:,:self.no_a] = self.Co 
          self.Cmo[:,self.no_a:] = self.Cv 
          
          if self.in_core is True:
            self.G_aibj = self.G_aibj[vlow:vhigh+1,olow:ohigh+1,vlow:vhigh+1,olow:ohigh+1]
            self.G_abji = self.G_abji[vlow:vhigh+1,vlow:vhigh+1,olow:ohigh+1,olow:ohigh+1]

        if self.nov < self.nroots:
          print("    WARNING: Resizing number of roots to %i"%self.nov)
          self.nroots = self.nov 
          self.maxdim = 50*self.nov 

        self.is_converged = np.zeros(self.nroots,dtype=int)

        total_tic = time.time()
        dvec  = np.zeros((self.nov,self.nroots))
        trial = np.zeros((self.nov,self.maxdim))
        
        self.ints = self.wfn.ints_factory

        occ = np.zeros((self.nv_a+self.no_a,self.nv_a+self.no_a))
        occ[:self.no_a,:self.no_a] = np.eye(self.no_a)

        if self.in_core is False:
          D   = self.wfn.D[0]
          jao, kao = self.wfn.jk.get_jk(mol=self.ints,dm=D,hermi=0)
          tmp = np.einsum("ma,mn->an",self.Cv,jao,optimize=True)
          J   = np.einsum("an,ni->ai",tmp,self.Co,optimize=True)
          tmp = np.einsum("ma,mn->an",self.Cv,kao,optimize=True)
          K   = np.einsum("an,ni->ai",tmp,self.Co,optimize=True)

        #dtic = time.time()
        Adiag = np.zeros(self.nov)
        #print(self.nov, self.no_a*self.nv_a, self.no_a, self.nv_a,self.G_aibj.shape)
        #exit(0)   
        for a in range(self.nv_a):
          for i in range(self.no_a):
            ai = a*self.no_a+i
            Adiag[ai]  = self.F_virt[a][a] - self.F_occ[i][i]
            if self.in_core is True:
              Adiag[ai] += 2.*self.G_aibj[a][i][a][i] - self.G_abji[a][a][i][i]
            else:
              if self.wfn.triplet is True: 
                Adiag[ai] += - K[a][i]
              else:
                Adiag[ai] += 2.*J[a][i] - K[a][i]

        #dtoc = time.time()
        #print("  Diagonal Guess Construction in Davidson took %f"%(dtoc-dtic),flush=True)

        Aidx = np.argsort(Adiag)[:self.nroots]
        trial[:,:self.nroots] = np.eye(self.nov)[:,Aidx]
        
        s = self.sigma(trial[:,:self.nroots])
        A = np.matmul(np.conjugate(trial[:,:self.nroots].T),s[:,:self.nroots])
        #A = np.einsum("nk,km->nm",np.conj(trial[:,:self.nroots].T),s[:,:self.nroots],optimize=True)
        evals , evecs = np.linalg.eigh(A)
        X  = np.matmul(trial[:,:self.nroots],evecs[:self.nroots,:self.nroots])
        q = self.sigma(X) - np.einsum("n,in->in",evals,X)
        
        trial_length = len(trial[0])
        self.conv_test(q, evals, self.nroots)
        if np.sum(self.is_converged) == self.nroots:
          print("   Eigenvalues converged!")
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
        else:
          #print(" ")
          ldim = self.nroots
          hdim = 2*self.nroots
          ones = np.ones((self.nov,self.nroots))


          for self.itter in range(self.maxiter):
            #print("   Iteration %i "%(itter+1),"")
        
            denom  = np.einsum("ki,i->ki",ones,evals[:self.nroots])
            denom -= np.einsum("k,ki->ki",Adiag,ones)
            denom += 1e-12 #to avoid hard zeroes
            dvec = -q/denom
        
            trial[:,ldim:hdim] = dvec
            orth_trial = sp.linalg.orth(trial[:,:hdim])
            #print("   Orthogonalization %f"%(np.sum(np.conjugate(orth_trial.T)*(orth_trial.T))/hdim))
        
            s = self.sigma(orth_trial)
            A = np.matmul(np.conjugate(orth_trial.T),s)
            evals , evecs = sp.linalg.eigh(A)
            X  = np.matmul(orth_trial,evecs[:,:self.nroots])
            q  = self.sigma(X) - np.einsum("n,in->in",evals[:self.nroots],X)
            self.conv_test(q, evals, self.nroots)
            trial_length = len(orth_trial[0])
            if np.sum(self.is_converged) == self.nroots:
              if self.cvs is True:
                #expand original CI vectors to avoid problem with dimensions later
                low  = self.wfn.options.occupied[0] - 1
                high = self.wfn.options.occupied[1] - 1
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
            else:
          #    print(" ")
              ldim += self.nroots
              hdim += self.nroots
          #    print(ldim, hdim)
            if (trial_length == self.nov) or (trial_length == self.maxdim):
              print("    Maximum dimension reached. Davidson algorithm failed to converge!",flush=True)    
              if (self.nov < 50000) and (self.in_core is True):
                self.nroots = self.nov
                print("    Attempting Full Diagonalization for %i roots..."%(self.nroots),flush=True)
                A  = np.einsum("ab,ij->aibj",self.F_virt,np.eye(self.no_a),optimize=True)
                A -= np.einsum("ab,ij->aibj",np.eye(self.nv_a),self.F_occ,optimize=True)
                if self.wfn.triplet is True:
                  A -= self.G_abji.swapaxes(1,3).swapaxes(2,3) 
                else:
                  A += 2.*self.G_aibj
                  A -= self.G_abji.swapaxes(1,3).swapaxes(2,3) 
                A  = A.reshape(self.no_a*self.nv_a,self.no_a*self.nv_a) 
                evals, X = sp.linalg.eigh(A)
              else:
                print("    Full dimension too large (>18 Gb) to be stored in memory",flush=True)
              return evals[:self.nroots], X
        
        #total_toc = time.time()
        #davidson_time = total_toc - tic
        #
        #print("   Davidson diagonalization performed in %8.5f seconds"%davidson_time)
        #print("   Ratio of the full space spanned by subspace: %f"%(trial_length/self.nov))

class GDAVIDSON():

    def conv_test(self,q, evals,nroots):
      de = np.zeros(nroots)
      for n in range(nroots):
        de[n] = np.abs(np.sum(q.T[n]))
        if de[n] < 1e-4:
          self.is_converged[n] = 1
        else:
          self.is_converged[n] = 0
#        if np.sum(self.is_converged) == self.nroots:
#          if n == 0:
#            print("    Root  Subspace  Energy   dE  converged? ",flush=True)
#          print("  %5i %5i %8.5f %8.5f %s"%(n+1, len(evals), 27.21138*evals[n],de[n],("Y" if (self.is_converged[n] == 1) else "N")),flush=True)
      print("    Iter %5i:  %5i roots converged. Subspace size: %8i. Elapsed time: %8.2f seconds"%(self.itter+1,np.sum(self.is_converged), len(evals), time.time()-self.tic),flush=True) 
      if np.sum(self.is_converged) == self.nroots:
        print("\n    Root  Subspace  Energy   dE  converged? ",flush=True)
        for n in range(nroots):
          print("  %5i %5i    %8.4f   %8.5f %s"%(n+1, len(evals), 27.21138*evals[n],de[n],("Y" if (self.is_converged[n] == 1) else "N")),flush=True)
      return
    
    def sigma(self, Xt):
        #s = np.zeros((Xt.shape),dtype=complex)
        ndim = len(Xt[0])
           
        Xt = Xt.reshape(self.nv_a,self.no_a,ndim)

        s  = np.einsum("ab,bin->ain",self.F_virt,Xt,optimize=True)
        s -= np.einsum("ji,ajn->ain",self.F_occ,Xt,optimize=True)

        ##DRN TEST BEGIN 
        if self.options.fonly is True:
          print("WARNING!!! Setting A = F",flush=True)
          s = s.reshape(self.nov,ndim)
          #s_b = s_b.reshape(self.nov_b,ndim)
          #print(s_a)
          #print("")
          #print(s_b)
          #exit(0)
          toc = time.time()
          #print("Sigma build",toc-tic)
          return s
        ##DRN TEST END 

        if self.in_core is False:
            nbf  = int(self.nbf/2)
            jbig = np.zeros((self.nbf,self.nbf),dtype=complex)
            kbig = np.zeros((self.nbf,self.nbf),dtype=complex)
            Darray = np.zeros((4,nbf,nbf),dtype=complex)
            for n in range(ndim):
              tmp = np.einsum("ma,ai->mi",self.Cv,Xt[:,:,n],optimize=True)
              D   = np.einsum("mi,ni->mn",tmp,np.conj(self.Co),optimize=True)
              #Darray[0] = D[:nbf,:nbf]
              #Darray[1] = D[nbf:,nbf:]
              #Darray[2] = D[:nbf,nbf:]
              #Darray[3] = D[nbf:,:nbf]
              #j, k = self.wfn.jk.get_jk(mol=self.ints,dm=Darray,hermi=0)
              jbig, kbig = self.wfn.jk.get_jk(mol=self.ints,dm=D,hermi=0)
              #jbig[:nbf,:nbf] = j[0] + j[1]
              #jbig[nbf:,nbf:] = j[0] + j[1]
              #kbig[:nbf,:nbf] = k[0]
              #kbig[nbf:,nbf:] = k[1]
              #kbig[:nbf,nbf:] = k[2]
              #kbig[nbf:,:nbf] = k[3]
              tmp = np.einsum("ma,mn->an",np.conj(self.Cv),jbig,optimize=True)
              J   = np.einsum("an,ni->ai",tmp,self.Co,optimize=True)
              tmp = np.einsum("ma,mn->an",np.conj(self.Cv),kbig,optimize=True)
              K   = np.einsum("an,ni->ai",tmp,self.Co,optimize=True)

              if (self.wfn.reference == "gks"):
                s[:,:,n] += J - self.options.xcalpha*K
              else:
                s[:,:,n] += J - K

           # jscript = ['ijkl,kl->ij'] * ndim
           # kscript = ['ijkl,jk->il'] * ndim

           # #alpha-alpha real
           # self.D = np.einsum("ma,aix,ni->xmn",self.Cmo[:,self.no_a:],Xt,np.conjugate(self.Cmo[:,:self.no_a]),optimize=True)[:,:nbf,:nbf].real
           # jbig[:,:nbf,:nbf] += np.asarray(jk.get_jk(mols=self.ints,dms=self.D,scripts=jscript,intor='int2e_sph'))
           # kbig[:,:nbf,:nbf] += np.asarray(jk.get_jk(mols=self.ints,dms=self.D,scripts=kscript,intor='int2e_sph'))

           # #alpha-alpha imag
           # self.D = np.einsum("ma,aix,ni->xmn",self.Cmo[:,self.no_a:],Xt,np.conjugate(self.Cmo[:,:self.no_a]),optimize=True)[:,:nbf,:nbf].imag
           # jbig[:,:nbf,:nbf] += 1j*np.asarray(jk.get_jk(mols=self.ints,dms=self.D,scripts=jscript,intor='int2e_sph'))
           # kbig[:,:nbf,:nbf] += 1j*np.asarray(jk.get_jk(mols=self.ints,dms=self.D,scripts=kscript,intor='int2e_sph'))

           # #beta-beta real
           # self.D = np.einsum("ma,aix,ni->xmn",self.Cmo[:,self.no_a:],Xt,np.conjugate(self.Cmo[:,:self.no_a]),optimize=True)[:,nbf:,nbf:].real
           # jbig[:,:nbf,:nbf] += np.asarray(jk.get_jk(mols=self.ints,dms=self.D,scripts=jscript,intor='int2e_sph'))
           # kbig[:,nbf:,nbf:] += np.asarray(jk.get_jk(mols=self.ints,dms=self.D,scripts=kscript,intor='int2e_sph'))

           # #beta-beta imag
           # self.D = np.einsum("ma,aix,ni->xmn",self.Cmo[:,self.no_a:],Xt,np.conjugate(self.Cmo[:,:self.no_a]),optimize=True)[:,nbf:,nbf:].imag
           # jbig[:,:nbf,:nbf] += 1j*np.asarray(jk.get_jk(mols=self.ints,dms=self.D,scripts=jscript,intor='int2e_sph'))
           # jbig[:,nbf:,nbf:] = jbig[:,:nbf,:nbf]
           # kbig[:,nbf:,nbf:] += 1j*np.asarray(jk.get_jk(mols=self.ints,dms=self.D,scripts=kscript,intor='int2e_sph'))

           # #alpha-beta real
           # self.D = np.einsum("ma,aix,ni->xmn",self.Cmo[:,self.no_a:],Xt,np.conjugate(self.Cmo[:,:self.no_a]),optimize=True)[:,:nbf,nbf:].real
           # kbig[:,:nbf,nbf:] += np.asarray(jk.get_jk(mols=self.ints,dms=self.D,scripts=kscript,intor='int2e_sph'))

           # #alpha-beta imag
           # self.D = np.einsum("ma,aix,ni->xmn",self.Cmo[:,self.no_a:],Xt,np.conjugate(self.Cmo[:,:self.no_a]),optimize=True)[:,:nbf,nbf:].imag
           # kbig[:,:nbf,nbf:] += 1j*np.asarray(jk.get_jk(mols=self.ints,dms=self.D,scripts=kscript,intor='int2e_sph'))

           # #beta-alpha real
           # self.D = np.einsum("ma,aix,ni->xmn",self.Cmo[:,self.no_a:],Xt,np.conjugate(self.Cmo[:,:self.no_a]),optimize=True)[:,nbf:,:nbf].real
           # kbig[:,nbf:,:nbf] += np.asarray(jk.get_jk(mols=self.ints,dms=self.D,scripts=kscript,intor='int2e_sph'))

           # #beta-alpha imag
           # self.D = np.einsum("ma,aix,ni->xmn",self.Cmo[:,self.no_a:],Xt,np.conjugate(self.Cmo[:,:self.no_a]),optimize=True)[:,nbf:,:nbf].imag
           # kbig[:,nbf:,:nbf] += 1j*np.asarray(jk.get_jk(mols=self.ints,dms=self.D,scripts=kscript,intor='int2e_sph'))

           # J = np.einsum("ma,Nmn,ni->aiN",np.conjugate(self.Cmo[:,self.no_a:]),jbig,self.Cmo[:,:self.no_a],optimize=True)
           # K = np.einsum("ma,Nmn,ni->aiN",np.conjugate(self.Cmo[:,self.no_a:]),kbig,self.Cmo[:,:self.no_a],optimize=True)
           # if (self.wfn.reference == "gks"):
           #   #tic = time.time()
           #   #xc_functional = xc_potential.XC(self.wfn)
           #   #F_a, F_b = xc_functional.computeVx([self.Co_a,self.Co_b],[self.Cv_a,self.Cv_b],[Xt_a,Xt_b],spin=1)
           #   #s_a += F_a
           #   #s_b += F_b
           #   #toc = time.time()
           #   ##print("  Fxc contrinution",toc-tic,flush=True)
           #   s += J - self.options.xcalpha*K
           # else:
           #   s += J - K
        else:
            s += np.einsum("aibj,bjN->aiN",self.G_aibj,Xt,optimize=True)           
            s -= np.einsum("abji,bjN->aiN",self.G_abji,Xt,optimize=True)           

        return s.reshape(self.nov,ndim)
    
    def __init__(self, wfn):
        #tic = time.time()
         
        #begin Davidson
        self.wfn = wfn
        self.options = self.wfn.options
        self.mol = wfn.ints_factory
        self.ref = wfn.reference
        if (self.ref == "ghf") or (self.ref == "gks") or (self.ref == "rgks") or (self.ref == "rghf"):
          self.nbf   = int(2*self.wfn.nbf)
          self.no_a  = int(self.wfn.nel[0])
        else:
          self.nbf   = int(self.wfn.nbf)
          self.no_a  = int(self.wfn.nel[0])
        self.nv_a  = int(self.nbf - self.no_a)
        self.nov   = int(self.no_a * self.nv_a)
        self.no_a_full  = int(self.wfn.nel[0])
        self.nov_a_full   = int(self.no_a * self.nv_a)
        print(self.nbf, self.no_a, self.nv_a, self.nov)

        #nbf   = int(2*self.mol["nbf"])
        #nv_a  = int(self.nbf - self.no_a)
        #nov   = int(self.no_a * self.nv_a)
        self.S = self.wfn.S
        self.nroots = self.options.nroots
        self.maxdim = 50 * self.nroots
        self.maxiter = 50

        self.Cmo = self.wfn.C[0]

        self.Co = self.Cmo[:,:self.no_a]
        self.Cv = self.Cmo[:,self.no_a:]

        eps_a = self.wfn.eps[0]

        self.F_occ  = np.diag(eps_a[:self.no_a]) + 1j*np.zeros((self.no_a, self.no_a))
        self.F_virt = np.diag(eps_a[self.no_a:]) + 1j*np.zeros((self.nv_a, self.nv_a))

        self.in_core = self.options.tdscf_in_core
        if self.in_core is True:
          self.G_aibj = self.wfn.G_array[0]
          self.G_abji = self.wfn.G_array[1]

        self.itter = -1
        #toc = time.time()
        #print(" Davidson Initialization",toc-tic)
       


    def compute(self):
        #print(self.cvs)
        #return
        print("\n")
        print("    Configuration Interaction Singles") 
        print("    ---------------------------------") 
        print("\n")

        self.tic = time.time()
  
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

          #if self.nov < self.nroots:
          #  print("    Resizing number of roots to %i"%self.nov)
          #  self.nroots = self.nov 
          #  self.maxdim = self.nov 

          self.Cmo = np.zeros((self.nbf,self.nv_a+self.no_a),dtype=complex)
          self.Cmo[:,:self.no_a] = self.Co 
          self.Cmo[:,self.no_a:] = self.Cv 
          
          if self.in_core is True:
            self.G_aibj = self.G_aibj[:,low:high+1,:,low:high+1]
            self.G_abji = self.G_abji[:,:,low:high+1,low:high+1]

        if self.nov < self.nroots:
          print("    WARNING: Resizing number of roots to %i"%self.nov)
          self.nroots = self.nov 
          self.maxdim = 50*self.nov 

        self.is_converged = np.zeros(self.nroots,dtype=int)

        total_tic = time.time()
        dvec  = np.zeros((self.nov,self.nroots),dtype=complex)
        trial = np.zeros((self.nov,self.maxdim),dtype=complex)
        
   
        #self.ghf_wfn = self.wfn["ghf"]
        self.ints = self.wfn.ints_factory

        occ = np.zeros((self.nv_a+self.no_a,self.nv_a+self.no_a))
        occ[:self.no_a,:self.no_a] = np.eye(self.no_a)

        if self.in_core is False:
          jbig = np.zeros((self.nbf,self.nbf),dtype=complex)
          kbig = np.zeros((self.nbf,self.nbf),dtype=complex)
          nbf = self.nbf//2

          #Darray = np.zeros((4,nbf,nbf),dtype=complex)

          D = self.wfn.D[0]
          #Darray[0] = D[:nbf,:nbf]
          #Darray[1] = D[nbf:,nbf:]
          #Darray[2] = D[:nbf,nbf:]
          #Darray[3] = D[nbf:,:nbf]
          #j, k = self.wfn.jk.get_jk(mol=self.ints,dm=Darray,hermi=0)
          jbig, kbig = self.wfn.jk.get_jk(mol=self.ints,dm=D,hermi=0)
          #jbig[:nbf,:nbf] = j[0] + j[1]
          #jbig[nbf:,nbf:] = j[0] + j[1]
          #kbig[:nbf,:nbf] = k[0]
          #kbig[nbf:,nbf:] = k[1]
          #kbig[:nbf,nbf:] = k[2]
          #kbig[nbf:,:nbf] = k[3]
          tmp = np.einsum("ma,mn->an",np.conj(self.Cv),jbig,optimize=True)
          J   = np.einsum("an,ni->ai",tmp,self.Co,optimize=True)
          tmp = np.einsum("ma,mn->an",np.conj(self.Cv),kbig,optimize=True)
          K   = np.einsum("an,ni->ai",tmp,self.Co,optimize=True)

#
#
#          #alpha-alpha
#          #self.D = np.einsum("mp,pq,nq->mn",self.Cmo,occ,np.conj(self.Cmo),optimize=True)[:hnbf,:hnbf]
#          self.D = self.wfn.D[0][:hnbf,:hnbf]
#          jbig[:hnbf,:hnbf] += jk.get_jk(mols=self.ints,dms=self.D,scripts='ijkl,kl->ij',intor='int2e_sph')
#          kbig[:hnbf,:hnbf] += jk.get_jk(mols=self.ints,dms=self.D,scripts='ijkl,jk->il',intor='int2e_sph')
#
#          #beta-beta
#          #self.D = np.einsum("mp,pq,nq->mn",self.Cmo,occ,np.conj(self.Cmo),optimize=True)[hnbf:,hnbf:]
#          self.D = self.wfn.D[0][hnbf:,hnbf:]
#          jbig[:hnbf,:hnbf] += jk.get_jk(mols=self.ints,dms=self.D,scripts='ijkl,kl->ij',intor='int2e_sph')
#          kbig[hnbf:,hnbf:] += jk.get_jk(mols=self.ints,dms=self.D,scripts='ijkl,jk->il',intor='int2e_sph')
#
#          jbig[hnbf:,hnbf:] = jbig[:hnbf,:hnbf]
#
#          #alpha-beta
#          #self.D = np.einsum("mp,pq,nq->mn",self.Cmo,occ,np.conj(self.Cmo),optimize=True)[:hnbf,hnbf:]
#          self.D = self.wfn.D[0][:hnbf,hnbf:]
#          kbig[:hnbf,hnbf:] += jk.get_jk(mols=self.ints,dms=self.D,scripts='ijkl,jk->il',intor='int2e_sph')
#          kbig[hnbf:,:hnbf] = np.conjugate(kbig[:hnbf,hnbf:])
#
#          J = np.einsum("ma,mn,ni->ai",np.conj(self.Cmo[:,self.no_a:]),jbig,self.Cmo[:,:self.no_a],optimize=True)
#          K = np.einsum("ma,mn,ni->ai",np.conj(self.Cmo[:,self.no_a:]),kbig,self.Cmo[:,:self.no_a],optimize=True) 

        Adiag = np.zeros(self.nov,dtype=complex)
        for a in range(self.nv_a):
          for i in range(self.no_a):
            ai = a*self.no_a+i
            Adiag[ai]  = self.F_virt[a][a] - self.F_occ[i][i]
            if self.in_core is True:
              Adiag[ai] += self.G_aibj[a][i][a][i] - self.G_abji[a][a][i][i]
            else:
              Adiag[ai] += J[a][i] - K[a][i]

        Aidx = np.argsort(Adiag)[:self.nroots]
        trial[:,:self.nroots] = np.eye(self.nov)[:,Aidx]
        
        s = self.sigma(trial[:,:self.nroots])
        A = np.matmul(np.conjugate(trial[:,:self.nroots].T),s[:,:self.nroots])
        #A = np.einsum("nk,km->nm",np.conj(trial[:,:self.nroots].T),s[:,:self.nroots],optimize=True)
        evals , evecs = sp.linalg.eigh(A)
        X  = np.matmul(trial[:,:self.nroots],evecs[:self.nroots,:self.nroots])
        q = self.sigma(X) - np.einsum("n,in->in",evals,X)
        
        trial_length = len(trial[0])
        self.conv_test(q, evals, self.nroots)
        if np.sum(self.is_converged) == self.nroots:
          print("   Eigenvalues converged!")
          if self.cvs is True:
            #expand original CI vectors to avoid problem with dimensions later
            low  = self.wfn.options.occupied[0] - 1
            high = self.wfn.options.occupied[1] - 1
            Xfull = np.zeros((self.nroots,self.nv_a,self.no_a_full),dtype=complex)
            print("    Expanding occupied space from %i to %i"%(self.no_a,self.no_a_full))
            X = (X.T).reshape(self.nroots,self.nv_a,self.no_a)
            Xfull[:,:,low:high+1] = X
            Xfull = Xfull.reshape(self.nroots,self.nov_a_full)
            print(Xfull.shape)
            self.Cmo = self.wfn.C[0]
            self.no_a = self.no_a_full
            self.nov  = self.nov_a_full
            return evals[:self.nroots], Xfull.T
          else:
            return evals[:self.nroots], X
        else:
          #print(" ")
          ldim = self.nroots
          hdim = 2*self.nroots
          ones = np.ones((self.nov,self.nroots),dtype=complex)


          for self.itter in range(self.maxiter):
            #print("   Iteration %i "%(itter+1),"")
        
            denom  = np.einsum("ki,i->ki",ones,evals[:self.nroots])
            denom -= np.einsum("k,ki->ki",Adiag,ones)
            denom += 1e-12 #to avoid hard zeroes
            dvec = -q/denom
        
            trial[:,ldim:hdim] = dvec
            orth_trial = sp.linalg.orth(trial[:,:hdim])
            #print("   Orthogonalization %f"%(np.sum(np.conjugate(orth_trial.T)*(orth_trial.T))/hdim))
        
            s = self.sigma(orth_trial)
            A = np.matmul(np.conjugate(orth_trial.T),s)
            evals , evecs = sp.linalg.eigh(A)
            X  = np.matmul(orth_trial,evecs[:,:self.nroots])
            q  = self.sigma(X) - np.einsum("n,in->in",evals[:self.nroots],X)
            self.conv_test(q, evals, self.nroots)
            trial_length = len(orth_trial[0])
            if np.sum(self.is_converged) == self.nroots:
              if self.cvs is True:
                #expand original CI vectors to avoid problem with dimensions later
                low  = self.wfn.options.occupied[0] - 1
                high = self.wfn.options.occupied[1] - 1
                Xfull = np.zeros((self.nroots,self.nv_a,self.no_a_full),dtype=complex)
                print("    Expanding occupied space from %i to %i"%(self.no_a,self.no_a_full))
                X = (X.T).reshape(self.nroots,self.nv_a,self.no_a)
                Xfull[:,:,low:high+1] = X
                Xfull = Xfull.reshape(self.nroots,self.nov_a_full)
                print(Xfull.shape)
                self.Cmo = self.wfn.C[0]
                self.no_a = self.no_a_full
                self.nov  = self.nov_a_full
                return evals[:self.nroots], Xfull.T
              else:
                return evals[:self.nroots], X
            else:
          #    print(" ")
              ldim += self.nroots
              hdim += self.nroots
          #    print(ldim, hdim)
            if (trial_length == self.nov) or (trial_length == self.maxdim):
              print("    Maximum dimension reached. Davidson algorithm failed to converge!",flush=True)    
              if (self.nov < 50000) and (self.in_core is True):
                self.nroots = self.nov
                print("    Attempting Full Diagonalization for %i roots..."%(self.nroots),flush=True)
                A  = np.einsum("ab,ij->aibj",self.F_virt,np.eye(self.no_a),optimize=True)
                A -= np.einsum("ab,ij->aibj",np.eye(self.nv_a),self.F_occ,optimize=True)
                A += self.G_aibj
                A -= self.G_abji.swapaxes(1,3).swapaxes(2,3) 
                A  = A.reshape(self.no_a*self.nv_a,self.no_a*self.nv_a) 
                evals, X = sp.linalg.eigh(A)
              else:
                print("    Full dimension too large (>18 Gb) to be stored in memory",flush=True)
              return evals[:self.nroots], X
        
        #total_toc = time.time()
        #davidson_time = total_toc - tic
        #
        #print("   Davidson diagonalization performed in %8.5f seconds"%davidson_time)
        #print("   Ratio of the full space spanned by subspace: %f"%(trial_length/self.nov))

class UDAVIDSON():

    def conv_test(self,q, evals,nroots):
      de = np.zeros(nroots)
      for n in range(nroots):
        de[n] = np.abs(np.sum(q.T[n]))
        if de[n] < 1e-4:
          self.is_converged[n] = 1
        else:
          self.is_converged[n] = 0
#        if np.sum(self.is_converged) == self.nroots:
#          if n == 0:
#            print("    Root  Subspace  Energy   dE  converged? ",flush=True)
#          print("  %5i %5i %8.5f %8.5f %s"%(n+1, len(evals), 27.21138*evals[n],de[n],("Y" if (self.is_converged[n] == 1) else "N")),flush=True)
      print("    Iter %5i:  %5i roots converged. Subspace size: %8i. Elapsed time: %8.2f seconds"%(self.itter+1,np.sum(self.is_converged), len(evals), time.time()-self.tic),flush=True) 
      if np.sum(self.is_converged) == self.nroots:
        print("\n    Root  Subspace  Energy   dE  converged? ",flush=True)
        for n in range(nroots):
          print("  %5i %5i    %8.4f   %8.5f %s"%(n+1, len(evals), 27.21138*evals[n],de[n],("Y" if (self.is_converged[n] == 1) else "N")),flush=True)
      return
    
    def sigma(self, Xt):
        tic = time.time()
        #s = np.zeros((Xt.shape),dtype=complex)
        ndim = len(Xt[0])
        #print(Xt[:,0])
           
        Xt_a = Xt[:self.nov_a,:].reshape(self.nv_a,self.no_a,ndim)
        Xt_b = Xt[self.nov_a:,:].reshape(self.nv_b,self.no_b,ndim)


        tic = time.time()
        s_a  = np.einsum("ab,bin->ain",self.F_virt_a,Xt_a,optimize=True)
        s_a -= np.einsum("ji,ajn->ain",self.F_occ_a,Xt_a,optimize=True)

        s_b  = np.einsum("ab,bin->ain",self.F_virt_b,Xt_b,optimize=True)
        s_b -= np.einsum("ji,ajn->ain",self.F_occ_b,Xt_b,optimize=True)
        toc = time.time()
        #print("  F contrinution",toc-tic,flush=True)

        ##DRN TEST BEGIN 
        if self.options.fonly is True:
          print("WARNING!!! Setting A = F",flush=True)
          s_a = s_a.reshape(self.nov_a,ndim)
          s_b = s_b.reshape(self.nov_b,ndim)
          #print(s_a)
          #print("")
          #print(s_b)
          #exit(0)
          toc = time.time()
          #print("Sigma build",toc-tic)
          return np.concatenate((s_a,s_b),axis=0)
        ##DRN TEST END 

        if self.in_core is False:
          if (self.wfn.reference == "uks"):
            ##DRN TEST BEGIN
            if self.options.nofxc is True:
              print("WARNING!!! Excluding fxc",flush=True)
            else:
              print("Computing fxc",flush=True)
              tic = time.time()
              xc_functional = xc_potential.XC(self.wfn)
              F_a, F_b = xc_functional.computeVx([self.Co_a,self.Co_b],[self.Cv_a,self.Cv_b],[Xt_a,Xt_b],spin=1)
              s_a += F_a
              s_b += F_b
              toc = time.time()

            #tic = time.time()
            #xc_functional = xc_potential.XC(self.wfn)
            #F_a, F_b = xc_functional.computeVx([self.Co_a,self.Co_b],[self.Cv_a,self.Cv_b],[Xt_a,Xt_b],spin=1)
            #s_a += F_a
            #s_b += F_b
            #toc = time.time()
            ##DRN TEST END
          print("akonly?", self.options.akonly)
          print("jkonly?", self.options.jkonly)
          print("jonly?", self.options.jonly)
          print("nofxc?", self.options.nofxc)
          for n in range(ndim):
            tic = time.time()
            
            if self.options.jkonly is True:
              D  = np.einsum("ma,ai,ni->mn",self.Cv_a,Xt_a[:,:,n],self.Co_a,optimize=True)
              D += np.einsum("ma,ai,ni->mn",self.Cv_b,Xt_b[:,:,n],self.Co_b,optimize=True)
              jao_a, _ = self.wfn.jk.get_jk(mol=self.ints,dm=D,hermi=0)

              D  = np.einsum("ma,ai,ni->mn",self.Cv_a,Xt_a[:,:,n],self.Co_a,optimize=True)
              _, kao_a = self.wfn.jk.get_jk(mol=self.ints,dm=D,hermi=0)

              D  = np.einsum("ma,ai,ni->mn",self.Cv_b,Xt_b[:,:,n],self.Co_b,optimize=True)
              _, kao_b = self.wfn.jk.get_jk(mol=self.ints,dm=D,hermi=0)

              J_a = np.einsum("ma,mn,ni->ai",self.Cv_a,jao_a,self.Co_a,optimize=True)
              J_b = np.einsum("ma,mn,ni->ai",self.Cv_b,jao_a,self.Co_b,optimize=True)
              K_a = np.einsum("ma,mn,ni->ai",self.Cv_a,kao_a,self.Co_a,optimize=True)
              K_b = np.einsum("ma,mn,ni->ai",self.Cv_b,kao_b,self.Co_b,optimize=True)
              toc = time.time()

              s_a[:,:,n] += J_a - K_a
              s_b[:,:,n] += J_b - K_b
           
            elif self.options.jonly is True:
              D  = np.einsum("ma,ai,ni->mn",self.Cv_a,Xt_a[:,:,n],self.Co_a,optimize=True)
              D += np.einsum("ma,ai,ni->mn",self.Cv_b,Xt_b[:,:,n],self.Co_b,optimize=True)
              jao_a, _ = self.wfn.jk.get_jk(mol=self.ints,dm=D,hermi=0)

              J_a = np.einsum("ma,mn,ni->ai",self.Cv_a,jao_a,self.Co_a,optimize=True)
              J_b = np.einsum("ma,mn,ni->ai",self.Cv_b,jao_a,self.Co_b,optimize=True)
              if n in self.options.roots_lookup_table:
                f1 = open("root_"+str(n)+"_dens_"+str(self.itter),"w")
                for mu in range(len(D)):
                  for nu in range(len(D[mu])):
                    f1.write("%i %i %20.15f \n"%(mu+1,nu+1,D[mu][nu]))
                  f1.write("\n")
                f1.close()  

                f1 = open("root_"+str(n)+"_coul_"+str(self.itter),"w")
                for mu in range(len(jao_a)):
                  for nu in range(len(jao_a[mu])):
                    f1.write("%i %i %20.15f \n"%(mu+1,nu+1,jao_a[mu][nu]))
                  f1.write("\n")
                f1.close()  

              toc = time.time()

              s_a[:,:,n] += J_a
              s_b[:,:,n] += J_b

            elif self.options.akonly is True:
              D_a  = np.einsum("ma,ai,ni->mn",self.Cv_a,Xt_a[:,:,n],self.Co_a,optimize=True)
              _, kao_a = self.wfn.jk.get_jk(mol=self.ints,dm=D_a,hermi=0)

              D_b  = np.einsum("ma,ai,ni->mn",self.Cv_b,Xt_b[:,:,n],self.Co_b,optimize=True)
              _, kao_b = self.wfn.jk.get_jk(mol=self.ints,dm=D_b,hermi=0)

              K_a = np.einsum("ma,mn,ni->ai",self.Cv_a,kao_a,self.Co_a,optimize=True)
              K_b = np.einsum("ma,mn,ni->ai",self.Cv_b,kao_b,self.Co_b,optimize=True)

              if n in self.options.roots_lookup_table:
                f1 = open("root_"+str(n)+"_dens_"+str(self.itter),"w")
                for mu in range(len(D_a)):
                  for nu in range(len(D_a[mu])):
                    f1.write("%i %i %20.15f %20.15f \n"%(mu+1,nu+1,D_a[mu][nu],D_b[mu][nu]))
                  f1.write("\n")
                f1.close()  

                f1 = open("root_"+str(n)+"_exch_"+str(self.itter),"w")
                for mu in range(len(kao_a)):
                  for nu in range(len(kao_a[mu])):
                    f1.write("%i %i %20.15f %20.12f \n"%(mu+1,nu+1,kao_a[mu][nu],kao_b[mu][nu]))
                  f1.write("\n")
                f1.close()  
              toc = time.time()

              s_a[:,:,n] += - self.options.xcalpha*K_a
              s_b[:,:,n] += - self.options.xcalpha*K_b

            else:
              D  = np.einsum("ma,ai,ni->mn",self.Cv_a,Xt_a[:,:,n],self.Co_a,optimize=True)
              D += np.einsum("ma,ai,ni->mn",self.Cv_b,Xt_b[:,:,n],self.Co_b,optimize=True)
              jao_a, _ = self.wfn.jk.get_jk(mol=self.ints,dm=D,hermi=0)

              D_a  = np.einsum("ma,ai,ni->mn",self.Cv_a,Xt_a[:,:,n],self.Co_a,optimize=True)
              _, kao_a = self.wfn.jk.get_jk(mol=self.ints,dm=D_a,hermi=0)

              D_b  = np.einsum("ma,ai,ni->mn",self.Cv_b,Xt_b[:,:,n],self.Co_b,optimize=True)
              _, kao_b = self.wfn.jk.get_jk(mol=self.ints,dm=D_b,hermi=0)

              #fo mu in range(len(jao_a)):
              #  for nu in range(len(jao_a[mu])):
              #    if (mu in list(np.arange(0,69,1))) or (nu in list(np.arange(0,69,1))) :
              #        continue
              #    else:
              #      jao_a[mu][nu] = 0.
              #      kao_a[mu][nu] = 0.
              #      kao_b[mu][nu] = 0.

              J_a = np.einsum("ma,mn,ni->ai",self.Cv_a,jao_a,self.Co_a,optimize=True)
              J_b = np.einsum("ma,mn,ni->ai",self.Cv_b,jao_a,self.Co_b,optimize=True)
              K_a = np.einsum("ma,mn,ni->ai",self.Cv_a,kao_a,self.Co_a,optimize=True)
              K_b = np.einsum("ma,mn,ni->ai",self.Cv_b,kao_b,self.Co_b,optimize=True)
              


              if n in self.options.roots_lookup_table:
                f1 = open("root_"+str(n)+"_Jai_"+str(self.itter),"w")
                for mu in range(len(J_a)):
                  for nu in range(len(J_a[mu])):
                    f1.write("%i %i %20.15f %20.15f\n"%(mu+1,nu+1,J_a[mu][nu],J_b[mu][nu]))
                  f1.write("\n")
                f1.close()  

                f1 = open("root_"+str(n)+"_Kai_"+str(self.itter),"w")
                for mu in range(len(K_a)):
                  for nu in range(len(K_a[mu])):
                    f1.write("%i %i %20.15f %20.15f\n"%(mu+1,nu+1,self.options.xcalpha*K_a[mu][nu],self.options.xcalpha*K_b[mu][nu]))
                  f1.write("\n")
                f1.close()  

                f1 = open("root_"+str(n)+"_dens_"+str(self.itter),"w")
                for mu in range(len(D_a)):
                  for nu in range(len(D_a[mu])):
                    f1.write("%i %i %20.15f %20.15f\n"%(mu+1,nu+1,D_a[mu][nu],D_b[mu][nu]))
                  f1.write("\n")
                f1.close()  

                f1 = open("root_"+str(n)+"_coul_"+str(self.itter),"w")
                for mu in range(len(jao_a)):
                  for nu in range(len(jao_a[mu])):
                    f1.write("%i %i %20.15f \n"%(mu+1,nu+1,jao_a[mu][nu]))
                  f1.write("\n")
                f1.close()  

                f1 = open("root_"+str(n)+"_exch_"+str(self.itter),"w")
                for mu in range(len(kao_a)):
                  for nu in range(len(kao_a[mu])):
                    f1.write("%i %i %20.15f %20.12f \n"%(mu+1,nu+1,kao_a[mu][nu],kao_b[mu][nu]))
                  f1.write("\n")
                f1.close()  

              toc = time.time()
              #print("  J contrinution",toc-tic,flush=True)

                #print("  Fxc contrinution",toc-tic,flush=True)
              if (self.wfn.reference == "uks"):
                s_a[:,:,n] += J_a - self.options.xcalpha*K_a
                s_b[:,:,n] += J_b - self.options.xcalpha*K_b
              else: 
                s_a[:,:,n] += J_a - K_a
                s_b[:,:,n] += J_b - K_b
        else:
            s_a += np.einsum("aibj,bjN->aiN",self.G_aibj[0],Xt_a,optimize=True)           
            s_a += np.einsum("aibj,bjN->aiN",self.G_aibj[1],Xt_b,optimize=True)           
            s_a -= np.einsum("abji,bjN->aiN",self.G_abji[0],Xt_a,optimize=True)           

            s_b += np.einsum("aibj,bjN->aiN",self.G_aibj[2],Xt_a,optimize=True)           
            s_b += np.einsum("aibj,bjN->aiN",self.G_aibj[3],Xt_b,optimize=True)           
            s_b -= np.einsum("abji,bjN->aiN",self.G_abji[1],Xt_b,optimize=True)           

#        print(s_a[:,:,0])
#        print("")
#        print(s_b[:,:,0])
#        exit(0)

        s_a = s_a.reshape(self.nov_a,ndim)
        s_b = s_b.reshape(self.nov_b,ndim)
        #print(s_a)
        #print("")
        #print(s_b)
        #exit(0)
        toc = time.time()
        #print("Sigma build",toc-tic)
        return np.concatenate((s_a,s_b),axis=0)
    
    def __init__(self, wfn):

        tic = time.time()
        #begin Davidson
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

        self.S = self.wfn.S.real
        self.nroots = self.options.nroots
        self.maxdim = 50 * self.nroots
        self.maxiter = 50

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

        self.in_core = self.options.tdscf_in_core
        if self.in_core is True:
          self.G_aibj = self.wfn.G_array[0]
          self.G_abji = self.wfn.G_array[1]

        self.itter = -1
       
        toc = time.time() 
        print("Initialize Davidson",toc-tic) 


    def compute(self):
        #print(self.cvs)
        #return
        print("\n")
        print("    Configuration Interaction Singles") 
        print("    ---------------------------------") 
        print("\n")

        self.tic = time.time()
  
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

          self.Cmo_a = np.zeros((self.nbf,self.nv_a+self.no_a))
          self.Cmo_b = np.zeros((self.nbf,self.nv_b+self.no_b))
          self.Cmo_a[:,:self.no_a] = self.Co_a 
          self.Cmo_a[:,self.no_a:] = self.Cv_a 
          self.Cmo_b[:,:self.no_b] = self.Co_b 
          self.Cmo_b[:,self.no_b:] = self.Cv_b 
          
          if self.in_core is True:
            print(low,high,self.Gaibj.shape)
            self.G_aibj = self.G_aibj[:,low:high+1,:,low:high+1]
            self.G_abji = self.G_abji[:,:,low:high+1,low:high+1]

        if (self.nov_a + self.nov_b)  < self.nroots:
          print("    Resizing number of roots to %i"%(self.nov_a+self.nov_b))
          self.nroots = self.nov_a + self.nov_b 
          self.maxdim = self.nov_a + self.nov_b 

        self.is_converged = np.zeros(self.nroots,dtype=int)

        self.nov = self.nov_a + self.nov_b
        total_tic = time.time()
        dvec  = np.zeros((self.nov,self.nroots))
        trial = np.zeros((self.nov,self.maxdim))
        
   
        self.ints = self.wfn.ints_factory

        twonbf = 2*self.nbf
        occ = np.zeros((twonbf,twonbf))
        occ[:self.no_a,:self.no_a] = np.eye(self.no_a)
        occ[self.nbf:self.no_b+self.nbf,self.nbf:self.no_b+self.nbf] = np.eye(self.no_b)

#        if self.in_core is False:
#
#          #alpha-alpha
#          #self.D = np.einsum("mp,pq,nq->mn",self.Cmo,occ,np.conj(self.Cmo),optimize=True)[:hnbf,:hnbf]
#          self.D  = self.wfn.D[0]
#          self.D += self.wfn.D[1]
#          jao_a = jk.get_jk(mols=self.ints,dms=self.D,scripts='ijkl,kl->ij',intor='int2e_sph')
#
#          self.D  = self.wfn.D[0]
#          kao_a = jk.get_jk(mols=self.ints,dms=self.D,scripts='ijkl,jk->il',intor='int2e_sph')
#
#          self.D  = self.wfn.D[1]
#          kao_b = jk.get_jk(mols=self.ints,dms=self.D,scripts='ijkl,jk->il',intor='int2e_sph')
#
#          ##beta-beta
#          ##self.D = np.einsum("mp,pq,nq->mn",self.Cmo,occ,np.conj(self.Cmo),optimize=True)[hnbf:,hnbf:]
#          #jao += jk.get_jk(mols=self.ints,dms=self.D,scripts='ijkl,kl->ij',intor='int2e_sph')
#          #kao += jk.get_jk(mols=self.ints,dms=self.D,scripts='ijkl,jk->il',intor='int2e_sph')
#
#          J_a = np.einsum("ma,mn,ni->ai",self.Cv_a,jao_a,self.Co_a,optimize=True)
#          J_b = np.einsum("ma,mn,ni->ai",self.Cv_b,jao_a,self.Co_b,optimize=True)
#          K_a = np.einsum("ma,mn,ni->ai",self.Cv_a,kao_a,self.Co_a,optimize=True) 
#          K_b = np.einsum("ma,mn,ni->ai",self.Cv_b,kao_b,self.Co_b,optimize=True) 
     
        Adiag = np.zeros(self.nov)
        for a in range(self.nv_a):
          for i in range(self.no_a):
            ai = a*self.no_a+i
            Adiag[ai]  = self.F_virt_a[a][a] - self.F_occ_a[i][i]
#            if self.in_core is True:
#              Adiag[ai] += self.G_aibj[0][a][i][a][i] - self.G_abji[0][a][a][i][i]
#            else:
#              Adiag[ai] += J_a[a][i] - K_a[a][i]

        for a in range(self.nv_b):
          for i in range(self.no_b):
            ai = a*self.no_b + i + self.nov_a
            Adiag[ai]  = self.F_virt_b[a][a] - self.F_occ_b[i][i]
#            if self.in_core is True:
#              Adiag[ai] += self.G_aibj[3][a][i][a][i] - self.G_abji[1][a][a][i][i]
#            else:
#              Adiag[ai] += J_b[a][i] - K_b[a][i]

        Aidx = np.argsort(Adiag)[:self.nroots]
        trial[:,:self.nroots] = np.eye(self.nov)[:,Aidx]
        
        s = self.sigma(trial[:,:self.nroots])
        A = np.matmul(trial[:,:self.nroots].T,s[:,:self.nroots])
        #A = np.einsum("nk,km->nm",np.conj(trial[:,:self.nroots].T),s[:,:self.nroots],optimize=True)
        evals , evecs = sp.linalg.eigh(A)
        X  = np.matmul(trial[:,:self.nroots],evecs[:self.nroots,:self.nroots])
        q = self.sigma(X) - np.einsum("n,in->in",evals,X,optimize=True)
        
        trial_length = len(trial[0])
        self.conv_test(q, evals, self.nroots)
        if np.sum(self.is_converged) == self.nroots:
          print("   Eigenvalues converged!")
          if self.cvs is True:
            #expand original CI vectors to avoid problem with dimensions later
            low  = self.wfn.options.occupied[0] - 1
            high = self.wfn.options.occupied[1] - 1
            Xafull = np.zeros((self.nroots,self.nv_a,self.no_a_full))
            Xbfull = np.zeros((self.nroots,self.nv_b,self.no_b_full))
            print("    Expanding occupied space from %i to %i"%(self.no_a,self.no_a_full))
            Xa = X[:self.nov_a,:]
            Xb = X[self.nov_a:,:]
            Xa = (Xa.T).reshape(self.nroots,self.nv_a,self.no_a)
            Xb = (Xb.T).reshape(self.nroots,self.nv_b,self.no_b)
            Xafull[:,:,low:high+1] = Xa
            Xbfull[:,:,low:high+1] = Xb
            Xafull = Xafull.reshape(self.nroots,self.nov_a_full)
            Xbfull = Xbfull.reshape(self.nroots,self.nov_b_full)
            Xfull = np.zeros((self.nroots,self.nov_a_full+self.nov_b_full))
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
        else:
          #print(" ")
          ldim = self.nroots
          hdim = 2*self.nroots
          ones = np.ones((self.nov,self.nroots))


          for self.itter in range(self.maxiter):
            #print("   Iteration %i "%(itter+1),"")
        
            denom  = np.einsum("ki,i->ki",ones,evals[:self.nroots],optimize=True)
            denom -= np.einsum("k,ki->ki",Adiag,ones,optimize=True)
            denom += 1e-12 #to avoid hard zeroes
            dvec = -q/denom
        
            trial[:,ldim:hdim] = dvec
            orth_trial = sp.linalg.orth(trial[:,:hdim])
            #print("   Orthogonalization %f"%(np.sum(np.conjugate(orth_trial.T)*(orth_trial.T))/hdim))
        
            s = self.sigma(orth_trial)
            A = np.matmul(np.conjugate(orth_trial.T),s)
            evals , evecs = sp.linalg.eigh(A)
            X  = np.matmul(orth_trial,evecs[:,:self.nroots])
            q  = self.sigma(X) - np.einsum("n,in->in",evals[:self.nroots],X,optimize=True)
            self.conv_test(q, evals, self.nroots)
            trial_length = len(orth_trial[0])
            if np.sum(self.is_converged) == self.nroots:
              if self.cvs is True:
                #expand original CI vectors to avoid problem with dimensions later
                low  = self.wfn.options.occupied[0] - 1
                high = self.wfn.options.occupied[1] - 1
                Xafull = np.zeros((self.nroots,self.nv_a,self.no_a_full))
                Xbfull = np.zeros((self.nroots,self.nv_b,self.no_b_full))
                print("    Expanding occupied space from %i to %i"%(self.no_a,self.no_a_full))
                Xa = X[:self.nov_a,:]
                Xb = X[self.nov_a:,:]
                Xa = (Xa.T).reshape(self.nroots,self.nv_a,self.no_a)
                Xb = (Xb.T).reshape(self.nroots,self.nv_b,self.no_b)
                Xafull[:,:,low:high+1] = Xa
                Xbfull[:,:,low:high+1] = Xb
                Xafull = Xafull.reshape(self.nroots,self.nov_a_full)
                Xbfull = Xbfull.reshape(self.nroots,self.nov_b_full)
                Xfull = np.zeros((self.nroots,self.nov_a_full+self.nov_b_full))
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
            else:
          #    print(" ")
              ldim += self.nroots
              hdim += self.nroots
          #    print(ldim, hdim)
            if (trial_length == self.nov) or (trial_length == self.maxdim):
              print("    Maximum dimension reached. Davidson algorithm failed to converge!",flush=True)    
              if (self.nov < 50000) and (self.in_core is True):
                self.nroots = self.nov
                print("    Attempting Full Diagonalization for %i roots..."%(self.nroots),flush=True)
                A_aa  = np.einsum("ab,ij->aibj",self.F_virt_a,np.eye(self.no_a),optimize=True)
                A_aa -= np.einsum("ab,ij->aibj",np.eye(self.nv_a),self.F_occ_a,optimize=True)
                A_bb  = np.einsum("ab,ij->aibj",self.F_virt_b,np.eye(self.no_b),optimize=True)
                A_bb -= np.einsum("ab,ij->aibj",np.eye(self.nv_b),self.F_occ_b,optimize=True)

                A_aa += self.G_aibj[0]
                A_aa -= self.G_abji[0].swapaxes(1,3).swapaxes(2,3) 
                A_bb += self.G_aibj[3]
                A_bb -= self.G_abji[1].swapaxes(1,3).swapaxes(2,3) 

                A_ab = self.G_aibj[1]
                A_ba = self.G_aibj[2]

                A_aa  = A_aa.reshape(self.no_a*self.nv_a,self.no_a*self.nv_a) 
                A_ab  = A_ab.reshape(self.no_a*self.nv_a,self.no_b*self.nv_b) 
                A_ba  = A_ba.reshape(self.no_b*self.nv_b,self.no_a*self.nv_a) 
                A_bb  = A_bb.reshape(self.no_b*self.nv_b,self.no_b*self.nv_b) 

                A = np.block([[A_aa,A_ab],[A_ba,A_bb]])             
                print(A.shape)

                evals, X = sp.linalg.eigh(A)
              else:
                print("    Full dimension too large (>18 Gb) to be stored in memory",flush=True)
              return evals[:self.nroots], X
        
        total_toc = time.time()
        davidson_time = total_toc - self.tic
        
        print("   Davidson diagonalization performed in %8.5f seconds"%davidson_time)
        print("   Ratio of the full space spanned by subspace: %f"%(trial_length/self.nov))

class RGDAVIDSON():

    def conv_test(self,q, evals,nroots):
      de = np.zeros(nroots)
      for n in range(nroots):
        de[n] = np.abs(np.sum(q.T[n]))
        if de[n] < 1e-4:
          self.is_converged[n] = 1
        else:
          self.is_converged[n] = 0
      print("    Iter %5i:  %5i roots converged. Subspace size: %8i. Elapsed time: %8.2f seconds"%(self.itter+1,np.sum(self.is_converged), len(evals), time.time()-self.tic),flush=True) 
      if np.sum(self.is_converged) == self.nroots:
        print("\n    Root  Subspace  Energy   dE  converged? ",flush=True)
        for n in range(nroots):
          print("  %5i %5i    %8.4f   %8.5f %s"%(n+1, len(evals), 27.21138*evals[n],de[n],("Y" if (self.is_converged[n] == 1) else "N")),flush=True)
      return
    
    def sigma(self, Xt):
        #s = np.zeros((Xt.shape),dtype=complex)
        ndim = len(Xt[0])
       # tstart = time.time()    
        Xt = Xt.reshape(self.nv_a,self.no_a,ndim)

        s  = np.einsum("ab,bin->ain",self.F_virt,Xt,optimize=True)
        s -= np.einsum("ji,ajn->ain",self.F_occ,Xt,optimize=True)
    #    print("    One-body",tmal.get_traced_memory()[1]/1e6,flush=True) 
    #    print("    TIME: One-body contribution to sigma: %f"%(time.time()-tstart),flush=True)

        if self.in_core is False:
            nbf  = int(self.nbf/2)
            jbig = np.zeros((self.nbf,self.nbf))
            kbig = np.zeros((self.nbf,self.nbf))
            #Darray = np.zeros((4,nbf,nbf))
            for n in range(ndim):
              tmp = np.einsum("ma,ai->mi",self.Cv,Xt[:,:,n],optimize=True)
              D   = np.einsum("mi,ni->mn",tmp,self.Co,optimize=True).real
              #Darray[0] = D[:nbf,:nbf]
              #Darray[1] = D[nbf:,nbf:]
              #Darray[2] = D[:nbf,nbf:]
              #Darray[3] = D[nbf:,:nbf]
              #j, k = self.wfn.jk.get_jk(mol=self.ints,dm=Darray,hermi=0)
              jbig, kbig = self.wfn.jk.get_jk(mol=self.ints,dm=D,hermi=0)
              #jbig[:nbf,:nbf] = j[0] + j[1]
              #jbig[nbf:,nbf:] = j[0] + j[1]
              #kbig[:nbf,:nbf] = k[0]
              #kbig[nbf:,nbf:] = k[1]
              #kbig[:nbf,nbf:] = k[2]
              #kbig[nbf:,:nbf] = k[3]
              tmp = np.einsum("ma,mn->an",self.Cv,jbig,optimize=True)
              J   = np.einsum("an,ni->ai",tmp,self.Co,optimize=True).real
              tmp = np.einsum("ma,mn->an",self.Cv,kbig,optimize=True)
              K   = np.einsum("an,ni->ai",tmp,self.Co,optimize=True).real

              if (self.wfn.reference == "rgks"):
                s[:,:,n] += J - self.options.xcalpha*K
              else:
                s[:,:,n] += J - K
        else:
            s += np.einsum("aibj,bjN->aiN",self.G_aibj,Xt,optimize=True)           
            s -= np.einsum("abji,bjN->aiN",self.G_abji,Xt,optimize=True)           
       
    #    print("    MEMORY: Sigma computaion used %f mb of memory"%(tmal.get_traced_memory()[1]/1024./1024.),flush=True)
    #    print("    TIME:   Sigma computaion took %f seconds"%(time.time()-tstart),flush=True)

        return s.reshape(self.nov,ndim)
    
    def __init__(self, wfn):
       # tmal.start()
       # tic = time.time()
         
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
        self.maxdim = 50 * self.nroots
        self.maxiter = 50

        self.Cmo = 1.*self.wfn.C[0]

        self.Co = self.Cmo[:,:self.no_a]
        self.Cv = self.Cmo[:,self.no_a:]

        eps_a = self.wfn.eps[0]

        self.F_occ  = np.diag(eps_a[:self.no_a]) 
        self.F_virt = np.diag(eps_a[self.no_a:]) 

        self.in_core = self.options.tdscf_in_core
        if self.in_core is True:
          self.G_aibj = self.wfn.G_array[0]
          self.G_abji = self.wfn.G_array[1]

        self.itter = -1
    #    toc = time.time()
    #    print(" Davidson Initialization memory usage",tmal.get_traced_memory()[1]/1e6,flush=True) 
    #    tmal.stop()
    #    print(" Davidson Initialization",toc-tic)
       


    def compute(self):
        #print(self.cvs)
        #return
        print("\n")
        print("    Configuration Interaction Singles") 
        print("    ---------------------------------") 
        print("\n")

        self.tic = time.time()
    #    tmal.start()
    #    print("    Strating Sigma Build memory tracing")
  
        #atomic L edge 
        #atomic K edge
        #low = 0
        #high = 1
        
     #   mem = 0.
 
        #tstart = time.time()
        if (self.cvs is True):
          low  = self.wfn.options.occupied[0] - 1
          high = self.wfn.options.occupied[1] - 1
          print("    Restricting occupied orbitals in the window [%i,%i]"%(low+1,high+1),flush=True)
          self.no_a = int(high-low+1)
          self.F_occ = self.F_occ[low:high+1,low:high+1]
          self.Co      = self.Co[:,low:high+1]
          self.nov = self.no_a * self.nv_a
          print("    Reducing the number of nov pairs from %i to %i"%(self.nov_a_full,self.nov))

          #if self.nov < self.nroots:
          #  print("    Resizing number of roots to %i"%self.nov)
          #  self.nroots = self.nov 
          #  self.maxdim = 50*self.nov 

          self.Cmo = np.zeros((self.nbf,self.nv_a+self.no_a))
          self.Cmo[:,:self.no_a] = self.Co 
          self.Cmo[:,self.no_a:] = self.Cv 
          
          if self.in_core is True:
            self.G_aibj = self.G_aibj[:,low:high+1,:,low:high+1]
            self.G_abji = self.G_abji[:,:,low:high+1,low:high+1]

      #  print("    TIME: CVS block: %f"%(time.time()-tstart),flush=True)
       
     #   print(" After CVS call",tmal.get_traced_memory()[1]/1e6,flush=True) 
        if self.nov < self.nroots:
          print("    WARNING: Resizing number of roots to %i"%self.nov)
          self.nroots = self.nov 
          self.maxdim = 50*self.nov 

        self.is_converged = np.zeros(self.nroots,dtype=int)

        total_tic = time.time()
        dvec  = np.zeros((self.nov,self.nroots))
        trial = np.zeros((self.nov,self.maxdim))
        
     #   mem += sys.getsizeof(dvec) + sys.getsizeof(trial)
    #    print(" Before integrals assignment",tmal.get_traced_memory()[1]/1e6,flush=True) 
   
        self.ints = self.wfn.ints_factory
    #    print(" After  integrals assignment",tmal.get_traced_memory()[1]/1e6,flush=True) 
    #    print("    MEMORY: The size of ints object is %f mb"%(sys.getsizeof(self.ints)),flush=True)

        occ = np.zeros((self.nv_a+self.no_a,self.nv_a+self.no_a))
        occ[:self.no_a,:self.no_a] = np.eye(self.no_a)
   #     mem += sys.getsizeof(occ)
   #     print(" Before incore block",tmal.get_traced_memory()[1]/1e6,flush=True) 

        if self.in_core is False:
   #       print(" Beginning of incore block",tmal.get_traced_memory()[1]/1e6,flush=True) 
        #  tstart = time.time()

          nbf = self.nbf//2

          jbig = np.zeros((self.nbf,self.nbf))
          kbig = np.zeros((self.nbf,self.nbf))
          D   = self.wfn.D[0]
          #Daa = D[:nbf,:nbf]
          #Dbb = D[nbf:,nbf:]
          #Dab = D[:nbf,nbf:]
          #Dba = D[nbf:,:nbf]
          #Darray = [Daa,Dbb,Dab,Dba]
   #       print(" Before j/k call",tmal.get_traced_memory()[1]/1e6,flush=True) 
          #j, k = self.wfn.jk.get_jk(mol=self.ints,dm=Darray,hermi=0)
          jbig, kbig = self.wfn.jk.get_jk(mol=self.ints,dm=D,hermi=0)
   #       print(" After j/k call",tmal.get_traced_memory()[1]/1e6,flush=True) 
          #jbig[:nbf,:nbf] = j[0].real + j[1].real
          #jbig[nbf:,nbf:] = j[0].real + j[1].real
          #kbig[:nbf,:nbf] = k[0].real
          #kbig[nbf:,nbf:] = k[1].real
          #kbig[:nbf,nbf:] = k[2].real
          #kbig[nbf:,:nbf] = k[3].real
   #       mem += sys.getsizeof(jbig) + sys.getsizeof(kbig) + sys.getsizeof(D) + sys.getsizeof(Daa) + sys.getsizeof(Dab) + sys.getsizeof(Dba) + sys.getsizeof(Dbb) + sys.getsizeof(Darray) + sys.getsizeof(j) + sys.getsizeof(k)
   #       print(" After j/kbig reconstruction",tmal.get_traced_memory()[1]/1e6,flush=True) 
          tmp = np.einsum("ma,mn->an",self.Cv,jbig,optimize=True)
          J   = np.einsum("an,ni->ai",tmp,self.Co,optimize=True)
   #       print(" After J MO transformation",tmal.get_traced_memory()[1]/1e6,flush=True) 
   #       mem += sys.getsizeof(tmp) + sys.getsizeof(J)
          tmp = np.einsum("ma,mn->an",self.Cv,kbig,optimize=True)
          K   = np.einsum("an,ni->ai",tmp,self.Co,optimize=True)
   #       print(" After K MO transformation",tmal.get_traced_memory()[1]/1e6,flush=True) 
   #       mem += sys.getsizeof(tmp) + sys.getsizeof(K)
   #       print("    TIME: First JK evaluation: %f"%(time.time()-tstart),flush=True)


    #    print(" Before diagonal population",tmal.get_traced_memory()[1]/1e6,flush=True) 
        Adiag = np.zeros(self.nov)
        for a in range(self.nv_a):
          for i in range(self.no_a):
            ai = a*self.no_a+i
            Adiag[ai]  = self.F_virt[a][a] - self.F_occ[i][i]
            if self.in_core is True:
              Adiag[ai] += self.G_aibj[a][i][a][i] - self.G_abji[a][a][i][i]
            else:
              Adiag[ai] += J[a][i] - K[a][i]
  #      print(" After diagonal population",tmal.get_traced_memory()[1]/1e6,flush=True) 

        Aidx = np.argsort(Adiag)[:self.nroots]
        trial[:,:self.nroots] = np.eye(self.nov)[:,Aidx]
    
  #      mem += sys.getsizeof(Aidx)
    
        s = self.sigma(trial[:,:self.nroots])
  #      tstart = time.time()
        A = np.matmul(np.conjugate(trial[:,:self.nroots].T),s[:,:self.nroots])
        #A = np.einsum("nk,km->nm",np.conj(trial[:,:self.nroots].T),s[:,:self.nroots],optimize=True) 
        evals , evecs = sp.linalg.eigh(A)
        X  = np.matmul(trial[:,:self.nroots],evecs[:self.nroots,:self.nroots])
        q = self.sigma(X) - np.einsum("n,in->in",evals,X)
  #      mem += sys.getsizeof(s) + sys.getsizeof(A) + sys.getsizeof(evals) + sys.getsizeof(evecs) + sys.getsizeof(X) + sys.getsizeof(q)
  #      print("    TIME: First A transformation and diagonalization: %f"%(time.time()-tstart),flush=True)
        
        trial_length = len(trial[0])
        self.conv_test(q, evals, self.nroots)
 #       print(" After trial evaluation",tmal.get_traced_memory()[1]/1e6,flush=True) 
        if np.sum(self.is_converged) == self.nroots:
          print("   Eigenvalues converged!")
          if self.cvs is True:
            #expand original CI vectors to avoid problem with dimensions later
            low  = self.wfn.options.occupied[0] - 1
            high = self.wfn.options.occupied[1] - 1
            Xfull = np.zeros((self.nroots,self.nv_a,self.no_a_full))
            print("    Expanding occupied space from %i to %i"%(self.no_a,self.no_a_full))
            X = (X.T).reshape(self.nroots,self.nv_a,self.no_a)
            Xfull[:,:,low:high+1] = X
            Xfull = Xfull.reshape(self.nroots,self.nov_a_full)
   #         print(Xfull.shape)  
            self.Cmo = self.wfn.C[0]
            self.no_a = self.no_a_full
            self.nov = self.nov_a_full
            return evals[:self.nroots], Xfull.T
          else:
            return evals[:self.nroots], X
        else:
          #print(" ")
          ldim = self.nroots
          hdim = 2*self.nroots
          ones = np.ones((self.nov,self.nroots))
  #        mem += sys.getsizeof(ones)
  #        print("    MEMORY: Pre-iteration sigma build used %f mb of memory"%(mem/1024./1024.),flush=True)
          for self.itter in range(self.maxiter):
            #print("   Iteration %i "%(itter+1),"")
        
 #           tstart = time.time()
            denom  = np.einsum("ki,i->ki",ones,evals[:self.nroots])
            denom -= np.einsum("k,ki->ki",Adiag,ones)
            denom += 1e-12 #to avoid hard zeroes
            dvec = -q/denom
        
            trial[:,ldim:hdim] = dvec
            orth_trial = sp.linalg.orth(trial[:,:hdim])
            #print("   Orthogonalization %f"%(np.sum(np.conjugate(orth_trial.T)*(orth_trial.T))/hdim))
        
            s = self.sigma(orth_trial)
            A = np.matmul(np.conjugate(orth_trial.T),s)
            evals , evecs = sp.linalg.eigh(A)
            X  = np.matmul(orth_trial,evecs[:,:self.nroots])
            q  = self.sigma(X) - np.einsum("n,in->in",evals[:self.nroots],X)
            self.conv_test(q, evals, self.nroots)
  #          print("    TIME: In-loop A transformation and diagonalization: %f"%(time.time()-tstart),flush=True)
            trial_length = len(orth_trial[0])
            if np.sum(self.is_converged) == self.nroots:
              if self.cvs is True:
                #expand original CI vectors to avoid problem with dimensions later
                low  = self.wfn.options.occupied[0] - 1
                high = self.wfn.options.occupied[1] - 1
                Xfull = np.zeros((self.nroots,self.nv_a,self.no_a_full))
                print("    Expanding occupied space from %i to %i"%(self.no_a,self.no_a_full))
                X = (X.T).reshape(self.nroots,self.nv_a,self.no_a)
                Xfull[:,:,low:high+1] = X
                Xfull = Xfull.reshape(self.nroots,self.nov_a_full)
   #             print(Xfull.shape)  
                self.Cmo = self.wfn.C[0]
                self.no_a = self.no_a_full
                self.nov = self.nov_a_full
                return evals[:self.nroots], Xfull.T
              else:
                return evals[:self.nroots], X
            else:
          #    print(" ")
              ldim += self.nroots
              hdim += self.nroots
    #          print(ldim, hdim)
            if (trial_length == self.maxdim):     
              print("    Length of trial vectors reached %i"%trial_length,flush=True)         
              print(self.nov, self.maxdim)
              print("    Maximum dimension reached. Davidson algorithm failed to converge!",flush=True)    
              #if (self.nov < 50000):
              #  self.nroots = self.nov
              #  print("    Attempting Full Diagonalization for %i roots..."%(self.nroots),flush=True)
              #  A  = np.einsum("ab,ij->aibj",self.F_virt,np.eye(self.no_a),optimize=True)
              #  A -= np.einsum("ab,ij->aibj",np.eye(self.nv_a),self.F_occ,optimize=True)
              #  A += self.G_aibj
              #  A -= self.G_abji.swapaxes(1,3).swapaxes(2,3) 
              #  A  = A.reshape(self.no_a*self.nv_a,self.no_a*self.nv_a) 
              #  evals, X = sp.linalg.eigh(A)
              #else:
              #  print("    Full dimension too large (>18 Gb) to be stored in memory",flush=True)
              return evals[:self.nroots], X
        
     #   tmal.stop()
    #    total_toc = time.time()
    #    davidson_time = total_toc - tic
    #    
    #    print("   Davidson diagonalization performed in %8.5f seconds"%davidson_time)
    #    print("   Ratio of the full space spanned by subspace: %f"%(trial_length/self.nov))
