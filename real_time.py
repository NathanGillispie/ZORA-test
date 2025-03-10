import numpy as np
import scipy as sp
import time

np.set_printoptions(precision=4, linewidth=200, suppress=True)

class RTTDSCF():

   def __init__(self,wfn):
     self.wfn = wfn
     self.options = wfn.options
     self.ints = wfn.ints_factory
     self.G_AO = self.ints.intor('int2e')
     self.options.total_time_steps = 100000
     self.options.time_step = 0.1

     print("    Initializing RT-TDSCF Class")

   def external_perturbation(self,t):
     Et = np.cos(self.options.external_field_frequency * t) * self.options.maximum_field_amplitude * np.exp(-2.5*(t-0.1)**2)
     #Et = np.sin(28.48858/27.21138 * t) * self.options.maximum_field_amplitude 


     return Et

   def LJKresponse(self,Dov,Dvo):
 

     JKov  = 2.*np.einsum("iajb,bj->ia",self.Iovov,Dvo)
     JKov += 2.*np.einsum("iabj,jb->ia",self.Iovvo,Dov)
     JKov -= 1.*np.einsum("ibja,bj->ia",self.Iovov,Dvo)
     JKov -= 1.*np.einsum("ijba,jb->ia",self.Ioovv,Dov)
     JKov += 2.*np.einsum("iajj,jj->ia",self.Iovoo,self.Doo)
     JKov -= 1.*np.einsum("ijja,jj->ia",self.Iooov,self.Doo)


     JKvo  = 2.*np.einsum("aijb,bj->ai",self.Ivoov,Dvo)
     JKvo += 2.*np.einsum("aibj,jb->ai",self.Ivovo,Dov)
     JKvo -= 1.*np.einsum("abji,bj->ai",self.Ivvoo,Dvo)
     JKvo -= 1.*np.einsum("ajbi,jb->ai",self.Ivovo,Dov)
     JKvo += 2.*np.einsum("aijj,jj->ai",self.Ivooo,self.Doo)
     JKvo -= 1.*np.einsum("ajji,jj->ai",self.Ivooo,self.Doo)

     return JKov, JKvo



   def JKresponse(self,Dt):
#     C = self.wfn.C[0]
#
#     D = np.einsum("mi,ij,nj->mn",C,Dt,C,optimize=True)
     if (self.wfn.reference == "rks"):
       
#       Vt  = self.wfn.jk.get_veff(self.ints,2.*D.real) + 1j*self.wfn.jk.get_veff(self.ints,2.*D.imag)
  
        J = np.einsum("pqrs,sr->pq",self.G,Dt)
        K = np.einsum("pqrs,qr->ps",self.G,Dt)

        return 2.*J - K

       #return np.einsum("mi,mn,nj->ij",C,Vt,C,optimize=True)

     else:
        exit("ERROR: Reference not yet implemented!")

#     Jaa, Kaa = wfn.jk.get_jk(dm=D[0])
#     Jbb, Kbb = wfn.jk.get_jk(dm=D[1])
#     Jab, _ = wfn.jk.get_jk(dm=D[1])
#     Jba, _ = wfn.jk.get_jk(dm=D[0])
#     Ka = Kaa
#     Kb = Kbb
#     Ja = Jaa + Jab
#     Jb = Jbb + Jba
#     Fa = wfn.T + wfn.Vne + Ja - Ka
#     Fb = wfn.T + wfn.Vne + Jb - Kb
   
#     return [Fa,Fb]

   def ft(self,f,dt):
      w = np.arange(0.01,1.5,0.001)
      fw = np.zeros(len(w),dtype=complex)
      for i, wi in enumerate(w):
          for tindex in range(len(f)):
              t = tindex*dt
              fw[i] += f[tindex] * np.exp(-1j*wi*t-0.05*t) * dt
      return w , fw


   def propagate_rk4(self):
   
      print("\n    Real-Time Hartree-Fock")
   
      C = self.wfn.C[0]
      D = self.wfn.D[0]
      S = self.wfn.S

      F0 = C.T@(self.wfn.T + self.wfn.Vne)@C  
      D0 = C.T@S@D@S@C
      mu0 = C.T@self.wfn.mu[self.options.external_field_polarization]@C
      self.G = np.einsum("Mp,Nq,Lr,Ss,MNLS->pqrs",C,C,C,C,self.G_AO,optimize=True)




#      with np.printoptions(precision=3,suppress =True):
##        print(D0)
#         print(occup)
#         print(np.einsum("p,pq->pq",occup,np.eye(len(occup))))

      occup = np.diag(D0)
      eta = np.zeros(F0.shape) 
      for p in range(len(occup)): 
        for q in range(len(occup)): 
          eta[p][q] = occup[p] - occup[q]

      t_array  = np.zeros(self.options.total_time_steps)
      E_array  = np.zeros(self.options.total_time_steps)
      mu_array = np.zeros(self.options.total_time_steps)
      lmu_array = np.zeros(self.options.total_time_steps)
    
      D = D0 + 0j*D0
      tic = time.time()
      for ti in range(self.options.total_time_steps):
        t_array[ti] = ti*self.options.time_step   
        t = t_array[ti]

        # compute k1
        field = self.external_perturbation(t) #returns array (E, -mu*E)
        E_array[ti] = field
        F  = F0 + self.JKresponse(D) -mu0*field
        k1 = -1j * (F@D - D@F)
        temp  = D + 0.5*k1*self.options.time_step 

        # compute k2
        field = self.external_perturbation(t+0.5*self.options.time_step) #returns array (E, -mu*E)
        F  = F0 + self.JKresponse(temp) - mu0*field
        k2 = -1j * (F@temp - temp@F)
        temp  = D + 0.5*k2*self.options.time_step 

        # compute k3
        F  = F0 + self.JKresponse(temp) - mu0*field
        k3 = -1j * (F@temp - temp@F)
        temp  = D + k3*self.options.time_step 

        # compute k4
        field = self.external_perturbation(t+1.0*self.options.time_step) #returns array (E, -mu*E)
        F  = F0 + self.JKresponse(temp) - mu0*field 
        k4 = -1j * (F@temp - temp@F)

        D  += (k1 + 2.*k2 + 2.*k3 + k4) * self.options.time_step/6. 

        mu_array[ti] = np.trace(mu0@D).real - np.trace(mu0@D0)
   
      toc = time.time()
      print("Propagation Complete in %f seconds. Starting Fourier transform"%(toc-tic),flush=True)

      #plt.plot(t_array,mu_array)
      #plt.show()

      fw = np.fft.rfft(np.einsum("i,i->i",mu_array,np.exp(-0.05*t_array),optimize=True), n = len(mu_array))
      w = np.fft.rfftfreq(mu_array.shape[-1], d=self.options.time_step)*27.21139*(2.*np.pi)

      print("Fast Fourier Tranform Finalized",flush=True)

      return w, fw


   def propagate_simpson(self):
   
      print("\n    Real-Time Hartree-Fock")
   
      C = self.wfn.C[0]
      D = self.wfn.D[0]
      S = self.wfn.S

      F0 = C.T@(self.wfn.T + self.wfn.Vne)@C  
      D0 = C.T@S@D@S@C
      mu0 = C.T@self.wfn.mu[self.options.external_field_polarization]@C
      self.G = np.einsum("Mp,Nq,Lr,Ss,MNLS->pqrs",C,C,C,C,self.G_AO,optimize=True)

      t_array  = np.zeros(self.options.total_time_steps)
      E_array  = np.zeros(self.options.total_time_steps)
      mu_array = np.zeros(self.options.total_time_steps)
      lmu_array = np.zeros(self.options.total_time_steps)
   
      dt = self.options.time_step
 
      D = D0 + 0j*D0
      F = F0 + 0j*F0
      tic = time.time()
      for ti in range(self.options.total_time_steps):
        t_array[ti] = ti*self.options.time_step   
        t = t_array[ti]

        # compute k1
        field = self.external_perturbation(t) #returns array (E, -mu*E)
        F1  = F0 + 2.*np.einsum("pqrs,sr->pq",self.G,D) 
        F1 -= np.einsum("pqrs,qr->ps",self.G,D) + mu0*field

        F2  = F0 + 2.*np.einsum("pqrs,sr->pq",self.G,D-0.5j*(F1@D-D@F1)*dt) 
        F2 -= np.einsum("pqrs,qr->ps",self.G,D-0.5j*(F1@D-D@F1)*dt) 
        F2 -= mu0*self.external_perturbation(t+0.5*dt)
      
        F3  = F0 + 2.*np.einsum("pqrs,sr->pq",self.G,D-1j*(F1@D-D@F1)*dt) 
        F3 -= np.einsum("pqrs,qr->ps",self.G,D-1j*(F1@D-D@F1)*dt) 
        F3 -= mu0*self.external_perturbation(t+dt)
     
        F = dt*(F1 + 4.*F2 + F3)/6.

        D -= 1j*(D@F - F@D)
        
        mu_array[ti] = np.trace(mu0@D).real - np.trace(mu0@D0)
   
      toc = time.time()
      print("Propagation Complete in %f seconds. Starting Fourier transform"%(toc-tic),flush=True)

      #plt.plot(t_array,mu_array)
      #plt.show()

      fw = np.fft.rfft(np.einsum("i,i->i",mu_array,np.exp(-0.05*t_array),optimize=True), n = len(mu_array))
      w = np.fft.rfftfreq(mu_array.shape[-1], d=self.options.time_step)*27.21139*(2.*np.pi)

      print("Fast Fourier Tranform Finalized",flush=True)

      return w, fw

#   def propagate_simpson(self):
#   
#      print("\n    Real-Time Hartree-Fock")
#   
#      C = self.wfn.C[0]
#      D = self.wfn.D[0]
#      S = self.wfn.S
#
#      self.G = np.einsum("Mp,Nq,Lr,Ss,MNLS->pqrs",C,C,C,C,self.G_AO,optimize=True)
##      with np.printoptions(precision=3,suppress =True):
###        print(D0)
##         print(occup)
##         print(np.einsum("p,pq->pq",occup,np.eye(len(occup))))
#
#      nbf = self.wfn.nbf
#      no = self.wfn.nel[0]
#      nv = nbf - no
#
#      Cv = C[:,no:]
#      Co = C[:,:no]
#
#      D0 = C.T@S@D@S@C
#
#      self.Doo = D0[:no,:no]
#
#      F0ov = Co.T@(self.wfn.T + self.wfn.Vne)@Cv  
#      F0vo = Cv.T@(self.wfn.T + self.wfn.Vne)@Co  
#      mu0vo = Cv.T@self.wfn.mu[self.options.external_field_polarization]@Co
#      mu0ov = Co.T@self.wfn.mu[self.options.external_field_polarization]@Cv
#
#      self.Iovoo = np.einsum("Mi,Na,Lj,Sk,MNLS->iajk",Co,Cv,Co,Co,self.G_AO,optimize=True)
#      self.Iooov = np.einsum("Mi,Nj,Lk,Sa,MNLS->ijka",Co,Co,Co,Cv,self.G_AO,optimize=True)
#      self.Ivooo = np.einsum("Ma,Ni,Lj,Sk,MNLS->aijk",Cv,Co,Co,Co,self.G_AO,optimize=True)
#
#      self.Iovov = np.einsum("Mi,Na,Lj,Sb,MNLS->iajb",Co,Cv,Co,Cv,self.G_AO,optimize=True)
#      self.Iovvo = np.einsum("Mi,Na,Lb,Sj,MNLS->iabj",Co,Cv,Cv,Co,self.G_AO,optimize=True)
#      self.Ioovv = np.einsum("Mi,Nj,La,Sb,MNLS->ijab",Co,Co,Cv,Cv,self.G_AO,optimize=True)
#
#      self.Ivoov = np.einsum("Ma,Ni,Lj,Sb,MNLS->aijb",Cv,Co,Co,Cv,self.G_AO,optimize=True)
#      self.Ivovo = np.einsum("Ma,Ni,Lb,Sj,MNLS->aibj",Cv,Co,Cv,Co,self.G_AO,optimize=True)
#      self.Ivvoo = np.einsum("Ma,Nb,Li,Sj,MNLS->abij",Cv,Cv,Co,Co,self.G_AO,optimize=True)
#
#      t_array  = np.zeros(self.options.total_time_steps)
#      E_array  = np.zeros(self.options.total_time_steps)
#      mu_array = np.zeros(self.options.total_time_steps)
#    
#      D = D0 + 0j*D0
#
#      Dov = np.zeros((no,nv),dtype=complex)
#      Dvo = np.zeros((nv,no),dtype=complex)
#
#      Fov = F0ov + 0j*F0ov
#      Fvo = F0vo + 0j*F0vo
#
#      dt = self.options.time_step
#      tic = time.time()
#      for ti in range(self.options.total_time_steps):
#        t_array[ti] = ti*self.options.time_step   
#        t = t_array[ti]
#
#        
#       # field = self.external_perturbation(t) #returns array (E, -mu*E)
#       # E_array[ti] = field
#       # JKov, JKvo = self.LJKresponse(Dov,Dvo)
#       # Fov  += (JKov - mu0ov*field)*self.options.time_step
#       # Fvo  += (JKvo - mu0vo*field)*self.options.time_step
#
#       # Dov  += 1j*Fov 
#       # Dvo  -= 1j*Fvo 
#
#        # compute k1
#        field = self.external_perturbation(t) #returns array (E, -mu*E)
#        E_array[ti] = field
#        JKov, JKvo = self.LJKresponse(Dov,Dvo)
#        Fov  = F0ov + JKov - mu0ov*field
#        Fvo  = F0vo + JKvo - mu0vo*field
#        k1ov =  1j * Fov
#        k1vo = -1j * Fvo
#        temp_ov  = Dov + 0.5*k1ov*self.options.time_step 
#        temp_vo  = Dvo + 0.5*k1vo*self.options.time_step 
#
#        # compute k2
#        field = self.external_perturbation(t+0.5*self.options.time_step) #returns array (E, -mu*E)
#        JKov, JKvo = self.LJKresponse(temp_ov,temp_vo)
#        Fov  = F0ov + JKov - mu0ov*field
#        Fvo  = F0vo + JKvo - mu0vo*field
#        k2ov =  1j * Fov
#        k2vo = -1j * Fvo
#        temp_ov  = Dov + 0.5*k2ov*self.options.time_step 
#        temp_vo  = Dvo + 0.5*k2vo*self.options.time_step 
#
#        # compute k3
#        JKov, JKvo = self.LJKresponse(temp_ov,temp_vo)
#        Fov  = F0ov + JKov - mu0ov*field
#        Fvo  = F0vo + JKvo - mu0vo*field
#        k3ov =  1j * Fov
#        k3vo = -1j * Fvo
#        temp_ov  = Dov + 1.0*k3ov*self.options.time_step 
#        temp_vo  = Dvo + 1.0*k3vo*self.options.time_step 
#
#        # compute k4
#        field = self.external_perturbation(t+1.0*self.options.time_step) #returns array (E, -mu*E)
#        JKov, JKvo = self.LJKresponse(temp_ov,temp_vo)
#        Fov  = F0ov + JKov - mu0ov*field
#        Fvo  = F0vo + JKvo - mu0vo*field
#        k4ov =  1j * Fov
#        k4vo = -1j * Fvo
#
#        Dov  += (k1ov + 2.*k2ov + 2.*k3ov + k4ov) * self.options.time_step/6. 
#        Dvo  += (k1vo + 2.*k2vo + 2.*k3vo + k4vo) * self.options.time_step/6. 
#
#        #with np.printoptions(precision=8,suppress =True):
#        #  print(Dov-Dvo.T.conj())
#
#        mu_array[ti]  = (np.trace(mu0ov@Dvo) - np.trace(mu0vo@Dov)).real
#  
# 
#      toc = time.time()
#      print("Propagation Complete in %f seconds. Starting Fourier transform"%(toc-tic),flush=True)
#
#      #plt.plot(t_array,mu_array)
#      #plt.show()
#
#      fw = np.fft.rfft(np.einsum("i,i->i",mu_array,np.exp(-0.05*t_array),optimize=True), n = len(mu_array))
#      w = np.fft.rfftfreq(mu_array.shape[-1], d=self.options.time_step)*27.21139*(2.*np.pi)
#
#      print("Fast Fourier Tranform Finalized",flush=True)
#
#      return w, fw

