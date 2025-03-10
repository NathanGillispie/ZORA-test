import numpy as np
import time
from pyscf import dft
import io_utils
import os


class XC():

    def __init__(self,wfn):
        self.wfn = wfn
        self.ints_factory = self.wfn.ints_factory
        self.options = wfn.options
        self.no_a  = int(wfn.nel[0])
        if self.wfn.reference == "uks":
          self.no_b  = int(wfn.nel[1])
        ##old style
        #self.exc_derivs = wfn.exc_derivs
        #self.ao_val = wfn.ao_val
        #self.weights = wfn.weights
        #self.rho_val = wfn.rho
        #new style
        self.wfn.jk.grids.level = self.options.grid_level
        self.grid  = self.wfn.jk.grids.build()
        self.points  = self.grid.coords
        self.weights = self.grid.weights
        self.options.xctype = dft.libxc.xc_type(self.options.xc)
        self.options.xcalpha = dft.libxc.hybrid_coeff(self.options.xc)
        npoints = len(self.points)
        #print("    Number of grid points: %i"%npoints)
        batch_size = self.options.batch_size
        excess = npoints%batch_size
        nbatches = (npoints-excess)//batch_size
        #print("    Number of batches: %i"%(nbatches+1))
        #print("    Maximum Batch Size: %i"%batch_size)
        #print("    Memory estimation: %8.4f mb"%(batch_size*self.wfn.nbf*6*8/1024./1024.))
        work_dir = os.getcwd()
        try:
          os.mkdir("ao_gridpoints")
          ao_grid_dir = work_dir+"/ao_gridpoints"
          self.pid = ao_grid_dir+"/"+self.options.inputfile.split(".")[0]
        except:
          ao_grid_dir = work_dir+"/ao_gridpoints"
          self.pid = ao_grid_dir+"/"+self.options.inputfile.split(".")[0]
          
          #print("    AO grid points found") 

#    def __del__(self):
#        npoints = len(self.points)
#        batch_size = self.options.batch_size
#        excess = npoints%batch_size
#        nbatches = (npoints-excess)//batch_size
#        for batch in range(nbatches+1):
#          try:
#            os.remove(str(self.pid)+".ao_grid"+str(batch))
#          except:
#            continue

    
    def computeV(self,D,spin):


        #self.exc_derivs = exc_derivs   
        #self.rho_val    = rho_val   
        if (self.options.xctype=="LDA"):
          npoints = len(self.points)
          batch_size = self.options.batch_size
          excess = npoints%batch_size
          nbatches = (npoints-excess)//batch_size
          nbf = self.wfn.nbf         
          Exc = 0.
          Vxc = np.zeros((2,nbf,nbf))
          for batch in range(nbatches+1):
            low = batch*batch_size
            if batch < nbatches:
              high = low+batch_size
            else:
              high = low+excess 
            points = self.points[low:high]
            weights = self.weights[low:high] 
            phi = dft.numint.eval_ao(self.ints_factory, points, deriv=0)
            if spin == 0:         
              rho = dft.numint.eval_rho(self.ints_factory, phi,2.*D, xctype=self.options.xctype)
              exc_derivs = dft.libxc.eval_xc(self.options.xc, rho,deriv=2,spin=0)[:3]
              exc = exc_derivs[0]
              vxc = exc_derivs[1][0]
              Exc += np.einsum('r,r->',exc,rho*weights)
              Vxc[0] += np.einsum('rm,rn,r->mn',phi,phi,vxc*weights,optimize=True)
              Vxc[1] += Vxc[0]
            else: 
              rhoa = dft.numint.eval_rho(self.ints_factory, phi,D[0], xctype=self.options.xctype)
              rhob = dft.numint.eval_rho(self.ints_factory, phi,D[1], xctype=self.options.xctype)
              exc_derivs = dft.libxc.eval_xc(self.options.xc, (rhoa,rhob),deriv=2,spin=1)[:3]
              rho = [rhoa, rhob]
              exc = exc_derivs[0]
              vxc = exc_derivs[1][0]
              Exc += np.einsum('r,Sr->',exc*weights,rho) #S here is the spin label
              Vxc += np.einsum('rm,rn,rS,r->Smn',phi,phi,vxc,weights,optimize=True)
          return Exc, Vxc

        elif (self.options.xctype=="GGA"):
          npoints = len(self.points)
          batch_size = self.options.batch_size
          excess = npoints%batch_size
          nbatches = (npoints-excess)//batch_size
          nbf = self.wfn.nbf         
          Exc = 0.
          Vxc = np.zeros((2,nbf,nbf))
          for batch in range(nbatches+1):
            low = batch*batch_size
            if batch < nbatches:
              high = low+batch_size
            else:
              high = low+excess 
            points = self.points[low:high]
            weights = self.weights[low:high] 
            ao_val = dft.numint.eval_ao(self.ints_factory, points, deriv=1)
            if spin == 0: 
              rho_val = dft.numint.eval_rho(self.ints_factory, ao_val,2.*D, xctype=self.options.xctype)
              exc_derivs = dft.libxc.eval_xc(self.options.xc, rho_val,deriv=2,spin=0)[:3]
              exc = exc_derivs[0]
              vrho   = exc_derivs[1][0]
              vgamma = exc_derivs[1][1]
    
              phi  = ao_val[0]
              dphi = ao_val[1:4]
    
              rho  = rho_val[0]
              drho = rho_val[1:4]
    
              Exc += np.einsum('r,r->',exc,rho*weights,optimize=True)
    
              Vxc[0] += np.einsum('rm,rn,r->mn',phi,phi,vrho*weights,optimize=True)
              tmp  = np.einsum('xrm,xr->rm',dphi,drho,optimize=True)
              Vxc[0] += 2.*np.einsum('rm,rn,r->mn',tmp,phi,vgamma*weights,optimize=True)
              tmp  = np.einsum('rm,xr->xrm',phi,drho,optimize=True)
              Vxc[0] += 2.*np.einsum('xrm,xrn,r->mn',tmp,dphi,vgamma*weights,optimize=True)
              Vxc[1] += Vxc[0]
            else: 
              rho_val_a = dft.numint.eval_rho(self.ints_factory, ao_val,D[0], xctype=self.options.xctype)
              rho_val_b = dft.numint.eval_rho(self.ints_factory, ao_val,D[1], xctype=self.options.xctype)
              rho_val = [rho_val_a, rho_val_b]
              exc_derivs = dft.libxc.eval_xc(self.options.xc, rho_val,deriv=2,spin=1)[:3]
              exc = exc_derivs[0]
              vrho_u = exc_derivs[1][0][:,0]
              vrho_d = exc_derivs[1][0][:,1]
              vgamma_uu = exc_derivs[1][1][:,0]
              vgamma_ud = exc_derivs[1][1][:,1]
              vgamma_dd = exc_derivs[1][1][:,2]
    
              phi  = ao_val[0]
              dphi = ao_val[1:4]
    
              rho_u  = rho_val[0][0]
              drho_u = rho_val[0][1:4]
    
              rho_d  = rho_val[1][0]
              drho_d = rho_val[1][1:4]
    
              Exc += np.einsum('r,r->',exc*weights,rho_u,optimize=True) 
              Exc += np.einsum('r,r->',exc*weights,rho_d,optimize=True) 
              Vxc[0] += np.einsum('rm,rn,r,r->mn',phi,phi,vrho_u,weights,optimize=True)
              Vxc[1] += np.einsum('rm,rn,r,r->mn',phi,phi,vrho_d,weights,optimize=True)
    
              tmp  = np.einsum('xrm,xr->rm',dphi,drho_u,optimize=True)
              Vxc[0] += 2.*np.einsum('rm,rn,r->mn',tmp,phi,vgamma_uu*weights,optimize=True)
              tmp  = np.einsum('rm,xr->xrm',phi,drho_u,optimize=True)
              Vxc[0] += 2.*np.einsum('xrm,xrn,r->mn',tmp,dphi,vgamma_uu*weights,optimize=True)
              tmp  = np.einsum('xrm,xr->rm',dphi,drho_d,optimize=True)
              Vxc[0] += 1.*np.einsum('rm,rn,r->mn',tmp,phi,vgamma_ud*weights,optimize=True)
              tmp  = np.einsum('rm,xr->xrm',phi,drho_d,optimize=True)
              Vxc[0] += 1.*np.einsum('xrm,xrn,r->mn',tmp,dphi,vgamma_ud*weights,optimize=True)
    
              tmp  = np.einsum('xrm,xr->rm',dphi,drho_d,optimize=True)
              Vxc[1] += 2.*np.einsum('rm,rn,r->mn',tmp,phi,vgamma_dd*weights,optimize=True)
              tmp  = np.einsum('rm,xr->xrm',phi,drho_d,optimize=True)
              Vxc[1]+= 2.*np.einsum('xrm,xrn,r->mn',tmp,dphi,vgamma_dd*weights,optimize=True)
              tmp  = np.einsum('xrm,xr->rm',dphi,drho_u,optimize=True)
              Vxc[1]+= 1.*np.einsum('rm,rn,r->mn',tmp,phi,vgamma_ud*weights,optimize=True)
              tmp  = np.einsum('rm,xr->xrm',phi,drho_u,optimize=True)
              Vxc[1] += 1.*np.einsum('xrm,xrn,r->mn',tmp,dphi,vgamma_ud*weights,optimize=True)
          return Exc, Vxc
    
    def computeF(self,Co,Cv,spin):
        nbf = len(Co[0]) #self.wfn.nbf         
        no_a  = len(Co[0][0]) #self.no_a
        nv_a  = len(Cv[0][0]) #self.nv_a
        if self.wfn.reference == "uks":
          no_b  = len(Co[1][0]) #self.no_b
          nv_b  = len(Cv[1][0]) #self.nv_b
        else:
          no_b  = no_a
          nv_b  = nv_a
        #print(no_a, nv_a, no_b, nv_b)
        #print(self.wfn.D[0].shape)
        #print(self.wfn.D[1].shape)
        #print(Co[0].shape, Cv[0].shape)
        #print(Co[1].shape, Cv[1].shape)
        #exit(0)

        F_aibj = np.zeros((nv_a,no_a,nv_a,no_a))
        F_aiBJ = np.zeros((nv_a,no_a,nv_b,no_b))
        F_AIbj = np.zeros((nv_b,no_b,nv_a,no_a))
        F_AIBJ = np.zeros((nv_b,no_b,nv_b,no_b))

        npoints = len(self.points)
        batch_size = self.options.batch_size
        excess = npoints%batch_size
        nbatches = (npoints-excess)//batch_size

        batch_time = 0.
        ao_time = 0.
        rho_time = 0.
        rho_vo_time = 0.
        exc_derivs_time = 0.
        a_time = 0.
        b1_time = 0.
        b2_time = 0.
        c_time = 0.
        d_time = 0.
        
        for batch in range(nbatches+1):
          btic = time.time()
          low = batch*batch_size
          if batch < nbatches:
            high = low+batch_size
          else:
            high = low+excess 
          points = self.points[low:high]
          weights = self.weights[low:high] 
         
          if (self.options.xctype=="LDA"):
            phi = dft.numint.eval_ao(self.ints_factory, points, deriv=0)
            if spin == 0:         
              rho = dft.numint.eval_rho(self.ints_factory, phi,2.*self.wfn.D[0], xctype=self.options.xctype)
              exc_derivs = dft.libxc.eval_xc(self.options.xc, rho,deriv=2,spin=0)[:3]
              vxc = exc_derivs[1][0]*weights
              fxc = exc_derivs[2][0]*weights
              phi_o = np.einsum("rm,mi->ri",phi,Co,optimize=True)
              phi_v = np.einsum("rm,ma->ra",phi,Cv,optimize=True)
              rho_vo = np.einsum("ra,ri->rai",phi_v,phi_o,optimize=True)
              F_aibj += np.einsum("rai,r,rbj->aibj",rho_vo,fxc,rho_vo,optimize=True)
            else: 
              rhoa = dft.numint.eval_rho(self.ints_factory, phi,self.wfn.D[0], xctype=self.options.xctype)
              rhob = dft.numint.eval_rho(self.ints_factory, phi,self.wfn.D[1], xctype=self.options.xctype)
              exc_derivs = dft.libxc.eval_xc(self.options.xc, (rhoa,rhob),deriv=2,spin=1)[:3]
              rho = [rhoa, rhob]
              v2rho2_uu   = exc_derivs[2][0][:,0]*weights
              v2rho2_ud   = exc_derivs[2][0][:,1]*weights
              v2rho2_dd   = exc_derivs[2][0][:,2]*weights
    
              rho_u  = rho[0][0]
              rho_d  = rho[1][0]
    
              phi_o  = np.einsum("rm,mi->ri",phi,Co[0],optimize=True)
              phi_v  = np.einsum("rm,ma->ra",phi,Cv[0],optimize=True)
              rho_vo = np.einsum("ra,ri->rai",phi_v,phi_o,optimize=True)

              phi_O  = np.einsum("rm,mi->ri",phi,Co[1],optimize=True)
              phi_V  = np.einsum("rm,ma->ra",phi,Cv[1],optimize=True)
              rho_VO = np.einsum("ra,ri->rai",phi_V,phi_O,optimize=True)
    
              #alpha-alpha terms
              tmp = np.einsum("r,rls->rls",v2rho2_uu,rho_vo,optimize=True)
              F_aibj += np.einsum("rmn,rls->mnls",rho_vo,tmp,optimize=True)

              #beta-beta terms
              tmp = np.einsum("r,rls->rls",v2rho2_dd,rho_VO,optimize=True)
              F_AIBJ += np.einsum("rmn,rls->mnls",rho_VO,tmp,optimize=True)
    
              #alpha-beta terms
              tmp = np.einsum("r,rls->rls",v2rho2_ud,rho_VO,optimize=True)
              F_aiBJ += np.einsum("rmn,rls->mnls",rho_vo,tmp,optimize=True)
    
              F_AIbj  = F_aiBJ.swapaxes(0,2).swapaxes(1,3) 
    
    
          elif (self.options.xctype=="GGA"):
            ao_tic = time.time()
            ao_val = dft.numint.eval_ao(self.ints_factory, points, deriv=1)
            ao_toc = time.time()
            ao_time += ao_toc-ao_tic  
            if spin == 0: 
              rho_val = dft.numint.eval_rho(self.ints_factory, ao_val,2.*self.wfn.D[0], xctype=self.options.xctype)
              exc_derivs = dft.libxc.eval_xc(self.options.xc, rho_val,deriv=2,spin=0)[:3]
              vrho   = exc_derivs[1][0]*weights
              vgamma = exc_derivs[1][1]*weights  
    
              v2rho2      = exc_derivs[2][0]*weights
              v2rho_gamma = exc_derivs[2][1]*weights
              v2gamma2    = exc_derivs[2][2]*weights
    
              phi  = ao_val[0]
              dphi = ao_val[1:4]
    
              rho  = rho_val[0]
              drho = rho_val[1:4]
    
              phi_o  = np.einsum("rm,mi->ri",phi,Co[0],optimize=True)
              phi_v  = np.einsum("rm,ma->ra",phi,Cv[0],optimize=True)
              rho_vo = np.einsum("ra,ri->rai",phi_v,phi_o,optimize=True)
    
              dphi_o   = np.einsum("xrm,mi->xri",dphi,Co[0],optimize=True)
              dphi_v   = np.einsum("xrm,ma->xra",dphi,Cv[0],optimize=True)
              drho_vo  = np.einsum("xra,ri->xrai",dphi_v,phi_o,optimize=True)
              drho_vo += np.einsum("ra,xri->xrai",phi_v,dphi_o,optimize=True)
    
              #A term
              F_aibj += np.einsum("rai,r,rbj->aibj",rho_vo,v2rho2,rho_vo,optimize=True)
    
              #B term
              tmp = np.einsum("r,xr,rls->xrls",v2rho_gamma,drho,rho_vo,optimize=True)
              F_aibj += 2.*np.einsum("xrmn,xrls->mnls",drho_vo,tmp,optimize=True)
              
              tmp = np.einsum("r,xr,xrls->rls",v2rho_gamma,drho,drho_vo,optimize=True)
              F_aibj += 2.*np.einsum("rmn,rls->mnls",rho_vo,tmp,optimize=True)
              #GGA contribution C
              
              sigma = np.einsum("xr,xrmn->rmn",drho,drho_vo)
              F_aibj += 4.*np.einsum("rmn,r,rls->mnls",sigma,v2gamma2,sigma,optimize=True)
              
              #GGA contribution D
              F_aibj += 2.*np.einsum("xrmn,r,xrls->mnls",drho_vo,vgamma,drho_vo,optimize=True)
    
              #return 2.*F_aibj
    
            elif spin == 1: 
              rho_tic = time.time()
              rho_val_a = dft.numint.eval_rho(self.ints_factory, ao_val,self.wfn.D[0], xctype=self.options.xctype)
              rho_val_b = dft.numint.eval_rho(self.ints_factory, ao_val,self.wfn.D[1], xctype=self.options.xctype)
              rho_toc = time.time()
              rho_time += rho_toc-rho_tic  
              rho_val = [rho_val_a, rho_val_b]
              exc_derivs_tic = time.time()
              exc_derivs = dft.libxc.eval_xc(self.options.xc, rho_val,deriv=2,spin=1)[:3]
              exc_derivs_toc = time.time()
              exc_derivs_time += exc_derivs_toc-exc_derivs_tic  
              vrho_u   = exc_derivs[1][0][:,0]*weights
              vrho_d   = exc_derivs[1][0][:,1]*weights
    
              vgamma_uu = exc_derivs[1][1][:,0]*weights  
              vgamma_ud = exc_derivs[1][1][:,1]*weights  
              vgamma_dd = exc_derivs[1][1][:,2]*weights  
    
              v2rho2_uu   = exc_derivs[2][0][:,0]*weights
              v2rho2_ud   = exc_derivs[2][0][:,1]*weights
              v2rho2_dd   = exc_derivs[2][0][:,2]*weights
    
              v2rho_gamma_u_uu = exc_derivs[2][1][:,0]*weights
              v2rho_gamma_u_ud = exc_derivs[2][1][:,1]*weights
              v2rho_gamma_u_dd = exc_derivs[2][1][:,2]*weights
              v2rho_gamma_d_uu = exc_derivs[2][1][:,3]*weights
              v2rho_gamma_d_ud = exc_derivs[2][1][:,4]*weights
              v2rho_gamma_d_dd = exc_derivs[2][1][:,5]*weights
    
              v2gamma2_uu_uu = exc_derivs[2][2][:,0]*weights
              v2gamma2_uu_ud = exc_derivs[2][2][:,1]*weights
              v2gamma2_uu_dd = exc_derivs[2][2][:,2]*weights
              v2gamma2_ud_ud = exc_derivs[2][2][:,3]*weights
              v2gamma2_ud_dd = exc_derivs[2][2][:,4]*weights
              v2gamma2_dd_dd = exc_derivs[2][2][:,5]*weights
    
              phi  = ao_val[0]
              dphi = ao_val[1:4]
    
              rho_u  = rho_val[0][0]
              drho_u = rho_val[0][1:4]
              rho_d  = rho_val[1][0]
              drho_d = rho_val[1][1:4]
    
              rho_vo_tic = time.time()
              phi_o  = np.einsum("rm,mi->ri",phi,Co[0],optimize=True)
              phi_v  = np.einsum("rm,ma->ra",phi,Cv[0],optimize=True)
              rho_vo = np.einsum("ra,ri->rai",phi_v,phi_o,optimize=True)
    
              dphi_o   = np.einsum("xrm,mi->xri",dphi,Co[0],optimize=True)
              dphi_v   = np.einsum("xrm,ma->xra",dphi,Cv[0],optimize=True)
              drho_vo  = np.einsum("xra,ri->xrai",dphi_v,phi_o,optimize=True)
              drho_vo += np.einsum("ra,xri->xrai",phi_v,dphi_o,optimize=True)
    
              phi_O  = np.einsum("rm,mi->ri",phi,Co[1],optimize=True)
              phi_V  = np.einsum("rm,ma->ra",phi,Cv[1],optimize=True)
              rho_VO = np.einsum("ra,ri->rai",phi_V,phi_O,optimize=True)
    
              dphi_O   = np.einsum("xrm,mi->xri",dphi,Co[1],optimize=True)
              dphi_V   = np.einsum("xrm,ma->xra",dphi,Cv[1],optimize=True)
              drho_VO  = np.einsum("xra,ri->xrai",dphi_V,phi_O,optimize=True)
              drho_VO += np.einsum("ra,xri->xrai",phi_V,dphi_O,optimize=True)
              rho_vo_toc = time.time()
              rho_vo_time += rho_vo_toc - rho_vo_tic
    
              #alpha-alpha terms
    
              #A term
              tic = time.time()
              tmp = np.einsum("r,rls->rls",v2rho2_uu,rho_vo,optimize=True)
              F_aibj += np.einsum("rmn,rls->mnls",rho_vo,tmp,optimize=True)
              toc = time.time()
              a_time += toc-tic
    
              #B1 term
              tic = time.time()
              tmp = np.einsum("r,xr,rls->xrls",v2rho_gamma_u_uu,drho_u,rho_vo,optimize=True)
              F_aibj += 2.*np.einsum("xrmn,xrls->mnls",drho_vo,tmp,optimize=True)
              
              tmp = np.einsum("r,xr,rls->xrls",v2rho_gamma_u_ud,drho_d,rho_vo,optimize=True)
              F_aibj += 1.*np.einsum("xrmn,xrls->mnls",drho_vo,tmp,optimize=True)
              toc = time.time()
              b1_time += toc-tic
              
              #B2 term
              tic = time.time()
              tmp = np.einsum("r,xr,xrls->rls",v2rho_gamma_u_uu,drho_u,drho_vo,optimize=True)
              F_aibj += 2.*np.einsum("rmn,rls->mnls",rho_vo,tmp,optimize=True)
              
              tmp = np.einsum("r,xr,xrls->rls",v2rho_gamma_u_ud,drho_d,drho_vo,optimize=True)
              F_aibj += 1.*np.einsum("rmn,rls->mnls",rho_vo,tmp,optimize=True)
              toc = time.time()
              b2_time += toc-tic
    
              #C term
              tic = time.time()
              sigma_u = np.einsum("xr,xrmn->rmn",drho_u,drho_vo,optimize=True)
              sigma_d = np.einsum("xr,xrmn->rmn",drho_d,drho_vo,optimize=True)
              F_aibj += 4.*np.einsum("rmn,r,rls->mnls",sigma_u,v2gamma2_uu_uu,sigma_u,optimize=True)
              F_aibj += 2.*np.einsum("rmn,r,rls->mnls",sigma_u,v2gamma2_uu_ud,sigma_d,optimize=True)
              F_aibj += 2.*np.einsum("rmn,r,rls->mnls",sigma_d,v2gamma2_uu_ud,sigma_u,optimize=True)
              F_aibj += 1.*np.einsum("rmn,r,rls->mnls",sigma_d,v2gamma2_ud_ud,sigma_d,optimize=True)
              toc = time.time()
              c_time += toc-tic
    
              #D term
              tic = time.time()
              F_aibj += 2.*np.einsum("xrmn,r,xrls->mnls",drho_vo,vgamma_uu,drho_vo,optimize=True)
              toc = time.time()
              d_time += toc-tic
    
    
              #beta-beta terms
    
              #A term
              tmp = np.einsum("r,rls->rls",v2rho2_dd,rho_VO,optimize=True)
              F_AIBJ += np.einsum("rmn,rls->mnls",rho_VO,tmp,optimize=True)
    
              #B1 term
              tmp = np.einsum("r,xr,rls->xrls",v2rho_gamma_d_dd,drho_d,rho_VO,optimize=True)
              F_AIBJ += 2.*np.einsum("xrmn,xrls->mnls",drho_VO,tmp,optimize=True)
              
              tmp = np.einsum("r,xr,rls->xrls",v2rho_gamma_d_ud,drho_u,rho_VO,optimize=True)
              F_AIBJ += 1.*np.einsum("xrmn,xrls->mnls",drho_VO,tmp,optimize=True)
              
              #B2 term
              tmp = np.einsum("r,xr,xrls->rls",v2rho_gamma_d_dd,drho_d,drho_VO,optimize=True)
              F_AIBJ += 2.*np.einsum("rmn,rls->mnls",rho_VO,tmp,optimize=True)
              
              tmp = np.einsum("r,xr,xrls->rls",v2rho_gamma_d_ud,drho_u,drho_VO,optimize=True)
              F_AIBJ += 1.*np.einsum("rmn,rls->mnls",rho_VO,tmp,optimize=True)
    
              #C term
              sigma_u = np.einsum("xr,xrmn->rmn",drho_u,drho_VO,optimize=True)
              sigma_d = np.einsum("xr,xrmn->rmn",drho_d,drho_VO,optimize=True)
              F_AIBJ += 4.*np.einsum("rmn,r,rls->mnls",sigma_d,v2gamma2_dd_dd,sigma_d,optimize=True)
              F_AIBJ += 2.*np.einsum("rmn,r,rls->mnls",sigma_d,v2gamma2_ud_dd,sigma_u,optimize=True)
              F_AIBJ += 2.*np.einsum("rmn,r,rls->mnls",sigma_u,v2gamma2_ud_dd,sigma_d,optimize=True)
              F_AIBJ += 1.*np.einsum("rmn,r,rls->mnls",sigma_u,v2gamma2_ud_ud,sigma_u,optimize=True)
    
              #D term
              F_AIBJ += 2.*np.einsum("xrmn,r,xrls->mnls",drho_VO,vgamma_dd,drho_VO,optimize=True)
    
              #alpha-beta terms
    
              #A term
              tmp = np.einsum("r,rls->rls",v2rho2_ud,rho_VO,optimize=True)
              F_aiBJ += np.einsum("rmn,rls->mnls",rho_vo,tmp,optimize=True)
    
              #B1 term
              tmp = np.einsum("r,xr,rls->xrls",v2rho_gamma_d_ud,drho_d,rho_VO,optimize=True)
              F_aiBJ += 1.*np.einsum("xrmn,xrls->mnls",drho_vo,tmp,optimize=True)
              
              tmp = np.einsum("r,xr,rls->xrls",v2rho_gamma_d_uu,drho_u,rho_VO,optimize=True)
              F_aiBJ += 2.*np.einsum("xrmn,xrls->mnls",drho_vo,tmp,optimize=True)
              
              #B2 term
              tmp = np.einsum("r,xr,xrls->rls",v2rho_gamma_u_ud,drho_u,drho_VO,optimize=True)
              F_aiBJ += 1.*np.einsum("rmn,rls->mnls",rho_vo,tmp,optimize=True)
              
              tmp = np.einsum("r,xr,xrls->rls",v2rho_gamma_u_dd,drho_d,drho_VO,optimize=True)
              F_aiBJ += 2.*np.einsum("rmn,rls->mnls",rho_vo,tmp,optimize=True)
              
              #C term
              sigma_l = np.einsum("xr,xrmn->rmn",drho_u,drho_vo,optimize=True)
              sigma_r = np.einsum("xr,xrmn->rmn",drho_d,drho_VO,optimize=True)
              F_aiBJ += 4.*np.einsum("rmn,r,rls->mnls",sigma_l,v2gamma2_uu_dd,sigma_r,optimize=True)
              sigma_l = np.einsum("xr,xrmn->rmn",drho_d,drho_vo,optimize=True)
              sigma_r = np.einsum("xr,xrmn->rmn",drho_d,drho_VO,optimize=True)
              F_aiBJ += 2.*np.einsum("rmn,r,rls->mnls",sigma_l,v2gamma2_ud_dd,sigma_r,optimize=True)
              sigma_l = np.einsum("xr,xrmn->rmn",drho_u,drho_vo,optimize=True)
              sigma_r = np.einsum("xr,xrmn->rmn",drho_u,drho_VO,optimize=True)
              F_aiBJ += 2.*np.einsum("rmn,r,rls->mnls",sigma_l,v2gamma2_uu_ud,sigma_r,optimize=True)
              sigma_l = np.einsum("xr,xrmn->rmn",drho_d,drho_vo,optimize=True)
              sigma_r = np.einsum("xr,xrmn->rmn",drho_u,drho_VO,optimize=True)
              F_aiBJ += 1.*np.einsum("rmn,r,rls->mnls",sigma_l,v2gamma2_ud_ud,sigma_r,optimize=True)
    
              #D term
              F_aiBJ += 1.*np.einsum("xrmn,r,xrls->mnls",drho_vo,vgamma_ud,drho_VO,optimize=True)
    
              F_AIbj = F_aiBJ.swapaxes(0,2).swapaxes(1,3) 
          #btoc = time.time()
          #print("    Batch %s took %f seconds"%(batch,btoc-btic),flush=True)
          #batch_time += btoc-btic

       
        print("    XC Functonal Timings:",flush=True) 
        print("    Total Batch: %f"%batch_time,flush=True)
        print("    AO evaluation: %f"%ao_time,flush=True)
        print("    RHO evaluation: %f"%rho_time,flush=True)
        print("    RHO_VO evaluation: %f"%rho_vo_time,flush=True)
        print("    EXC deriv evaluation: %f"%exc_derivs_time,flush=True)
        print("    A Term: %f"%(4*a_time),flush=True)
        print("    B1 Term: %f"%(4*b1_time),flush=True)
        print("    B2 Term: %f"%(4*b2_time),flush=True)
        print("    C Term: %f"%(4*c_time),flush=True)
        print("    D Term: %f"%(4*d_time),flush=True)
        #exit(0)
    
        return F_aibj, F_aiBJ,F_AIbj,F_AIBJ


    def computeFdiag(self,Co,Cv,spin):
        nbf = len(Co[0]) #self.wfn.nbf         
        no_a  = len(Co[0][0]) #self.no_a
        nv_a  = len(Cv[0][0]) #self.nv_a
        if self.wfn.reference == "uks":
          no_b  = len(Co[1][0]) #self.no_b
          nv_b  = len(Cv[1][0]) #self.nv_b
        else:
          no_b  = no_a
          nv_b  = nv_a
        #print(no_a, nv_a, no_b, nv_b)
        #print(self.wfn.D[0].shape)
        #print(self.wfn.D[1].shape)
        #print(Co[0].shape, Cv[0].shape)
        #print(Co[1].shape, Cv[1].shape)
        #exit(0)

        F_ai = np.zeros((nv_a,no_a))
        F_AI = np.zeros((nv_b,no_b))

        npoints = len(self.points)
        batch_size = self.options.batch_size
        excess = npoints%batch_size
        nbatches = (npoints-excess)//batch_size

        batch_time = 0.
        ao_time = 0.
        rho_time = 0.
        rho_vo_time = 0.
        exc_derivs_time = 0.
        a_time = 0.
        b1_time = 0.
        b2_time = 0.
        c_time = 0.
        d_time = 0.
        
        for batch in range(nbatches+1):
          btic = time.time()
          low = batch*batch_size
          if batch < nbatches:
            high = low+batch_size
          else:
            high = low+excess 
          points = self.points[low:high]
          weights = self.weights[low:high] 
         
          if (self.options.xctype=="LDA"):
            phi = dft.numint.eval_ao(self.ints_factory, points, deriv=0)
            if spin == 0:         
              rho = dft.numint.eval_rho(self.ints_factory, phi,2.*self.wfn.D[0], xctype=self.options.xctype)
              exc_derivs = dft.libxc.eval_xc(self.options.xc, rho,deriv=2,spin=0)[:3]
              vxc = exc_derivs[1][0]*weights
              fxc = exc_derivs[2][0]*weights
              phi_o = np.einsum("rm,mi->ri",phi,Co,optimize=True)
              phi_v = np.einsum("rm,ma->ra",phi,Cv,optimize=True)
              rho_vo = np.einsum("ra,ri->rai",phi_v,phi_o,optimize=True)
              F_ai += np.einsum("rai,r,rai->ai",rho_vo,fxc,rho_vo,optimize=True)
            else: 
              rhoa = dft.numint.eval_rho(self.ints_factory, phi,self.wfn.D[0], xctype=self.options.xctype)
              rhob = dft.numint.eval_rho(self.ints_factory, phi,self.wfn.D[1], xctype=self.options.xctype)
              exc_derivs = dft.libxc.eval_xc(self.options.xc, (rhoa,rhob),deriv=2,spin=1)[:3]
              rho = [rhoa, rhob]
              v2rho2_uu   = exc_derivs[2][0][:,0]*weights
              v2rho2_ud   = exc_derivs[2][0][:,1]*weights
              v2rho2_dd   = exc_derivs[2][0][:,2]*weights
    
              rho_u  = rho[0][0]
              rho_d  = rho[1][0]
    
              phi_o  = np.einsum("rm,mi->ri",phi,Co[0],optimize=True)
              phi_v  = np.einsum("rm,ma->ra",phi,Cv[0],optimize=True)
              rho_vo = np.einsum("ra,ri->rai",phi_v,phi_o,optimize=True)

              phi_O  = np.einsum("rm,mi->ri",phi,Co[1],optimize=True)
              phi_V  = np.einsum("rm,ma->ra",phi,Cv[1],optimize=True)
              rho_VO = np.einsum("ra,ri->rai",phi_V,phi_O,optimize=True)
    
              #alpha-alpha terms
              tmp = np.einsum("r,rls->rls",v2rho2_uu,rho_vo,optimize=True)
              F_ai += np.einsum("rmn,rmn->mn",rho_vo,tmp,optimize=True)

              #beta-beta terms
              tmp = np.einsum("r,rls->rls",v2rho2_dd,rho_VO,optimize=True)
              F_AI += np.einsum("rmn,rmn->mn",rho_VO,tmp,optimize=True)
    
          elif (self.options.xctype=="GGA"):
            ao_tic = time.time()
            ao_val = dft.numint.eval_ao(self.ints_factory, points, deriv=1)
            ao_toc = time.time()
            ao_time += ao_toc-ao_tic  
            if spin == 0: 
              rho_val = dft.numint.eval_rho(self.ints_factory, ao_val,2.*self.wfn.D[0], xctype=self.options.xctype)
              exc_derivs = dft.libxc.eval_xc(self.options.xc, rho_val,deriv=2,spin=0)[:3]
              vrho   = exc_derivs[1][0]*weights
              vgamma = exc_derivs[1][1]*weights  
    
              v2rho2      = exc_derivs[2][0]*weights
              v2rho_gamma = exc_derivs[2][1]*weights
              v2gamma2    = exc_derivs[2][2]*weights
    
              phi  = ao_val[0]
              dphi = ao_val[1:4]
    
              rho  = rho_val[0]
              drho = rho_val[1:4]
    
              phi_o  = np.einsum("rm,mi->ri",phi,Co,optimize=True)
              phi_v  = np.einsum("rm,ma->ra",phi,Cv,optimize=True)
              rho_vo = np.einsum("ra,ri->rai",phi_v,phi_o,optimize=True)
    
              dphi_o   = np.einsum("xrm,mi->xri",dphi,Co,optimize=True)
              dphi_v   = np.einsum("xrm,ma->xra",dphi,Cv,optimize=True)
              drho_vo  = np.einsum("xra,ri->xrai",dphi_v,phi_o,optimize=True)
              drho_vo += np.einsum("ra,xri->xrai",phi_v,dphi_o,optimize=True)
    
              #A term
              F_ai += np.einsum("rai,r,rai->ai",rho_vo,v2rho2,rho_vo,optimize=True)
    
              #B term
              tmp = np.einsum("r,xr,rls->xrls",v2rho_gamma,drho,rho_vo,optimize=True)
              F_ai += 2.*np.einsum("xrmn,xrmn->mn",drho_vo,tmp,optimize=True)
              
              tmp = np.einsum("r,xr,xrls->rls",v2rho_gamma,drho,drho_vo,optimize=True)
              F_ai += 2.*np.einsum("rmn,rmn->mn",rho_vo,tmp,optimize=True)
              #GGA contribution C
              
              sigma = np.einsum("xr,xrmn->rmn",drho,drho_vo)
              F_ai += 4.*np.einsum("rmn,r,rmn->mn",sigma,v2gamma2,sigma,optimize=True)
              
              #GGA contribution D
              F_ai += 2.*np.einsum("xrmn,r,xrmn->mn",drho_vo,vgamma,drho_vo,optimize=True)
    
              #return 2.*F_aibj
    
            elif spin == 1: 
              rho_tic = time.time()
              rho_val_a = dft.numint.eval_rho(self.ints_factory, ao_val,self.wfn.D[0], xctype=self.options.xctype)
              rho_val_b = dft.numint.eval_rho(self.ints_factory, ao_val,self.wfn.D[1], xctype=self.options.xctype)
              rho_toc = time.time()
              rho_time += rho_toc-rho_tic  
              rho_val = [rho_val_a, rho_val_b]
              exc_derivs_tic = time.time()
              exc_derivs = dft.libxc.eval_xc(self.options.xc, rho_val,deriv=2,spin=1)[:3]
              exc_derivs_toc = time.time()
              exc_derivs_time += exc_derivs_toc-exc_derivs_tic  
              vrho_u   = exc_derivs[1][0][:,0]*weights
              vrho_d   = exc_derivs[1][0][:,1]*weights
    
              vgamma_uu = exc_derivs[1][1][:,0]*weights  
              vgamma_ud = exc_derivs[1][1][:,1]*weights  
              vgamma_dd = exc_derivs[1][1][:,2]*weights  
    
              v2rho2_uu   = exc_derivs[2][0][:,0]*weights
              v2rho2_ud   = exc_derivs[2][0][:,1]*weights
              v2rho2_dd   = exc_derivs[2][0][:,2]*weights
    
              v2rho_gamma_u_uu = exc_derivs[2][1][:,0]*weights
              v2rho_gamma_u_ud = exc_derivs[2][1][:,1]*weights
              v2rho_gamma_u_dd = exc_derivs[2][1][:,2]*weights
              v2rho_gamma_d_uu = exc_derivs[2][1][:,3]*weights
              v2rho_gamma_d_ud = exc_derivs[2][1][:,4]*weights
              v2rho_gamma_d_dd = exc_derivs[2][1][:,5]*weights
    
              v2gamma2_uu_uu = exc_derivs[2][2][:,0]*weights
              v2gamma2_uu_ud = exc_derivs[2][2][:,1]*weights
              v2gamma2_uu_dd = exc_derivs[2][2][:,2]*weights
              v2gamma2_ud_ud = exc_derivs[2][2][:,3]*weights
              v2gamma2_ud_dd = exc_derivs[2][2][:,4]*weights
              v2gamma2_dd_dd = exc_derivs[2][2][:,5]*weights
    
              phi  = ao_val[0]
              dphi = ao_val[1:4]
    
              rho_u  = rho_val[0][0]
              drho_u = rho_val[0][1:4]
              rho_d  = rho_val[1][0]
              drho_d = rho_val[1][1:4]
    
              rho_vo_tic = time.time()
              phi_o  = np.einsum("rm,mi->ri",phi,Co[0],optimize=True)
              phi_v  = np.einsum("rm,ma->ra",phi,Cv[0],optimize=True)
              rho_vo = np.einsum("ra,ri->rai",phi_v,phi_o,optimize=True)
    
              dphi_o   = np.einsum("xrm,mi->xri",dphi,Co[0],optimize=True)
              dphi_v   = np.einsum("xrm,ma->xra",dphi,Cv[0],optimize=True)
              drho_vo  = np.einsum("xra,ri->xrai",dphi_v,phi_o,optimize=True)
              drho_vo += np.einsum("ra,xri->xrai",phi_v,dphi_o,optimize=True)
    
              phi_O  = np.einsum("rm,mi->ri",phi,Co[1],optimize=True)
              phi_V  = np.einsum("rm,ma->ra",phi,Cv[1],optimize=True)
              rho_VO = np.einsum("ra,ri->rai",phi_V,phi_O,optimize=True)
    
              dphi_O   = np.einsum("xrm,mi->xri",dphi,Co[1],optimize=True)
              dphi_V   = np.einsum("xrm,ma->xra",dphi,Cv[1],optimize=True)
              drho_VO  = np.einsum("xra,ri->xrai",dphi_V,phi_O,optimize=True)
              drho_VO += np.einsum("ra,xri->xrai",phi_V,dphi_O,optimize=True)
              rho_vo_toc = time.time()
              rho_vo_time += rho_vo_toc - rho_vo_tic
    
              #alpha-alpha terms
    
              #A term
              tic = time.time()
              tmp = np.einsum("r,rls->rls",v2rho2_uu,rho_vo,optimize=True)
              F_ai += np.einsum("rmn,rmn->mn",rho_vo,tmp,optimize=True)
              toc = time.time()
              a_time += toc-tic
    
              #B1 term
              tic = time.time()
              tmp = np.einsum("r,xr,rls->xrls",v2rho_gamma_u_uu,drho_u,rho_vo,optimize=True)
              F_ai += 2.*np.einsum("xrmn,xrmn->mn",drho_vo,tmp,optimize=True)
              
              tmp = np.einsum("r,xr,rls->xrls",v2rho_gamma_u_ud,drho_d,rho_vo,optimize=True)
              F_ai += 1.*np.einsum("xrmn,xrmn->mn",drho_vo,tmp,optimize=True)
              toc = time.time()
              b1_time += toc-tic
              
              #B2 term
              tic = time.time()
              tmp = np.einsum("r,xr,xrls->rls",v2rho_gamma_u_uu,drho_u,drho_vo,optimize=True)
              F_ai += 2.*np.einsum("rmn,rmn->mn",rho_vo,tmp,optimize=True)
              
              tmp = np.einsum("r,xr,xrls->rls",v2rho_gamma_u_ud,drho_d,drho_vo,optimize=True)
              F_ai += 1.*np.einsum("rmn,rmn->mn",rho_vo,tmp,optimize=True)
              toc = time.time()
              b2_time += toc-tic
    
              #C term
              tic = time.time()
              sigma_u = np.einsum("xr,xrmn->rmn",drho_u,drho_vo,optimize=True)
              sigma_d = np.einsum("xr,xrmn->rmn",drho_d,drho_vo,optimize=True)
              F_ai += 4.*np.einsum("rmn,r,rmn->mn",sigma_u,v2gamma2_uu_uu,sigma_u,optimize=True)
              F_ai += 2.*np.einsum("rmn,r,rmn->mn",sigma_u,v2gamma2_uu_ud,sigma_d,optimize=True)
              F_ai += 2.*np.einsum("rmn,r,rmn->mn",sigma_d,v2gamma2_uu_ud,sigma_u,optimize=True)
              F_ai += 1.*np.einsum("rmn,r,rmn->mn",sigma_d,v2gamma2_ud_ud,sigma_d,optimize=True)
              toc = time.time()
              c_time += toc-tic
    
              #D term
              tic = time.time()
              F_ai += 2.*np.einsum("xrmn,r,xrmn->mn",drho_vo,vgamma_uu,drho_vo,optimize=True)
              toc = time.time()
              d_time += toc-tic
    
    
              #beta-beta terms
    
              #A term
              tmp = np.einsum("r,rls->rls",v2rho2_dd,rho_VO,optimize=True)
              F_AI += np.einsum("rmn,rmn->mn",rho_VO,tmp,optimize=True)
    
              #B1 term
              tmp = np.einsum("r,xr,rls->xrls",v2rho_gamma_d_dd,drho_d,rho_VO,optimize=True)
              F_AI += 2.*np.einsum("xrmn,xrmn->mn",drho_VO,tmp,optimize=True)
              
              tmp = np.einsum("r,xr,rls->xrls",v2rho_gamma_d_ud,drho_u,rho_VO,optimize=True)
              F_AI += 1.*np.einsum("xrmn,xrmn->mn",drho_VO,tmp,optimize=True)
              
              #B2 term
              tmp = np.einsum("r,xr,xrls->rls",v2rho_gamma_d_dd,drho_d,drho_VO,optimize=True)
              F_AI += 2.*np.einsum("rmn,rmn->mn",rho_VO,tmp,optimize=True)
              
              tmp = np.einsum("r,xr,xrls->rls",v2rho_gamma_d_ud,drho_u,drho_VO,optimize=True)
              F_AI += 1.*np.einsum("rmn,rmn->mn",rho_VO,tmp,optimize=True)
    
              #C term
              sigma_u = np.einsum("xr,xrmn->rmn",drho_u,drho_VO,optimize=True)
              sigma_d = np.einsum("xr,xrmn->rmn",drho_d,drho_VO,optimize=True)
              F_AI += 4.*np.einsum("rmn,r,rmn->mn",sigma_d,v2gamma2_dd_dd,sigma_d,optimize=True)
              F_AI += 2.*np.einsum("rmn,r,rmn->mn",sigma_d,v2gamma2_ud_dd,sigma_u,optimize=True)
              F_AI += 2.*np.einsum("rmn,r,rmn->mn",sigma_u,v2gamma2_ud_dd,sigma_d,optimize=True)
              F_AI += 1.*np.einsum("rmn,r,rmn->mn",sigma_u,v2gamma2_ud_ud,sigma_u,optimize=True)
    
              #D term
              F_AI += 2.*np.einsum("xrmn,r,xrmn->mn",drho_VO,vgamma_dd,drho_VO,optimize=True)
    
        print("    XC Functonal Timings:",flush=True) 
        print("    Total Batch: %f"%batch_time,flush=True)
        print("    AO evaluation: %f"%ao_time,flush=True)
        print("    RHO evaluation: %f"%rho_time,flush=True)
        print("    RHO_VO evaluation: %f"%rho_vo_time,flush=True)
        print("    EXC deriv evaluation: %f"%exc_derivs_time,flush=True)
        print("    A Term: %f"%(4*a_time),flush=True)
        print("    B1 Term: %f"%(4*b1_time),flush=True)
        print("    B2 Term: %f"%(4*b2_time),flush=True)
        print("    C Term: %f"%(4*c_time),flush=True)
        print("    D Term: %f"%(4*d_time),flush=True)
        #exit(0)
    
        return F_ai, F_AI
    
    
    def computeVx(self,Co,Cv,X,spin):
        nbf = self.wfn.nbf         
        if (spin==1):
          nroots = len(X[0].T) 
          no_a   = len(Co[0][0])
          nv_a   = len(Cv[0][0])
        else:
          nroots = len(X.T) 
          no_a   = len(Co[0])
          nv_a   = len(Cv[0])

        if self.wfn.reference == "uks":
          no_b   = len(Co[1][0])
          nv_b   = len(Cv[1][0])
        else:
          no_b  = no_a
          nv_b  = nv_a

        F_ai = np.zeros((nv_a,no_a,nroots))
        F_AI = np.zeros((nv_b,no_b,nroots))

        npoints = len(self.points)
        batch_size = self.options.batch_size
        excess = npoints%batch_size
        nbatches = (npoints-excess)//batch_size

        time_ao = 0.
        time1 = 0.
        time2 = 0.
        time3 = 0.
        time4 = 0.
        time5 = 0.
        timek = 0.
        for batch in range(nbatches+1):
          low = batch*batch_size
          if batch < nbatches:
            high = low+batch_size
          else:
            high = low+excess 
          points = self.points[low:high]
          weights = self.weights[low:high] 
         
          if (self.options.xctype=="LDA"):
            tic = time.time()
            try:
              ao_file = open(str(self.pid)+".ao_lda_grid"+str(batch),"rb")
              ao_val = io_utils.read_ao_grid(ao_file,"LDA")
            except:
              ao_val = dft.numint.eval_ao(self.ints_factory, points, deriv=0)
              io_utils.write_ao_grid(ao_val,str(self.pid)+".ao_lda_grid"+str(batch),"LDA")
            toc = time.time()
            time_ao = toc-tic
            #print("      AO grid evaluation took: %f"%(toc-tic))
            if spin == 0:         
              tic = time.time()
              rho = dft.numint.eval_rho(self.ints_factory, ao_val,2.*self.wfn.D[0], xctype=self.options.xctype)
              toc = time.time()
              print("      RHO grid evaluation took: %f"%(toc-tic),flush=True)
              tic = time.time()
              exc_derivs = dft.libxc.eval_xc(self.options.xc, rho,deriv=2,spin=0)[:3]
              toc = time.time()
              print("      DERIVATIVES grid evaluation took: %f"%(toc-tic),flush=True)
              tic = time.time()
              exc = exc_derivs[0]*weights
              vxc = exc_derivs[1][0]*weights
              fxc = exc_derivs[2][0]*weights
              phi = ao_val
              toc = time.time()
              #print("      ASSIGNMENTS grid evaluation took: %f"%(toc-tic))
              ktic = time.time()
              tic1 = time.time()
              phi_o = np.einsum("rm,mi->ri",phi,Co,optimize=True)
              toc1 = time.time()
              time1 += toc1-tic1
              #print("      TERM 1 took: %f"%(toc1-tic1))
              tic2 = time.time()
              phi_v = np.einsum("rm,ma->ra",phi,Cv,optimize=True)
              toc2 = time.time()
              time2 += toc2-tic2
              #print("      TERM 2 took: %f"%(toc2-tic2))
              tic3 = time.time()
              rho_vo = np.einsum("ra,ri->rai",phi_v,phi_o,optimize=True)
              toc3 = time.time()
              time3 += toc3-tic3
              #print("      TERM 3 took: %f"%(toc-tic))
              tic4 = time.time()
              Xr     = np.einsum("rai,aiN->rN",rho_vo,X,optimize=True)
              toc4 = time.time()
              time4 += toc4 - tic4
              #print("      TERM 4 took: %f"%(toc-tic))
              tic5 = time.time()
              fxcX  = np.einsum("r,rN->rN",fxc,Xr,optimize=True)
              F_ai += np.einsum("rai,rN->aiN",rho_vo,fxcX,optimize=True)
              #F_ai += np.einsum("rai,r,rN->aiN",rho_vo,fxc,Xr,optimize=True)
              toc5 = time.time()
              time5 += toc5-tic5
              #print("      TERM 5 took: %f"%(toc-tic))
              F_AI   = F_ai
              ktoc = time.time()
              timek += ktoc-ktic
              #print("      KERNEL grid evaluation took: %f"%(ktoc-ktic))
            else: 
              tic = time.time() 
              rho_val_a = dft.numint.eval_rho(self.ints_factory, ao_val,self.wfn.D[0], xctype=self.options.xctype)
              rho_val_b = dft.numint.eval_rho(self.ints_factory, ao_val,self.wfn.D[1], xctype=self.options.xctype)
              rho_val = [rho_val_a, rho_val_b]  
              exc_derivs = dft.libxc.eval_xc(self.options.xc, rho_val,deriv=2,spin=1)[:3]
              v2rho2_uu   = exc_derivs[2][0][:,0]*weights
              v2rho2_ud   = exc_derivs[2][0][:,1]*weights
              v2rho2_dd   = exc_derivs[2][0][:,2]*weights
    
              phi  = ao_val
              rho_u  = rho_val[0][0]
              rho_d  = rho_val[1][0]
              toc = time.time() 
              #print("Assigning pointers",toc-tic,flush=True)   
 
              tic = time.time() 
              phi_o  = np.einsum("rm,mi->ri",phi,Co[0],optimize=True)
              phi_v  = np.einsum("rm,ma->ra",phi,Cv[0],optimize=True)
              rho_vo = np.einsum("ra,ri->rai",phi_v,phi_o,optimize=True)
    
              phi_O  = np.einsum("rm,mi->ri",phi,Co[1],optimize=True)
              phi_V  = np.einsum("rm,ma->ra",phi,Cv[1],optimize=True)
              rho_VO = np.einsum("ra,ri->rai",phi_V,phi_O,optimize=True)
              toc = time.time() 
              #print("Computing orbitals and density properties",toc-tic,flush=True)   
    
              #alpha-alpha terms
    
              #A term
              tic = time.time()
              tmp = np.einsum("r,rls->rls",v2rho2_uu,rho_vo,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[0],optimize=True)
              F_ai += np.einsum("rmn,rN->mnN",rho_vo,Xr,optimize=True)

              tmp = np.einsum("r,rls->rls",v2rho2_ud,rho_VO,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[1],optimize=True)
              F_ai += np.einsum("rmn,rN->mnN",rho_vo,Xr,optimize=True)
              toc = time.time() 
              #print(" A term",toc-tic,flush=True)   

              #beta-beta terms
    
              #A term
              tic = time.time()
              tmp = np.einsum("r,rls->rls",v2rho2_dd,rho_VO,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[1],optimize=True)
              F_AI += np.einsum("rmn,rN->mnN",rho_VO,Xr,optimize=True)

              tmp  = np.einsum("r,rls->rls",v2rho2_ud,rho_vo,optimize=True)
              Xr   = np.einsum("rai,aiN->rN",tmp,X[0],optimize=True)
              F_AI += np.einsum("rmn,rN->mnN",rho_VO,Xr,optimize=True)
              toc = time.time() 
              #print(" A term",toc-tic,flush=True)   

    
          elif (self.options.xctype=="GGA"):
            try:
              ao_file = open(str(self.pid)+".ao_gga_grid"+str(batch),"rb")
              ao_val = io_utils.read_ao_grid(ao_file,"GGA")
            except:
              ao_val = dft.numint.eval_ao(self.ints_factory, points, deriv=1)
              io_utils.write_ao_grid(ao_val,str(self.pid)+".ao_gga_grid"+str(batch),"GGA")
            if spin == 0: 
              rho_val = dft.numint.eval_rho(self.ints_factory, ao_val,2.*self.wfn.D[0], xctype=self.options.xctype)
              exc_derivs = dft.libxc.eval_xc(self.options.xc, rho_val,deriv=2,spin=0)[:3]
              exc    = exc_derivs[0]*weights
              vrho   = exc_derivs[1][0]*weights
              vgamma = exc_derivs[1][1]*weights  
    
              v2rho2      = exc_derivs[2][0]*weights
              v2rho_gamma = exc_derivs[2][1]*weights
              v2gamma2    = exc_derivs[2][2]*weights
    
              phi  = ao_val[0]
              dphi = ao_val[1:4]
    
              rho  = rho_val[0]
              drho = rho_val[1:4]
    
              phi_o  = np.einsum("rm,mi->ri",phi,Co,optimize=True)
              phi_v  = np.einsum("rm,ma->ra",phi,Cv,optimize=True)
              rho_vo = np.einsum("ra,ri->rai",phi_v,phi_o,optimize=True)
    
              dphi_o   = np.einsum("xrm,mi->xri",dphi,Co,optimize=True)
              dphi_v   = np.einsum("xrm,ma->xra",dphi,Cv,optimize=True)
              drho_vo  = np.einsum("xra,ri->xrai",dphi_v,phi_o,optimize=True)
              drho_vo += np.einsum("ra,xri->xrai",phi_v,dphi_o,optimize=True)
    
              #A term
              Xr    = np.einsum("rai,aiN->rN",rho_vo,X,optimize=True)
              F_ai += np.einsum("rai,r,rN->aiN",rho_vo,v2rho2,Xr,optimize=True)
    
              #B term
              tmp   = np.einsum("r,xr,rls->xrls",v2rho_gamma,drho,rho_vo,optimize=True)
              Xr    = np.einsum("xrai,aiN->xrN",tmp,X,optimize=True)
              F_ai += 2.*np.einsum("xrmn,xrN->mnN",drho_vo,Xr,optimize=True)
              
              tmp = np.einsum("r,xr,xrls->rls",v2rho_gamma,drho,drho_vo,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X,optimize=True)
              F_ai += 2.*np.einsum("rmn,rN->mnN",rho_vo,Xr,optimize=True)
              #GGA contribution C
              
              sigma = np.einsum("xr,xrmn->rmn",drho,drho_vo,optimize=True)
              Xr    = np.einsum("rai,aiN->rN",sigma,X,optimize=True)
              F_ai += 4.*np.einsum("rmn,r,rN->mnN",sigma,v2gamma2,Xr,optimize=True)
              
              #GGA contribution D
              Xr    = np.einsum("xrai,aiN->xrN",drho_vo,X,optimize=True)
              F_ai += 2.*np.einsum("xrmn,r,xrN->mnN",drho_vo,vgamma,Xr,optimize=True)
              F_AI = F_ai
    
            elif spin == 1: 
              tic = time.time() 
              rho_val_a = dft.numint.eval_rho(self.ints_factory, ao_val,self.wfn.D[0], xctype=self.options.xctype)
              rho_val_b = dft.numint.eval_rho(self.ints_factory, ao_val,self.wfn.D[1], xctype=self.options.xctype)
              rho_val = [rho_val_a, rho_val_b]  
              exc_derivs = dft.libxc.eval_xc(self.options.xc, rho_val,deriv=2,spin=1)[:3]
              vrho_u   = exc_derivs[1][0][:,0]*weights
              vrho_d   = exc_derivs[1][0][:,1]*weights
    
              vgamma_uu = exc_derivs[1][1][:,0]*weights  
              vgamma_ud = exc_derivs[1][1][:,1]*weights  
              vgamma_dd = exc_derivs[1][1][:,2]*weights  
    
              v2rho2_uu = exc_derivs[2][0][:,0]*weights
              v2rho2_ud = exc_derivs[2][0][:,1]*weights
              v2rho2_dd = exc_derivs[2][0][:,2]*weights
    
              v2rho_gamma_u_uu = exc_derivs[2][1][:,0]*weights
              v2rho_gamma_u_ud = exc_derivs[2][1][:,1]*weights
              v2rho_gamma_u_dd = exc_derivs[2][1][:,2]*weights
              v2rho_gamma_d_uu = exc_derivs[2][1][:,3]*weights
              v2rho_gamma_d_ud = exc_derivs[2][1][:,4]*weights
              v2rho_gamma_d_dd = exc_derivs[2][1][:,5]*weights
    
              v2gamma2_uu_uu = exc_derivs[2][2][:,0]*weights
              v2gamma2_uu_ud = exc_derivs[2][2][:,1]*weights
              v2gamma2_uu_dd = exc_derivs[2][2][:,2]*weights
              v2gamma2_ud_ud = exc_derivs[2][2][:,3]*weights
              v2gamma2_ud_dd = exc_derivs[2][2][:,4]*weights
              v2gamma2_dd_dd = exc_derivs[2][2][:,5]*weights
    
              phi  = ao_val[0]
              dphi = ao_val[1:4]
    
              rho_u  = rho_val[0][0]
              drho_u = rho_val[0][1:4]
              rho_d  = rho_val[1][0]
              drho_d = rho_val[1][1:4]
              toc = time.time() 
              #print("Assigning pointers",toc-tic,flush=True)   
 
              tic = time.time() 
              phi_o  = np.einsum("rm,mi->ri",phi,Co[0],optimize=True)
              phi_v  = np.einsum("rm,ma->ra",phi,Cv[0],optimize=True)
              rho_vo = np.einsum("ra,ri->rai",phi_v,phi_o,optimize=True)
    
              dphi_o   = np.einsum("xrm,mi->xri",dphi,Co[0],optimize=True)
              dphi_v   = np.einsum("xrm,ma->xra",dphi,Cv[0],optimize=True)
              drho_vo  = np.einsum("xra,ri->xrai",dphi_v,phi_o,optimize=True)
              drho_vo += np.einsum("ra,xri->xrai",phi_v,dphi_o,optimize=True)
    
              phi_O  = np.einsum("rm,mi->ri",phi,Co[1],optimize=True)
              phi_V  = np.einsum("rm,ma->ra",phi,Cv[1],optimize=True)
              rho_VO = np.einsum("ra,ri->rai",phi_V,phi_O,optimize=True)
    
              dphi_O   = np.einsum("xrm,mi->xri",dphi,Co[1],optimize=True)
              dphi_V   = np.einsum("xrm,ma->xra",dphi,Cv[1],optimize=True)
              drho_VO  = np.einsum("xra,ri->xrai",dphi_V,phi_O,optimize=True)
              drho_VO += np.einsum("ra,xri->xrai",phi_V,dphi_O,optimize=True)
              toc = time.time() 
              #print("Computing orbitals and density properties",toc-tic,flush=True)   
    
              #alpha-alpha terms
    
              #A term
              tic = time.time()
              tmp = np.einsum("r,rls->rls",v2rho2_uu,rho_vo,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[0],optimize=True)
              F_ai += np.einsum("rmn,rN->mnN",rho_vo,Xr,optimize=True)

              tmp = np.einsum("r,rls->rls",v2rho2_ud,rho_VO,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[1],optimize=True)
              F_ai += np.einsum("rmn,rN->mnN",rho_vo,Xr,optimize=True)
              toc = time.time() 
              #print(" A term",toc-tic,flush=True)   


              tic = time.time()
              sigma_uu = np.einsum("xrmn,xr->rmn",drho_vo,drho_u,optimize=True) 
              sigma_ud = np.einsum("xrmn,xr->rmn",drho_vo,drho_d,optimize=True) 
              sigma_du = np.einsum("xrmn,xr->rmn",drho_VO,drho_u,optimize=True) 
              sigma_dd = np.einsum("xrmn,xr->rmn",drho_VO,drho_d,optimize=True) 
              toc = time.time() 
              #print(" Sigma term",toc-tic,flush=True)   
    
              #B1 term
              tic = time.time()
              tmp = np.einsum("r,rls->rls",v2rho_gamma_u_uu,rho_vo,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[0],optimize=True)
              F_ai += 2.*np.einsum("rmn,rN->mnN",sigma_uu,Xr,optimize=True)
              
              tmp = np.einsum("r,rls->rls",v2rho_gamma_u_ud,rho_vo,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[0],optimize=True)
              F_ai += 1.*np.einsum("rmn,rN->mnN",sigma_ud,Xr,optimize=True)

              tmp = np.einsum("r,rls->rls",v2rho_gamma_d_ud,rho_VO,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[1],optimize=True)
              F_ai += 1.*np.einsum("rmn,rN->mnN",sigma_ud,Xr,optimize=True)
              
              tmp = np.einsum("r,rls->rls",v2rho_gamma_d_uu,rho_VO,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[1],optimize=True)
              F_ai += 2.*np.einsum("rmn,rN->mnN",sigma_uu,Xr,optimize=True)


              toc = time.time() 
              #print(" B1 term",toc-tic,flush=True)   


              #B2 term
              tic = time.time()
              tmp = np.einsum("r,rls->rls",v2rho_gamma_u_uu,sigma_uu,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[0],optimize=True)
              F_ai += 2.*np.einsum("rmn,rN->mnN",rho_vo,Xr,optimize=True)
              
              tmp = np.einsum("r,rls->rls",v2rho_gamma_u_ud,sigma_ud,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[0],optimize=True)
              F_ai += 1.*np.einsum("rmn,rN->mnN",rho_vo,Xr,optimize=True)

              tmp = np.einsum("r,rls->rls",v2rho_gamma_u_ud,sigma_du,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[1],optimize=True)
              F_ai += 1.*np.einsum("rmn,rN->mnN",rho_vo,Xr,optimize=True)
              
              tmp = np.einsum("r,rls->rls",v2rho_gamma_u_dd,sigma_dd,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[1],optimize=True)
              F_ai += 2.*np.einsum("rmn,rN->mnN",rho_vo,Xr,optimize=True)
              toc = time.time() 
              #print(" B2 term",toc-tic,flush=True)   
    
              #C term
              tic = time.time()
              X_uu_u  = np.einsum("rai,aiN->rN",sigma_uu,X[0],optimize=True)
              X_ud_u  = np.einsum("rai,aiN->rN",sigma_ud,X[0],optimize=True)
              F_ai += 4.*np.einsum("rmn,r,rN->mnN",sigma_uu,v2gamma2_uu_uu,X_uu_u,optimize=True)
              F_ai += 2.*np.einsum("rmn,r,rN->mnN",sigma_uu,v2gamma2_uu_ud,X_ud_u,optimize=True)
              F_ai += 2.*np.einsum("rmn,r,rN->mnN",sigma_ud,v2gamma2_uu_ud,X_uu_u,optimize=True)
              F_ai += 1.*np.einsum("rmn,r,rN->mnN",sigma_ud,v2gamma2_ud_ud,X_ud_u,optimize=True)

              X_dd_d = np.einsum("rai,aiN->rN",sigma_dd,X[1],optimize=True)
              X_du_d = np.einsum("rai,aiN->rN",sigma_du,X[1],optimize=True)
              F_ai += 4.*np.einsum("rmn,r,rN->mnN",sigma_uu,v2gamma2_uu_dd,X_dd_d,optimize=True)
              F_ai += 2.*np.einsum("rmn,r,rN->mnN",sigma_ud,v2gamma2_ud_dd,X_dd_d,optimize=True)
              F_ai += 2.*np.einsum("rmn,r,rN->mnN",sigma_uu,v2gamma2_uu_ud,X_du_d,optimize=True)
              F_ai += 1.*np.einsum("rmn,r,rN->mnN",sigma_ud,v2gamma2_ud_ud,X_du_d,optimize=True)
              toc = time.time() 
              #print(" C term",toc-tic,flush=True)   
    
    
              #D term
              tic = time.time()
              Xr  = np.einsum("xrai,aiN->xrN",drho_vo,X[0],optimize=True)
              F_ai += 2.*np.einsum("xrmn,r,xrN->mnN",drho_vo,vgamma_uu,Xr,optimize=True)

              Xr      = np.einsum("xrai,aiN->xrN",drho_VO,X[1],optimize=True)
              F_ai += 1.*np.einsum("xrmn,r,xrN->mnN",drho_vo,vgamma_ud,Xr,optimize=True)
              toc = time.time() 
              #print(" D term",toc-tic,flush=True)   
    
    
              #beta-beta terms
    
              #A term
              tic = time.time()
              tmp = np.einsum("r,rls->rls",v2rho2_dd,rho_VO,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[1],optimize=True)
              F_AI += np.einsum("rmn,rN->mnN",rho_VO,Xr,optimize=True)

              tmp  = np.einsum("r,rls->rls",v2rho2_ud,rho_vo,optimize=True)
              Xr   = np.einsum("rai,aiN->rN",tmp,X[0],optimize=True)
              F_AI += np.einsum("rmn,rN->mnN",rho_VO,Xr,optimize=True)
              toc = time.time() 
              #print(" A term",toc-tic,flush=True)   

              #B1 term
              tic = time.time()
              tmp = np.einsum("r,rls->rls",v2rho_gamma_d_dd,rho_VO,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[1],optimize=True)
              F_AI += 2.*np.einsum("rmn,rN->mnN",sigma_dd,Xr,optimize=True)
              
              tmp = np.einsum("r,rls->rls",v2rho_gamma_d_ud,rho_VO,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[1],optimize=True)
              F_AI += 1.*np.einsum("rmn,rN->mnN",sigma_du,Xr,optimize=True)

              tmp = np.einsum("r,rls->rls",v2rho_gamma_u_dd,rho_vo,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[0],optimize=True)
              F_AI += 2.*np.einsum("rmn,rN->mnN",sigma_dd,Xr,optimize=True)
              
              tmp = np.einsum("r,rls->rls",v2rho_gamma_u_ud,rho_vo,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[0],optimize=True)
              F_AI += 1.*np.einsum("rmn,rN->mnN",sigma_du,Xr,optimize=True)
              toc = time.time() 
              #print(" B1 term",toc-tic,flush=True)   

              
              #B2 term
              tic = time.time()
              tmp = np.einsum("r,rls->rls",v2rho_gamma_d_dd,sigma_dd,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[1],optimize=True)
              F_AI += 2.*np.einsum("rmn,rN->mnN",rho_VO,Xr,optimize=True)
              
              tmp = np.einsum("r,rls->rls",v2rho_gamma_d_ud,sigma_du,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[1],optimize=True)
              F_AI += 1.*np.einsum("rmn,rN->mnN",rho_VO,Xr,optimize=True)

              tmp = np.einsum("r,rls->rls",v2rho_gamma_d_uu,sigma_uu,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[0],optimize=True)
              F_AI += 2.*np.einsum("rmn,rN->mnN",rho_VO,Xr,optimize=True)
              
              tmp = np.einsum("r,rls->rls",v2rho_gamma_d_ud,sigma_ud,optimize=True)
              Xr  = np.einsum("rai,aiN->rN",tmp,X[0],optimize=True)
              F_AI += 1.*np.einsum("rmn,rN->mnN",rho_VO,Xr,optimize=True)
              toc = time.time() 
              #print(" B2 term",toc-tic,flush=True)   
    
              #C term
              tic = time.time()
              Xu      = np.einsum("rai,aiN->rN",sigma_du,X[1],optimize=True)
              Xd      = np.einsum("rai,aiN->rN",sigma_dd,X[1],optimize=True)
              F_AI += 4.*np.einsum("rmn,r,rN->mnN",sigma_dd,v2gamma2_dd_dd,Xd,optimize=True)
              F_AI += 2.*np.einsum("rmn,r,rN->mnN",sigma_dd,v2gamma2_ud_dd,Xu,optimize=True)
              F_AI += 2.*np.einsum("rmn,r,rN->mnN",sigma_du,v2gamma2_ud_dd,Xd,optimize=True)
              F_AI += 1.*np.einsum("rmn,r,rN->mnN",sigma_du,v2gamma2_ud_ud,Xu,optimize=True)

              Xr      = np.einsum("rai,aiN->rN",sigma_uu,X[0],optimize=True)
              F_AI += 4.*np.einsum("rmn,r,rN->mnN",sigma_dd,v2gamma2_uu_dd,Xr,optimize=True)
              Xr      = np.einsum("rai,aiN->rN",sigma_ud,X[0],optimize=True)
              F_AI += 2.*np.einsum("rmn,r,rN->mnN",sigma_dd,v2gamma2_ud_dd,Xr,optimize=True)
              Xr      = np.einsum("rai,aiN->rN",sigma_uu,X[0],optimize=True)
              F_AI += 2.*np.einsum("rmn,r,rN->mnN",sigma_du,v2gamma2_uu_ud,Xr,optimize=True)
              Xr      = np.einsum("rai,aiN->rN",sigma_uu,X[0],optimize=True)
              F_AI += 1.*np.einsum("rmn,r,rN->mnN",sigma_dd,v2gamma2_ud_ud,Xr,optimize=True)
              toc = time.time() 
              #print(" C term",toc-tic,flush=True)   
    
              #D term
              tic = time.time()
              Xr  = np.einsum("xrai,aiN->xrN",drho_VO,X[1],optimize=True)
              F_AI += 2.*np.einsum("xrmn,r,xrN->mnN",drho_VO,vgamma_dd,Xr,optimize=True)

              Xr    = np.einsum("xrai,aiN->xrN",drho_vo,X[0],optimize=True)
              F_AI += 1.*np.einsum("xrmn,r,xrN->mnN",drho_VO,vgamma_ud,Xr,optimize=True)
              toc = time.time() 
              #print(" D term",toc-tic,flush=True)   

#        print(" TAO: %f"%time_ao,flush=True)
#        print(" T1: %f"%time1,flush=True)
#        print(" T2: %f"%time2,flush=True)
#        print(" T3: %f"%time3,flush=True)
#        print(" T4: %f"%time4,flush=True)
#        print(" T5: %f"%time5,flush=True)
#        print(" TK: %f"%timek,flush=True)
#        #exit(0)
        return F_ai, F_AI
