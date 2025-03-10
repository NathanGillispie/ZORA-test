import numpy as np
from pyscf import ao2mo

def uccd_energy(wfn):

  Ca = wfn.C[0]
  Cb = wfn.C[1]  

  
  nbf  = wfn.nbf
  print(nbf)
  no_a = wfn.nel[0]
  no_b = wfn.nel[1]
  nv_a = nbf - no_a # 3 #
  nv_b = nbf - no_b # 4 #
  print(no_a, no_b, nv_a, nv_b) 

  Co_a = Ca[:,:no_a]
  Co_b = Cb[:,:no_b]
  Cv_a = Ca[:,no_a:no_a+nv_a]
  Cv_b = Cb[:,no_b:no_b+nv_b]
  print(Cv_a.shape) 
  print(Cv_b.shape) 
#  print(Cv_a.shape) 
  eps_a = wfn.eps[0][:no_a+nv_a]
  eps_b = wfn.eps[1][:no_b+nv_b]

  maxiter = 500

#  #build antisymmetrized two-electron integrals
#  I_aa  = ao2mo.general(wfn.ints_factory,[Ca,Ca,Ca,Ca],compact=False)
#  I_bb  = ao2mo.general(wfn.ints_factory,[Cb,Cb,Cb,Cb],compact=False)
#  I_ab  = ao2mo.general(wfn.ints_factory,[Ca,Ca,Cb,Cb],compact=False)
#  I_ba  = ao2mo.general(wfn.ints_factory,[Cb,Cb,Ca,Ca],compact=False)
#
#  I_aa = I_aa.reshape(nbf,nbf,nbf,nbf)
#  I_bb = I_bb.reshape(nbf,nbf,nbf,nbf)
#  I_ab = I_ab.reshape(nbf,nbf,nbf,nbf)
#  I_ba = I_ba.reshape(nbf,nbf,nbf,nbf)
  
  t2_aa = np.zeros((nv_a,nv_a,no_a,no_a))
  t2_bb = np.zeros((nv_b,nv_b,no_b,no_b))
  t2_ab = np.zeros((nv_a,nv_b,no_a,no_b))
  
  #I_vvoo = I_aa[no_a:,:no_a,no_a:,:no_a].swapaxes(1,2)
  #I_VVOO = I_bb[no_b:,:no_b,no_b:,:no_b].swapaxes(1,2)
  #I_vVoO = I_ab[no_a:,:no_a,no_b:,:no_b].swapaxes(1,2)
  #I_VvOo = I_ba[no_b:,:no_b,no_a:,:no_a].swapaxes(1,2)
  I_vvoo = ao2mo.general(wfn.ints_factory,[Cv_a,Co_a,Cv_a,Co_a],compact=False).reshape(nv_a,no_a,nv_a,no_a).swapaxes(1,2)
  I_VVOO = ao2mo.general(wfn.ints_factory,[Cv_b,Co_b,Cv_b,Co_b],compact=False).reshape(nv_b,no_b,nv_b,no_b).swapaxes(1,2)
  I_vVoO = ao2mo.general(wfn.ints_factory,[Cv_a,Co_a,Cv_b,Co_b],compact=False).reshape(nv_a,no_a,nv_b,no_b).swapaxes(1,2)
  I_VvOo = I_vVoO.swapaxes(0,1).swapaxes(2,3) #ao2mo.general(wfn.ints_factory,[Cv_b,Co_b,Cv_a,Co_a],compact=True).reshape(nv_b,no_b,nv_a,no_a).swapaxes(1,2)

  I_vvoo -= I_vvoo.swapaxes(2,3)
  I_VVOO -= I_VVOO.swapaxes(2,3)

  def d_aa():
    d_aa  = np.einsum("i,abij->abij",eps_a[:no_a],np.ones((nv_a,nv_a,no_a,no_a)),optimize=True)
    d_aa += np.einsum("j,abij->abij",eps_a[:no_a],np.ones((nv_a,nv_a,no_a,no_a)),optimize=True)
    d_aa -= np.einsum("a,abij->abij",eps_a[no_a:],np.ones((nv_a,nv_a,no_a,no_a)),optimize=True)
    d_aa -= np.einsum("b,abij->abij",eps_a[no_a:],np.ones((nv_a,nv_a,no_a,no_a)),optimize=True)
    return d_aa

  def d_ab():
    d_ab  = np.einsum("i,abij->abij",eps_a[:no_a],np.ones((nv_a,nv_b,no_a,no_b)),optimize=True)
    d_ab += np.einsum("j,abij->abij",eps_b[:no_b],np.ones((nv_a,nv_b,no_a,no_b)),optimize=True)
    d_ab -= np.einsum("a,abij->abij",eps_a[no_a:],np.ones((nv_a,nv_b,no_a,no_b)),optimize=True)
    d_ab -= np.einsum("b,abij->abij",eps_b[no_b:],np.ones((nv_a,nv_b,no_a,no_b)),optimize=True)
    return d_ab

  def d_bb():
    d_bb  = np.einsum("i,abij->abij",eps_b[:no_b],np.ones((nv_b,nv_b,no_b,no_b)),optimize=True)
    d_bb += np.einsum("j,abij->abij",eps_b[:no_b],np.ones((nv_b,nv_b,no_b,no_b)),optimize=True)
    d_bb -= np.einsum("a,abij->abij",eps_b[no_b:],np.ones((nv_b,nv_b,no_b,no_b)),optimize=True)
    d_bb -= np.einsum("b,abij->abij",eps_b[no_b:],np.ones((nv_b,nv_b,no_b,no_b)),optimize=True)
    return d_bb


  D_aa = d_aa()
  D_ab = d_ab()
  D_bb = d_bb()

  t2_aa = I_vvoo/D_aa
  t2_ab = I_vVoO/D_ab
  t2_bb = I_VVOO/D_bb
  
  Emp2_aa  = 0.25 * np.einsum("abij,abij->",t2_aa,I_vvoo,optimize=True)
  Emp2_bb  = 0.25 * np.einsum("abij,abij->",t2_bb,I_VVOO,optimize=True)
  Emp2_os  = 1.00 * np.einsum("abij,abij->",t2_ab,I_vVoO,optimize=True)
  
  print(Emp2_aa)
  print(Emp2_os)
  print(Emp2_bb)
  print((Emp2_aa+Emp2_bb+ Emp2_os+wfn.scf_energy))
#  return None

  #coupled-cluster code
  
  # voov integrals
#  I_voov = I_aa[no_a:,:no_a,:no_a,no_a:].swapaxes(1,2)
#  I_VOOV = I_bb[no_b:,:no_b,:no_b,no_b:].swapaxes(1,2)
#  I_vOoV = I_ab[no_a:,:no_a,:no_b,no_b:].swapaxes(1,2)
#  I_VoOv = I_ba[no_b:,:no_b,:no_a,no_a:].swapaxes(1,2)

  I_voov = ao2mo.general(wfn.ints_factory,[Cv_a,Co_a,Co_a,Cv_a],compact=False).reshape(nv_a,no_a,no_a,nv_a).swapaxes(1,2)
  I_VOOV = ao2mo.general(wfn.ints_factory,[Cv_b,Co_b,Co_b,Cv_b],compact=False).reshape(nv_b,no_b,no_b,nv_b).swapaxes(1,2)
  I_vOoV = ao2mo.general(wfn.ints_factory,[Cv_a,Co_a,Co_b,Cv_b],compact=False).reshape(nv_a,no_a,no_b,nv_b).swapaxes(1,2)
  I_VoOv = I_vOoV.swapaxes(0,3).swapaxes(1,2) #ao2mo.general(wfn.ints_factory,[Cv_b,Co_b,Co_a,Cv_a],compact=False).reshape(nv_b,no_b,no_a,nv_a).swapaxes(1,2)
  
#  I_vovo = I_aa[no_a:,no_a:,:no_a,:no_a].swapaxes(1,2)
#  I_VOVO = I_bb[no_b:,no_b:,:no_b,:no_b].swapaxes(1,2)
#  I_vOvO = I_ab[no_a:,no_a:,:no_b,:no_b].swapaxes(1,2)
#  I_VoVo = I_ba[no_b:,no_b:,:no_a,:no_a].swapaxes(1,2)
  I_vovo = ao2mo.general(wfn.ints_factory,[Cv_a,Cv_a,Co_a,Co_a],compact=False).reshape(nv_a,nv_a,no_a,no_a).swapaxes(1,2)
  I_VOVO = ao2mo.general(wfn.ints_factory,[Cv_b,Cv_b,Co_b,Co_b],compact=False).reshape(nv_b,nv_b,no_b,no_b).swapaxes(1,2)
  I_vOvO = ao2mo.general(wfn.ints_factory,[Cv_a,Cv_a,Co_b,Co_b],compact=False).reshape(nv_a,nv_a,no_b,no_b).swapaxes(1,2)
  I_VoVo = ao2mo.general(wfn.ints_factory,[Cv_b,Cv_b,Co_a,Co_a],compact=False).reshape(nv_b,nv_b,no_a,no_a).swapaxes(1,2)
  
  I_voov -= I_vovo.swapaxes(2,3)
  I_VOOV -= I_VOVO.swapaxes(2,3)
  I_vOOv  = -I_vOvO.swapaxes(2,3)
  I_VooV  = -I_VoVo.swapaxes(2,3)
  
  # vvvv integrals
#  I_vvvv = I_aa[no_a:,no_a:,no_a:,no_a:].swapaxes(1,2)
#  I_VVVV = I_bb[no_b:,no_b:,no_b:,no_b:].swapaxes(1,2)
#  I_vVvV = I_ab[no_a:,no_a:,no_b:,no_b:].swapaxes(1,2)
  I_vvvv = ao2mo.general(wfn.ints_factory,[Cv_a,Cv_a,Cv_a,Cv_a],compact=False).reshape(nv_a,nv_a,nv_a,nv_a).swapaxes(1,2)
  I_VVVV = ao2mo.general(wfn.ints_factory,[Cv_b,Cv_b,Cv_b,Cv_b],compact=False).reshape(nv_b,nv_b,nv_b,nv_b).swapaxes(1,2)
  I_vVvV = ao2mo.general(wfn.ints_factory,[Cv_a,Cv_a,Cv_b,Cv_b],compact=False).reshape(nv_a,nv_a,nv_b,nv_b).swapaxes(1,2)
  
  I_vvvv -= I_vvvv.swapaxes(2,3)
  I_VVVV -= I_VVVV.swapaxes(2,3)
  
  
  # oooo integrals
#  I_oooo = I_aa[:no_a,:no_a,:no_a,:no_a].swapaxes(1,2)
#  I_OOOO = I_bb[:no_b,:no_b,:no_b,:no_b].swapaxes(1,2)
#  I_oOoO = I_ab[:no_a,:no_a,:no_b,:no_b].swapaxes(1,2)
  I_oooo = ao2mo.general(wfn.ints_factory,[Co_a,Co_a,Co_a,Co_a],compact=False).reshape(no_a,no_a,no_a,no_a).swapaxes(1,2)
  I_OOOO = ao2mo.general(wfn.ints_factory,[Co_b,Co_b,Co_b,Co_b],compact=False).reshape(no_b,no_b,no_b,no_b).swapaxes(1,2)
  I_oOoO = ao2mo.general(wfn.ints_factory,[Co_a,Co_a,Co_b,Co_b],compact=False).reshape(no_a,no_a,no_b,no_b).swapaxes(1,2)
  
  I_oooo -= I_oooo.swapaxes(2,3)
  I_OOOO -= I_OOOO.swapaxes(2,3)

  def Fac():
      F  = -0.5 * np.einsum("dakl,dckl->ac",t2_aa,I_vvoo,optimize=True)
      F -=  1.0 * np.einsum("adlk,cdlk->ac",t2_ab,I_vVoO,optimize=True)
      return F
  
  def FAC():
      F  = -0.5 * np.einsum("dakl,dckl->ac",t2_bb,I_VVOO,optimize=True)
      F -=  1.0 * np.einsum("dakl,dckl->ac",t2_ab,I_vVoO,optimize=True)
      return F
  
  def Fki():
      F  = 0.5 *np.einsum("cdli,cdlk->ki",t2_aa,I_vvoo,optimize=True)
      F += 1.0 *np.einsum("dcil,dckl->ki",t2_ab,I_vVoO,optimize=True)
      return F
  
  def FKI():
      F  = 0.5 *np.einsum("cdli,cdlk->ki",t2_bb,I_VVOO,optimize=True)
      F += 1.0 *np.einsum("cdli,cdlk->ki",t2_ab,I_vVoO,optimize=True)
      return F

  # Wvoov intermediates
  
  # Wvoov
  def Wakic():
      W  = I_voov.copy()
      W += 0.5 * np.einsum("adil,dclk->akic",t2_aa,I_vvoo,optimize=True)
      W += 0.5 * np.einsum("adil,dclk->akic",t2_ab,I_VvOo,optimize=True)
      return W
  
  # WVOOV
  def WAKIC():
      W  = I_VOOV.copy()
      W += 0.5 * np.einsum("adil,dclk->akic",t2_bb,I_VVOO,optimize=True)
      W += 0.5 * np.einsum("dali,dclk->akic",t2_ab,I_vVoO,optimize=True)
      return W
  
  # WvOoV
  def WaKiC():
      W  = I_vOoV.copy()
      W += 0.5 * np.einsum("adil,dclk->akic",t2_aa,I_vVoO,optimize=True)
      W += 0.5 * np.einsum("adil,dclk->akic",t2_ab,I_VVOO,optimize=True)
      return W
  
  def Wakic_aa():
      W  = I_voov.copy()
      W += 0.5 * np.einsum("adil,dclk->akic",t2_aa,I_vvoo,optimize=True)
      return W
  
  def WAKIC_bb():
      W  = I_VOOV.copy()
      W += 0.5 * np.einsum("adil,dclk->akic",t2_bb,I_VVOO,optimize=True)
      return W
  
  def WaKiC_aa():
      W  = I_vOoV.copy()
      W += 1.0 * np.einsum("adil,dclk->akic",t2_aa,I_vVoO,optimize=True)
      W += 0.5 * np.einsum("adil,dclk->akic",t2_ab,I_VVOO,optimize=True)
      return W
  
  def WAkIc_bb():
      W  = I_VoOv.copy()
      W += 0.5 * np.einsum("dali,dclk->akic",t2_ab,I_vvoo,optimize=True)
      W += 1.0 * np.einsum("adil,dclk->akic",t2_bb,I_VvOo,optimize=True)
      return W

  ## WVoOv
  def WAkIc():
      W  = I_VoOv.copy()
      W += 0.5 * np.einsum("dali,dclk->akic",t2_ab,I_vvoo,optimize=True)
      W += 0.5 * np.einsum("adil,dclk->akic",t2_bb,I_VvOo,optimize=True)
      return W
  
  # WVooV
  def WAkiC():
      W  = I_VooV.copy()
      W += 0.5 * np.einsum("dail,cdlk->akic",t2_ab,I_VvOo,optimize=True)
      return W
  
  # WvOOv
  def WaKIc():
      W = I_vOOv.copy()
      W += 0.5 * np.einsum("adli,dckl->akic",t2_ab,I_VvOo,optimize=True)
      return W
  
  def Wabcd():
      W  = I_vvvv.copy()
      W += 0.25 * np.einsum("abkl,cdkl->abcd",t2_aa,I_vvoo,optimize=True)
      return W
  
  def WaBcD():
      W  = I_vVvV.copy()
      W += 0.5 * np.einsum("abkl,cdkl->abcd",t2_ab,I_vVoO,optimize=True)
      return W
  
  def WABCD():
      W  = I_VVVV.copy()
      W += 0.25 * np.einsum("abkl,cdkl->abcd",t2_bb,I_VVOO,optimize=True)
      return W
  
  def Wklij():
      W  = I_oooo.copy()
      W += 0.25 * np.einsum("cdij,cdkl->klij",t2_aa,I_vvoo,optimize=True)
      return W
  
  def WkLiJ():
      W  = I_oOoO.copy()
      W += 0.5 * np.einsum("cdij,cdkl->klij",t2_ab,I_vVoO,optimize=True)
      return W

  def WKLIJ():
      W  = I_OOOO.copy()
      W += 0.25 * np.einsum("cdij,cdkl->klij",t2_bb,I_VVOO,optimize=True)
      return W

#  D_aa = d_aa()
#  D_ab = d_ab()
#  D_bb = d_bb()
  
  e_ccd = Emp2_aa+Emp2_bb+ Emp2_os
  print("Reference Energy: %20.12f"%e_ccd)

  E_conv = 1e-8
  for itter in range(maxiter):
  
      # alpha-beta block
      rhs_T2_ab  = I_vVoO.copy()
  
      rhs_T2_ab += np.einsum("cbij,ac->abij",t2_ab,Fac(),optimize=True)
      rhs_T2_ab += np.einsum("acij,bc->abij",t2_ab,FAC(),optimize=True)
  
      rhs_T2_ab -= np.einsum("abkj,ki->abij",t2_ab,Fki(),optimize=True)
      rhs_T2_ab -= np.einsum("abik,kj->abij",t2_ab,FKI(),optimize=True)
  
      rhs_T2_ab += np.einsum("abkl,klij->abij",t2_ab,WkLiJ(),optimize=True)
      rhs_T2_ab += np.einsum("cdij,abcd->abij",t2_ab,WaBcD(),optimize=True)
  
      rhs_T2_ab += np.einsum("cbkj,akic->abij",t2_ab,Wakic(),optimize=True)
      rhs_T2_ab += np.einsum("ackj,bkic->abij",t2_ab,WAkiC(),optimize=True)
      rhs_T2_ab += np.einsum("cbik,akjc->abij",t2_ab,WaKIc(),optimize=True)
      rhs_T2_ab += np.einsum("acik,bkjc->abij",t2_ab,WAKIC(),optimize=True)
  
      rhs_T2_ab += np.einsum("acik,bkjc->abij",t2_aa,WAkIc(),optimize=True)
      rhs_T2_ab += np.einsum("cbkj,akic->abij",t2_bb,WaKiC(),optimize=True)
  
  
      # alpha-alpha block
      rhs_T2_aa  = I_vvoo.copy()
  
      rhs_T2_aa += np.einsum("cbij,ac->abij",t2_aa,Fac(),optimize=True)
      rhs_T2_aa += np.einsum("acij,bc->abij",t2_aa,Fac(),optimize=True)
  
      rhs_T2_aa -= np.einsum("abkj,ki->abij",t2_aa,Fki(),optimize=True)
      rhs_T2_aa -= np.einsum("abik,kj->abij",t2_aa,Fki(),optimize=True)
  
      rhs_T2_aa += 0.5 * np.einsum("abkl,klij->abij",t2_aa,Wklij(),optimize=True)
      rhs_T2_aa += 0.5 * np.einsum("cdij,abcd->abij",t2_aa,Wabcd(),optimize=True)
  
      Pabij = np.einsum("cbkj,akic->abij",t2_aa,Wakic_aa(),optimize=True)
      rhs_T2_aa += Pabij
      rhs_T2_aa -= Pabij.swapaxes(0,1)
      rhs_T2_aa -= Pabij.swapaxes(2,3)
      rhs_T2_aa += Pabij.swapaxes(2,3).swapaxes(0,1)
  
      Pabij = np.einsum("bcjk,akic->abij",t2_ab,WaKiC_aa(),optimize=True)
      rhs_T2_aa += Pabij
      rhs_T2_aa -= Pabij.swapaxes(0,1)
      rhs_T2_aa -= Pabij.swapaxes(2,3)
      rhs_T2_aa += Pabij.swapaxes(2,3).swapaxes(0,1)


      rhs_T2_bb  = I_VVOO.copy()
  
      rhs_T2_bb += np.einsum("cbij,ac->abij",t2_bb,FAC(),optimize=True)
      rhs_T2_bb += np.einsum("acij,bc->abij",t2_bb,FAC(),optimize=True)
  
      rhs_T2_bb -= np.einsum("abkj,ki->abij",t2_bb,FKI(),optimize=True)
      rhs_T2_bb -= np.einsum("abik,kj->abij",t2_bb,FKI(),optimize=True)
  
      rhs_T2_bb += 0.5 * np.einsum("abkl,klij->abij",t2_bb,WKLIJ(),optimize=True)
      rhs_T2_bb += 0.5 * np.einsum("cdij,abcd->abij",t2_bb,WABCD(),optimize=True)
  
      Pabij = np.einsum("cbkj,akic->abij",t2_bb,WAKIC_bb(),optimize=True)
      rhs_T2_bb += Pabij
      rhs_T2_bb -= Pabij.swapaxes(0,1)
      rhs_T2_bb -= Pabij.swapaxes(2,3)
      rhs_T2_bb += Pabij.swapaxes(2,3).swapaxes(0,1)
  
      Pabij = np.einsum("cbkj,akic->abij",t2_ab,WAkIc_bb(),optimize=True)
      rhs_T2_bb += Pabij
      rhs_T2_bb -= Pabij.swapaxes(0,1)
      rhs_T2_bb -= Pabij.swapaxes(2,3)
      rhs_T2_bb += Pabij.swapaxes(2,3).swapaxes(0,1)
  
  
      t2_ab = rhs_T2_ab/D_ab
      t2_aa = rhs_T2_aa/D_aa
      t2_bb = rhs_T2_bb/D_bb
  
      e_ccd_new_os  = np.einsum("abij,abij->",t2_ab,I_vVoO,optimize=True)
      e_ccd_new_ss  = 0.25 * np.einsum("abij,abij->",t2_aa,I_vvoo,optimize=True)
      e_ccd_new_ss += 0.25 * np.einsum("abij,abij->",t2_bb,I_VVOO,optimize=True)
  
      e_ccd_new = e_ccd_new_os + e_ccd_new_ss
  
      de = np.abs(e_ccd_new-e_ccd)
      print("Iteration %3i %20.12f %20.12f"%(itter+1,e_ccd_new,de))
      if de < E_conv:
         break
  
      e_ccd = e_ccd_new
  print("    CCD Correlation Energy: %20.12f"%(e_ccd))
  print("    CCD Total Energy:       %20.12f"%(e_ccd+wfn.scf_energy))
