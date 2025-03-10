import time
import numpy as np
import scipy as sp
#import grid
#import lsda
import scf
import time
import ghf
#import matplotlib.pyplot as plt

np.set_printoptions(precision=8, linewidth=200, suppress=True)

def print_header():
    print("")
    print("    ==========================================")
    print("    *  Complex Polarization Propagator (CPP) *")
    print("    *                  by                    *")
    print("    *   Sarah Pak and Daniel R. Nascimento   *")
    print("    ==========================================")
    print("")
    return None

def transform_teis(ERI, Cp, Cq, Cr, Cs): 
    #function takes ERI in the AO basis and transforms to the 
    #MO space defined by the p,q,r and s indices
#    tic = time.perf_counter()
    
    tmp = np.einsum('MNSL,Ls->MNSs',ERI,Cs)
    tmp = np.einsum('MNSs,Sr->MNrs',tmp,Cr)
    tmp = np.einsum('MNrs,Nq->Mqrs',tmp,Cq)

#    toc = time.perf_counter()
#    print("AO->MO transformation:",toc-tic)

    return np.einsum('Mqrs,Mp->pqrs',tmp,Cp)

def oscillator_strengths(wfn):

    na_docc = wfn["ndocc"][0]
    na_socc = wfn["nsocc"][0]
    nb_docc = wfn["ndocc"][1]
    nb_socc = wfn["nsocc"][1]
    if na_docc == nb_docc: 
      restricted = True
    else:
      restricted = False

    nmo   = wfn["nbf"]
    na_virt = nmo - na_docc - na_socc
    nb_virt = nmo - nb_docc - nb_socc
    #redefine na_docc as all that matters is whether it's virtual or occupied
    na_docc += na_socc 

    SCF_E = wfn["total_energy"]
 
    mu    = wfn["dipole_integrals"]
    Ca    = wfn["molecular_orbitals"][0]
    Cb    = wfn["molecular_orbitals"][1]
    w_wfn = wfn["excitation_energies"]
    X_wfn = wfn["excited_wfn"]

    #MO dipole integrals of dimension virt x occ
    MUx_a = np.einsum('uj,vi,uv', Ca, Ca, mu[0])[na_docc:,:na_docc]
    MUy_a = np.einsum('uj,vi,uv', Ca, Ca, mu[1])[na_docc:,:na_docc]
    MUz_a = np.einsum('uj,vi,uv', Ca, Ca, mu[2])[na_docc:,:na_docc]

    if restricted is False:
      MUx_b = np.einsum('uj,vi,uv', Cb, Cb, mu[0])[nb_docc:,:nb_docc]
      MUy_b = np.einsum('uj,vi,uv', Cb, Cb, mu[1])[nb_docc:,:nb_docc]
      MUz_b = np.einsum('uj,vi,uv', Cb, Cb, mu[2])[nb_docc:,:nb_docc]
    nroots = na_docc * na_virt 
    offset = 0
    if restricted is False:
      nroots += nb_docc * nb_virt
      offset += na_docc * na_virt    

    tdm = np.zeros((nroots,3))
    os  = np.zeros((nroots))
    spectrum = []
    for root in range(nroots):
      tdm[root][0] = 0.0
      tdm[root][1] = 0.0
      tdm[root][2] = 0.0
      for i in range(na_docc):
        for a in range(na_virt):
          tdm[root][0] += MUx_a[a][i]* X_wfn[i*na_virt+a][root] 
          tdm[root][1] += MUy_a[a][i]* X_wfn[i*na_virt+a][root] 
          tdm[root][2] += MUz_a[a][i]* X_wfn[i*na_virt+a][root] 
      if restricted is False:
        for i in range(nb_docc):
          for a in range(nb_virt):
            tdm[root][0] += MUx_b[a][i]* X_wfn[i*nb_virt+a+offset][root] 
            tdm[root][1] += MUy_b[a][i]* X_wfn[i*nb_virt+a+offset][root] 
            tdm[root][2] += MUz_b[a][i]* X_wfn[i*nb_virt+a+offset][root] 

        os[root] += 2.0/3.0 * w_wfn[root] * tdm[root][0]**2
        os[root] += 2.0/3.0 * w_wfn[root] * tdm[root][1]**2
        os[root] += 2.0/3.0 * w_wfn[root] * tdm[root][2]**2
      else:
        os[root] += 4.0/3.0 * w_wfn[root] * tdm[root][0]**2
        os[root] += 4.0/3.0 * w_wfn[root] * tdm[root][1]**2
        os[root] += 4.0/3.0 * w_wfn[root] * tdm[root][2]**2
 
      spectrum.append([w_wfn[root]*27.21138, tdm[root][0], tdm[root][1], tdm[root][2], os[root]])
    return spectrum

def compute(wfn):

    print_header()
    ## Get wavefunction info
    nocc = wfn.nel
    nmo  = np.asarray([wfn.nbf,wfn.nbf])
    nvir = nmo - nocc
    C = 1.*wfn.C
    nov = int(nocc[0] * nvir[0])
    eps = wfn.eps
    print(eps)
    mu = wfn.mu

    G = wfn.ints_factory.intor('int2e')
    G_aibj = transform_teis(G, C[:,nocc[0]:], C[:,:nocc[0]],C[:,nocc[0]:], C[:,:nocc[0]]) 
    G_abij = transform_teis(G, C[:,nocc[0]:], C[:,nocc[0]:], C[:,:nocc[0]],C[:,:nocc[0]]) 

    A = np.zeros((nov,nov))
    for a in range(nvir[0]):
      for i in range(nocc[0]):
         A[a*nocc[0]+i][a*nocc[0]+i]  = (eps[a+nocc[0]] - eps[i]) 
 
    A += 2*G_aibj.reshape((nov,nov)) - G_abij.swapaxes(1,2).reshape((nov,nov)) 
   
    omega, X = sp.linalg.eigh(A)

    print(np.sum(C[:,0]))

    mu_ai_x = np.einsum("ma,ni,mn->ai",C[:,nocc[0]:],C[:,:nocc[0]],mu[0]).reshape((nov))    
    mu_ai_y = np.einsum("ma,ni,mn->ai",C[:,nocc[0]:],C[:,:nocc[0]],mu[1]).reshape((nov))    
    mu_ai_z = np.einsum("ma,ni,mn->ai",C[:,nocc[0]:],C[:,:nocc[0]],mu[2]).reshape((nov))    
 
    tdmx = np.einsum("nN,n->N",X,mu_ai_x)*np.sqrt(2.)
    tdmy = np.einsum("nN,n->N",X,mu_ai_y)*np.sqrt(2.)
    tdmz = np.einsum("nN,n->N",X,mu_ai_z)*np.sqrt(2.)
    print(omega[0:20])
  
    ds = (tdmx**2 + tdmy**2 + tdmz**2) #* omega
    #multiply by 2w/3 to get oscillator strengths
    print(ds[0:20])
 
    wrange = wfn.frequencies[:3]
    dump   = wfn.frequencies[3]
    w1 , S1 = ghf.lorentzian(wrange,dump,omega,ds)   
    
    O = np.eye(nov,dtype=complex)
    Px = -mu_ai_x
    Py = -mu_ai_y
    Pz = -mu_ai_z

    w2 = np.arange(wrange[0],wrange[1],wrange[2])
    S2 = np.zeros(len(w2)) 
    for wi, w in enumerate(w2):
      dA = A - O*(w+1j*dump) 
      Rx = np.linalg.solve(dA,Px)
      Ry = np.linalg.solve(dA,Py)
      Rz = np.linalg.solve(dA,Pz)
      S2[wi]  = 2.*np.einsum("p,p->",Rx,Px).imag
      S2[wi] += 2.*np.einsum("p,p->",Ry,Py).imag
      S2[wi] += 2.*np.einsum("p,p->",Rz,Pz).imag

    #plt.plot(w1,S1/max(S1))
    #plt.plot(w2,S2/max(S2))
    #plt.show()     

    

    print(nocc)
    print(nmo)
    print(nvir)

    return
