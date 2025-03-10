import numpy as np

class DIIS():
    def __init__(self, wfn,is_complex):
      self.is_complex = is_complex
      self.diis_max = 20
      self.diis_dim = 0
      if is_complex is not False:
        print("   WARNING: Enabling Complex DIIS")
        self.F_vecs = np.zeros((self.diis_max,wfn.nbf,wfn.nbf),dtype=complex)
        self.e_vecs = np.zeros((self.diis_max,wfn.nbf,wfn.nbf),dtype=complex)
      else:
        print("   WARNING: Enabling Real DIIS")
        self.F_vecs = np.zeros((self.diis_max,wfn.nbf,wfn.nbf))
        self.e_vecs = np.zeros((self.diis_max,wfn.nbf,wfn.nbf))
      self.diis_init = False
      self.curdim = 0
      self.diis_alpha = 0.75

    def do_diis(self,F,D,S,X):
 
      #e = np.conj(X.T)@(F@D@S - np.conj(S@D@F))@X
      e = np.conj(X.T)@(F@D@S - np.conj(F@D@S).T)@X
      #e = np.conj(X.T)@(F@D@S - S@D@F)@X
      emax = np.max(np.abs(e))
      #print(self.curdim)
      if (emax > 1.2): # or (emax < 1e-4):
        self.is_diis = "DAMP"
        i = self.curdim
        self.F_vecs[i] = np.conj(X.T)@F@X
        if i == 0:
          self.curdim += 1
          return self.F_vecs[i], emax
        else:
          self.curdim = 0
          newF = self.diis_alpha * self.F_vecs[0] + (1.-self.diis_alpha) * self.F_vecs[1]
          return newF, emax

      #elif (self.curdim < 2):
      #  self.is_diis = "DAMP"
      #  i = self.curdim
      #  self.F_vecs[i] = np.conj(X.T)@F@X
      #  self.curdim += 1
      #  return self.F_vecs[i], emax

      else:
        self.is_diis = "DIIS"
        dim = self.diis_dim
        if dim == self.diis_max-1:
          print("    Collapsing DIIS Subspace")
          self.diis_dim = 0

        self.e_vecs[dim] = e
        self.F_vecs[dim] = np.conj(X.T)@F@X
    
        if self.is_complex is not False:
           B = np.ones((dim+2,dim+2),dtype=complex)
        else:
           B = np.ones((dim+2,dim+2))
        Bred = np.einsum("Aqp,Bpq->AB",np.conj(self.e_vecs[:dim+1]),self.e_vecs[:dim+1],optimize=True)
        B[:dim+1,:dim+1] = Bred
        B[-1][-1] = 0. 

        y = np.zeros(dim+2)
        y[-1] = 1.

       # #check if the B matrix is invertible
       # detB = np.linalg.det(B)
       # if abs(detB) < 1e-12:
       #   #print("  Bad B matrix, collapsing")
       #   self.curdim = 0
       #   #return Fock matrix with smallest error
       #   for i in range(1,self.diis_dim+1):
       #     Bred = B[i:,i:]
       #     yred = y[i:]
       #     Fred = self.F_vecs[i:self.diis_dim+1]
       #     ered = self.e_vecs[i:self.diis_dim+1]
       #     dim -= 1
       #     detBred = np.linalg.det(Bred)
       #     if abs(detBred) > 1e-8:
       #       B = Bred
       #       y = yred
       #       self.F_vecs[:dim+1] = Fred
       #       self.e_vecs[:dim+1] = ered
       #       self.diis_dim = dim
       #       break
       # #print(B.real)     
       # #solve system of linear equations
        c = np.linalg.solve(B,y)
        #build extrapolated Fock matrix
        newF = np.einsum("Apq,A->pq",self.F_vecs[:dim+1],c[:dim+1],optimize=True)
        self.diis_dim += 1

        return newF, emax

    def do_ediis(self,F,C,S,X,eps):
 
      #e = np.conj(X.T)@(F@D@S - np.conj(S@D@F))@X
      e = np.conj(X.T)@(F@C - S@C@(np.eye(len(eps))*eps))@X
      #e = np.conj(X.T)@(F@D@S - S@D@F)@X
      emax = np.max(np.abs(e))
      #print(self.curdim)
      if (emax > 1.2): # or (emax < 1e-4):
        self.is_diis = "DAMP"
        i = self.curdim
        self.F_vecs[i] = np.conj(X.T)@F@X
        if i == 0:
          self.curdim += 1
          return self.F_vecs[i], emax
        else:
          self.curdim = 0
          newF = self.diis_alpha * self.F_vecs[0] + (1.-self.diis_alpha) * self.F_vecs[1]
          return newF, emax

      #elif (self.curdim < 2):
      #  self.is_diis = "DAMP"
      #  i = self.curdim
      #  self.F_vecs[i] = np.conj(X.T)@F@X
      #  self.curdim += 1
      #  return self.F_vecs[i], emax

      else:
        self.is_diis = "DIIS"
        dim = self.diis_dim
        if dim == self.diis_max-1:
          print("    Collapsing DIIS Subspace")
          self.diis_dim = 0

        self.e_vecs[dim] = e
        self.F_vecs[dim] = np.conj(X.T)@F@X
    
        if self.is_complex is not False:
           B = np.ones((dim+2,dim+2),dtype=complex)
        else:
           B = np.ones((dim+2,dim+2))
        Bred = np.einsum("Aqp,Bpq->AB",np.conj(self.e_vecs[:dim+1]),self.e_vecs[:dim+1],optimize=True)
        B[:dim+1,:dim+1] = Bred
        B[-1][-1] = 0. 

        y = np.zeros(dim+2)
        y[-1] = 1.

       # #check if the B matrix is invertible
       # detB = np.linalg.det(B)
       # if abs(detB) < 1e-12:
       #   #print("  Bad B matrix, collapsing")
       #   self.curdim = 0
       #   #return Fock matrix with smallest error
       #   for i in range(1,self.diis_dim+1):
       #     Bred = B[i:,i:]
       #     yred = y[i:]
       #     Fred = self.F_vecs[i:self.diis_dim+1]
       #     ered = self.e_vecs[i:self.diis_dim+1]
       #     dim -= 1
       #     detBred = np.linalg.det(Bred)
       #     if abs(detBred) > 1e-8:
       #       B = Bred
       #       y = yred
       #       self.F_vecs[:dim+1] = Fred
       #       self.e_vecs[:dim+1] = ered
       #       self.diis_dim = dim
       #       break
       # #print(B.real)     
       # #solve system of linear equations
        c = np.linalg.solve(B,y)
        #build extrapolated Fock matrix
        newF = np.einsum("Apq,A->pq",self.F_vecs[:dim+1],c[:dim+1],optimize=True)
        self.diis_dim += 1

        return newF, emax


def do_diis(F,D,S,Shalf,err_vec,Fdiis):
    diis_max = 10
    diis_count = 0
    e_rms = 1e6
    nbf = len(F)
    if len(err_vec[0]) == 1:
      dim = 0 
    else:
      dim = len(err_vec)

    #compute current error vector  
    e  = np.matmul(F.T,np.matmul(D,S))
    e -= np.matmul(S.T,np.matmul(D,F))
    eorth_it = np.matmul(Shalf.T,np.matmul(e,Shalf))
    currForth = np.matmul(Shalf.T,np.matmul(F,Shalf))
    e_curr = abs(np.trace(np.matmul(eorth_it.T,eorth_it)))
   
    #check whether current Fock matrix is accurate enough
    #if it is, add it to the DIIS subspace, if not return it 
    #in the orthogonal basis
#    if e_curr > 10.1:
#      return currForth, err_vec, e_rms, Fdiis, ""
#    else:  
    if dim <= diis_max:
      #augment error vector size
      new_eorth = np.zeros((dim+1,nbf,nbf))
      new_Forth = np.zeros((dim+1,nbf,nbf))

      #copy old error vectors
      for i in range(dim):
        new_eorth[i] = err_vec[i].real
        new_Forth[i] = Fdiis[i].real
      
      #populate new entry
      new_eorth[dim] = eorth_it.real
      new_Forth[dim] = currForth.real

      #build new B matrix
      newB = np.zeros((dim+2,dim+2))
      y = np.zeros((dim+2))
      newB[0][0] = 0.
      y[0] = 1
      for i in range(1,dim+2):
        newB[0][i] = 1.
        newB[i][0] = 1.
      
      #populate new B matrix        
      for i in range(dim+1):
        for j in range(dim+1):
          newB[i+1][j+1] = np.trace(np.matmul(new_eorth[i].T,new_eorth[j]))

      #solve system of linear equations
      c = np.linalg.solve(newB,y)

      #build extrapolated Fock matrix
      Forth = np.zeros((nbf,nbf))
      for k in range(dim+1):
        Forth += c[k+1] * new_Forth[k]

      rms = np.zeros((dim,dim))
      for i in range(dim):
        for j in range(dim):
          rms[i][j] = np.trace(np.matmul(c[i+1]*new_eorth[i].T,c[j+1]*new_eorth[j]))
      e_rms = np.trace(rms)
        
    else:
      #check largest error vector 
      norm = np.zeros((dim))
      for i in range(dim):
        norm[i] = abs(np.trace(np.matmul(err_vec[i].T,err_vec[i])))
     
      largest = np.argmax(norm)
      
      #replace largest error vector and corresponding Fock matrix with the new one
      new_eorth = err_vec
      new_eorth[largest] = eorth_it.real
      new_Forth = Fdiis
      new_Forth[largest] = currForth.real  

      #build new B matrix
      newB = np.zeros((dim+1,dim+1))
      y = np.zeros((dim+1))
      newB[0][0] = 0.
      y[0] = 1
      for i in range(1,dim+1):
        newB[0][i] = 1.
        newB[i][0] = 1.
      
      #populate new B matrix        
      for i in range(dim):
        for j in range(dim):
          newB[i+1][j+1] = np.trace(np.matmul(new_eorth[i].T,new_eorth[j]))

      #solve system of linear equations
      c = np.linalg.solve(newB,y)

      #build extrapolated Fock matrix
      Forth = np.zeros((nbf,nbf))
      for k in range(dim):
        Forth += c[k+1] * new_Forth[k]

      rms = np.zeros((dim,dim))
      for i in range(dim):
        for j in range(dim):
          rms[i][j] = np.trace(np.matmul(c[i+1]*new_eorth[i].T,c[j+1]*new_eorth[j]))
      e_rms = np.trace(rms)
        
    return Forth, new_eorth, e_rms, new_Forth, "DIIS"

def do_complex_diis(F,D,S,Shalf,err_vec,Fdiis):

    diis_max = 6
    diis_count = 0
    e_rms = 1e6
    nbf = len(F)
    if len(err_vec[0]) == 1:
      dim = 0 
    else:
      dim = len(err_vec)

    #compute current error vector  
    e  = np.matmul(F,np.matmul(D,S))
    e -= np.matmul(S,np.matmul(D,F))
    eorth_it = np.matmul(Shalf.T,np.matmul(e,Shalf))
    currForth = np.matmul(Shalf.T,np.matmul(F,Shalf))
    e_curr = np.abs(np.trace(np.matmul(np.conjugate(eorth_it.T),eorth_it)))
   
    #check whether current Fock matrix is accurate enough
    #if it is, add it to the DIIS subspace, if not return it 
    #in the orthogonal basis
    #if e_curr > 1.0e20:
    #  return currForth, err_vec, e_rms, Fdiis, ""
    #else:  
    if dim <= diis_max:
      #augment error vector size
      new_eorth = np.zeros((dim+1,nbf,nbf),dtype=complex)
      new_Forth = np.zeros((dim+1,nbf,nbf),dtype=complex)

      ##copy old error vectors
      #for i in range(dim):
      #  new_eorth[i] = err_vec[i]
      #  new_Forth[i] = Fdiis[i] 
      new_eorth[:dim] = err_vec
      new_Forth[:dim] = Fdiis 
      
      #populate new entry
      new_eorth[dim] = eorth_it
      new_Forth[dim] = currForth

      #build new B matrix
      newB = np.zeros((dim+2,dim+2),dtype=complex)
      y = np.zeros((dim+2),dtype=complex)
      newB[0][0] = 0.
      y[0] = 1

      newB[0,1:] = np.ones(dim+1)
      newB[1:,0] = np.ones(dim+1)
#      for i in range(1,dim+2):
#        newB[0][i] = 1.
#        newB[i][0] = 1.
      
      #populate new B matrix        
#      for i in range(dim+1):
#        for j in range(dim+1):
#          newB[i+1][j+1] = np.trace(np.matmul(np.conjugate(new_eorth[i].T),new_eorth[j].T))
#
#      print(newB.shape)
#      print(new_eorth.shape)
#      exit(0)
      
      newB[1:,1:] = np.einsum('ikl,jlk->ij',np.conjugate(new_eorth),new_eorth,optimize=True)

      #solve system of linear equations
      c = np.linalg.solve(newB,y)

      #build extrapolated Fock matrix
      #Forth = np.zeros((nbf,nbf),dtype=complex)
      Forth = np.einsum('k,kij->ij',c[1:],new_Forth,optimize=True)
#      for k in range(dim+1):
#        Forth += c[k+1] * new_Forth[k]
     
      


      #rms = np.zeros((dim,dim),dtype=complex)
      #for i in range(dim):
      #  for j in range(dim):
      #    rms[i][j] = np.trace(np.matmulyy(np.conjugate(c[i+1]*new_eorth[i].T),c[j+1]*new_eorth[j]))
      rms = np.einsum('i,ikl,j,jlk->ij',np.conjugate(c[1:]),np.conjugate(new_eorth),c[1:],new_eorth,optimize=True)
      e_rms = np.trace(rms)
        
    else:
      #check largest error vector 
      #norm = np.zeros((dim))
      #for i in range(dim):
      #  norm[i] = np.abs(np.trace(np.matmul(np.conjugate(err_vec[i].T),err_vec[i])))
      norm = np.abs(np.einsum('ijk,ikj->i',np.conjugate(err_vec),err_vec,optimize=True))
     
      largest = np.argmax(norm)
      
      #replace largest error vector and corresponding Fock matrix with the new one
      new_eorth = err_vec
      new_eorth[largest] = eorth_it
      new_Forth = Fdiis
      new_Forth[largest] = currForth  

      #build new B matrix
      newB = np.zeros((dim+1,dim+1),dtype=complex)
      y = np.zeros((dim+1),dtype=complex)
      newB[0][0] = 0.
      y[0] = 1
      #for i in range(1,dim+1):
      #  newB[0][i] = 1.
      #  newB[i][0] = 1.
      newB[0,1:] = np.ones(dim)
      newB[1:,0] = np.ones(dim)
      
      ##populate new B matrix        
      #for i in range(dim):
      #  for j in range(dim):
      #    newB[i+1][j+1] = np.trace(np.matmul(np.conjugate(new_eorth[i]),new_eorth[j].T))
      newB[1:,1:] = np.einsum('ikl,jlk->ij',np.conjugate(new_eorth),new_eorth,optimize=True)

      #solve system of linear equations
      c = np.linalg.solve(newB,y)

      #build extrapolated Fock matrix
      Forth = np.einsum('k,kij->ij',c[1:],new_Forth,optimize=True)
      #Forth = np.zeros((nbf,nbf),dtype=complex)
      #for k in range(dim):
      #  Forth += c[k+1] * new_Forth[k]

      #rms = np.zeros((dim,dim),dtype=complex)
      #for i in range(dim):
      #  for j in range(dim):
      #    rms[i][j] = np.trace(np.matmul(np.conjugate(c[i+1]*new_eorth[i].T),c[j+1]*new_eorth[j]))
      #e_rms = np.trace(rms)
      rms = np.einsum('i,ikl,j,jlk->ij',np.conjugate(c[1:]),np.conjugate(new_eorth),c[1:],new_eorth,optimize=True)
      e_rms = np.trace(rms)
        
    return Forth, new_eorth, e_rms, new_Forth, "DIIS"
