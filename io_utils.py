import numpy as np
from scipy.io import FortranFile
import constants
from pyscf.data import elements

def save_options(opts, *args):
  import json

  d = opts.__dict__
  d['com'] = d.pop('com').tolist()

  coords = d.pop('coords')
  new_coords = {}
  for atom in coords:
    new_coords[atom] = coords[atom].tolist() 
  d['coords'] = new_coords

  if len(args) == 0:
      with open('options.json', 'w') as f:
          json.dump(d, f, ensure_ascii=True, indent=4, sort_keys=True)
  else:
      with open(str(args[0]), 'w') as f:
          json.dump(d, f, ensure_ascii=True, indent=4, sort_keys=True)

def write_mos(C,nel,mos_filename):
#   if pol == 1:
#     nbf = len(C) 
#     nmo = len(C[0]) 
#     print(nbf,nmo)
#     header_num = np.array([pol,nbf,nmo])
#     header     = bytearray(header_num)
#     f = open(mos_filename,"wb")
#     f.write(header)
#     C = C.reshape(nbf*nmo)
#     for i in range(nbf*nmo):
#       f.write(bytearray(C[i]))
#     f.close()
#     print("File written to %s!"%mos_filename)
#   if pol == 2:
    C = np.asarray(C)
    nbf = len(C[0]) 
    nmo = len(C[0][0]) 
    header_num = np.array([nbf,nmo,nel[0],nel[1]])
    header     = bytearray(header_num)
    f = open(mos_filename,"wb")
    f.write(header)
    C = C.reshape(2*nbf*nmo)
    for i in range(2*nbf*nmo):
      f.write(bytearray(C[i]))
    f.close()
    print("File written to %s!"%mos_filename)

def read_real_mos(mos_filename):
#   if pol == 1:
#     f = open(mos_filename,"rb")
#     index = 0
#     f.seek(index)
#     pol, nbf, nmo = np.fromfile(f,dtype="int64",count=3)
#     index += 3*8
#     C = np.zeros(nbf*nmo,dtype=complex)
#     f.seek(index)
#     C = np.fromfile(f,dtype="float64",count=nbf*nmo)
#     C = C.reshape(nbf,nmo)
#     f.close()
#     print("MOs read from %s"%(mos_filename))
#   if pol == 2:
   f = open(mos_filename,"rb")
   index = 0
   f.seek(index)
   nbf, nmo, nela, nelb = np.fromfile(f,dtype="int64",count=4)
   index += 4*8
   C = np.zeros(2*nbf*nmo)
   f.seek(index)
   C = np.fromfile(f,dtype="float64",count=2*nbf*nmo)
   C = C.reshape(2,nbf,nmo)
   f.close()
   print("MOs read from %s"%(mos_filename))
   return C, [nela,nelb]  

#the code below accounts for complex orbitals
def read_complex_mos(mos_filename):
   f = open(mos_filename,"rb")
   index = 0
   f.seek(index)
   nbf, nmo, nela, nelb = np.fromfile(f,dtype="int64",count=4)
   index += 4*8
   C = np.zeros(2*nbf*nmo,dtype=complex)
   f.seek(index)
   C = np.fromfile(f,dtype="complex",count=2*nbf*nmo)
   C = C.reshape(2,nbf,nmo)
   f.close()
   print("Complex MOs read from %s"%(mos_filename))
   return C, [nela,nelb]  


def write_ao_grid(AO,aos_filename,xctype):
   if xctype == "LDA":  
     npoints = len(AO)
     nbf     = len(AO[0]) 
     header_num = np.array([npoints,nbf])
     header     = bytearray(header_num)
     f = open(aos_filename,"wb")
     f.write(header)
     AO = AO.reshape(npoints*nbf)
     for i in range(npoints*nbf):
       f.write(bytearray(AO[i]))
     f.close()
   else:
     npoints = len(AO[0])
     nbf     = len(AO[0][0]) 
     header_num = np.array([npoints,nbf])
     header     = bytearray(header_num)
     f = open(aos_filename,"wb")
     f.write(header)
     AO = AO.reshape(4*npoints*nbf)
     for i in range(4*npoints*nbf):
       f.write(bytearray(AO[i]))
#     for i in range(npoints*nbf):
#       f.write(bytearray(AO[1][i]))
#     for i in range(npoints*nbf):
#       f.write(bytearray(AO[2][i]))
#     for i in range(npoints*nbf):
#       f.write(bytearray(AO[3][i]))
     f.close()

angmom = {0:"s",1:"p",2:"d",3:"f",4:"g"}

def write_molden(wfn,filename):
     basis = list(wfn.ints_factory._basis.items())
     basis_set = {}  
     for bi, b in enumerate(basis):
       basis_set[b[0]] = b[1:]

     f = open(filename+".molden","w")
     f.write("[Molden Format]\n")   
     f.write("[Atoms] AU\n")
     atoms = wfn.ints_factory.atom.split(";")
     for i, atom in enumerate(atoms):
       label = atom.split()
       el = elements.ELEMENTS.index(label[0])
       f.write("%4s %4s %4s %12.8f %12.8f %12.8f \n"%(label[0], str(i+1), el, float(label[1]), float(label[2]), float(label[3]))) 
     f.write("[GTO]\n")
     for i, atom in enumerate(atoms):
       label = atom.split()
       f.write("%5i %5i\n"%(i+1,0))
       basis = basis_set[label[0]]
       for b in basis:
         ncont = len(b)
         for cont in b:
           nprim = len(cont)-1
           f.write("%2s %5i %5i\n"%(angmom[cont[0]], nprim, 0))   
           for prim in range(1,nprim+1):
             f.write("%20.12f %20.12f \n"%(cont[prim][0],cont[prim][1])) 
       f.write("\n") 
#     if (wfn.reference == "gks" or wfn.reference == "rgks"):
#       for i, atom in enumerate(atoms):
#         label = atom.split()
#         f.write("%5i %5i\n"%(i+1,0))
#         basis = basis_set[label[0]]
#         for b in basis:
#           ncont = len(b)
#           for cont in b:
#             nprim = len(cont)-1
#             f.write("%2s %5i %5i\n"%(angmom[cont[0]], nprim, 0))
#             for prim in range(1,nprim+1):
#               f.write("%20.12f %20.12f \n"%(cont[prim][0],cont[prim][1]))
#         f.write("\n")
     if wfn.options.cartesian is True:
       f.write("[6D]\n")
       f.write("[10F]\n")
       f.write("[15G]\n")
       f.write("[MO]\n")
     else:
       f.write("[5D]\n")
       f.write("[7F]\n")
       f.write("[9G]\n")
       f.write("[MO]\n")

     if wfn.reference == "gks" or wfn.reference == "rgks":
       offset = len(wfn.molden_reorder)
       for p in range(len(wfn.C[0])):
         Ca = np.sum(wfn.C[0][:wfn.nbf,p])
         Cb = np.sum(wfn.C[0][wfn.nbf:,p])
         f.write("Sym  = A\n")
         if abs(Ca) > abs(Cb):
           f.write("Spin = Alpha\n")
         else:
           f.write("Spin = Beta \n")
         f.write("Ene  = %12.8f\n"%(wfn.eps[0][p])) 
         if p < wfn.nel[0]:
           f.write("Occup = 1\n")
         else:
           f.write("Occup = 0\n")
       
         if abs(Ca) > abs(Cb):
           for mu_i,mu in enumerate(wfn.molden_reorder):
             f.write("%5i %20.12f\n"%(mu_i+1,(wfn.C[0][mu][p]).real)) 
         else:
           for mu_i,mu in enumerate(wfn.molden_reorder):
             f.write("%5i %20.12f\n"%(mu_i+1,(wfn.C[0][mu+offset][p]).real)) 
#       for p in range(len(Cb)):
#         f.write("Sym  = A\n")
#         #f.write("Spin = Beta\n")
#         f.write("Ene  = %12.8f\n"%(wfn.eps[0][p])) 
#         f.write("Occup = 0\n")
#         for mu_i,mu in enumerate(wfn.molden_reorder):
#           f.write("%5i %20.12f\n"%(mu_i+1,(Cb[mu][p]).real)) 
     else:
       for p in range(len(wfn.C[0][0])):
         f.write("Sym  = A\n")
         f.write("Ene  = %12.8f\n"%(wfn.eps[0][p])) 
         f.write("Spin = Alpha\n")
         if p < wfn.nel[0]:
           f.write("Occup = 1\n")
         else:
           f.write("Occup = 0\n")
         for mu_i, mu in enumerate(wfn.molden_reorder):
           f.write("%5i %20.12f\n"%(mu_i+1,wfn.C[0][mu][p])) 
       for p in range(len(wfn.C[1][0])):
         f.write("Sym  = A\n")
         f.write("Ene  = %12.8f\n"%(wfn.eps[1][p])) 
         f.write("Spin = Beta\n")
         if p < wfn.nel[1]:
           f.write("Occup = 1\n")
         else:
           f.write("Occup = 0\n")
         for mu_i, mu in enumerate(wfn.molden_reorder):
           f.write("%5i %20.12f\n"%(mu_i+1,wfn.C[1][mu][p])) 
#       
##     print(ints.atom)
#     for atom, basis in ints._basis.items():
#        print(atom)
#        for b in basis:
#           print(b)
##          if isinstance(b[1], int):
##            kappa = b[1]
##            coeff = b[2:]
##          else:
##            kappa = 0
##            coeff = b[1:]
##          nprim = len(coeff)
##          nctr  = len(coeff[0])-1  
##     for i, atom in enumerate(atoms):
##       label = atom.split()
##       print(basis[label[0]])
##     print(ints.atoms())
##
##
     f.close()



def read_ao_grid(f,xctype):
   if xctype == "LDA":  
     index = 0
     f.seek(index)
     npoints, nbf = np.fromfile(f,dtype="int64",count=2)
     index += 2*8
     AO = np.zeros(npoints*nbf)
     f.seek(index)
     AO = np.fromfile(f,dtype="float64",count=npoints*nbf)
     AO = AO.reshape(npoints,nbf)
     f.close()
   else:
     index = 0
     f.seek(index)
     npoints, nbf = np.fromfile(f,dtype="int64",count=2)
     index += 2*8
     AO = np.zeros(4*npoints*nbf)
     f.seek(index)
     AO = np.fromfile(f,dtype="float64",count=4*npoints*nbf)
#     index += npoints*nbf*8
#     f.seek(index)
#     AO[1] = np.fromfile(f,dtype="float64",count=npoints*nbf)
#     index += npoints*nbf*8
#     f.seek(index)
#     AO[2] = np.fromfile(f,dtype="float64",count=npoints*nbf)
#     index += npoints*nbf*8
#     f.seek(index)
#     AO[3] = np.fromfile(f,dtype="float64",count=npoints*nbf)
#     index += npoints*nbf*8
     AO = AO.reshape(4,npoints,nbf)
     f.close()
   return AO  
