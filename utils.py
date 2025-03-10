import numpy as np
import scipy as sp
import os
from scipy import special
#from lebedev import *
#from gauss_chebyshev import *
#from becke import *

#get basis set information
def read_basis(basis, atoms):
    d = {}
    a = {}
    l = {}
    f = open(sys.path.insert(1,os.path.abspath(os.path.dirname(__file__)+"/basis/"))+basis["name"]+".gbs","r")
    count = 0
    shell = 0
    atype = None
    nprim = 1e5
    alpha  = {}
    contr  = {}
    amomns = []
    for line in f:
#        print(count)
        if count == (nprim + 3):
            count = 2

        if count > 2:
            line = line.replace("D","e",3)
            for i, subshell in enumerate(amomn):
#                print(i,str(shell)+subshell)
                if count == 3:
                    alpha[str(shell)+"-"+subshell] = []
                    contr[str(shell)+"-"+subshell] = []
                alpha[str(shell)+"-"+subshell].append(float(line.split()[0]))
                contr[str(shell)+"-"+subshell].append(float(line.split()[i+1]))
            count += 1

        if count == 2:
            if ("****" in line):
                a[atype] = alpha
                d[atype] = contr
                l[atype] = amomns
                count = 0
            else:
                shell += 1 
                amomn = line.split()[0]
                for subshell in amomn:
                    amomns.append(str(shell)+"-"+subshell)
                nprim = int(line.split()[1])   
                count += 1
        
        if count == 1:
            if line.split()[0] in atoms:
                alpha  = {}
                contr  = {}
                amomns = []
                atype = line.split()[0]
                shell = 0
                count += 1

        if ("****" in line) and (count == 0):
            count += 1

    return l, d, a       

def lorentzian(wrange,dump,roots,os):
    nw = int((float(wrange[1]) - float(wrange[0]))/float(wrange[2]))
    w = np.zeros(nw)
    S = np.zeros(nw)
    for windex in range(nw):
        w[windex] = (float(wrange[0]) + float(wrange[2]) * windex)
        for root in range(len(roots)):
            S[windex] += os[root]*dump/((w[windex]-roots[root])**2 + dump**2)
        S[windex] /= (1.0 * np.pi)
    return w, S    
