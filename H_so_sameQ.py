import pickle as pkl
import numpy as np

def init_hso():
    with open('H_old.mat', 'rb') as f:
        global H
        H = pkl.load(f)

    with open('H_new.mat', 'rb') as f:
        global Hnew
        Hnew = pkl.load(f)

def diff(i, j):
    a = H[i,j]
    b = Hnew[i,j]
    d_re = abs(b.real - a.real)
    d_im = abs(b.imag - a.imag)
    return float(d_re), float(d_im)

def total_err():
    nbf = len(H)
    re_cum = 0
    im_cum = 0
    for i in range(nbf):
        for j in range(len(H[i])):
            d = diff(i,j)
            re_cum += d[0]
            im_cum += d[1]
    print("Cumulative real error:", re_cum)
    print("Cumulative imag error:", im_cum)

def total_err_veff():
    H = []
    Hnew = []
    with open('veff_old.mat', 'rb') as f:
        H = pkl.load(f)
    with open('veff_new.mat', 'rb') as f:
        Hnew = pkl.load(f)
    cum = 0
    diff = np.abs(H-Hnew)
    large_ds = [d for d in diff if d > 10]
    print("Cumulative error:", diff.sum())
    #print("Big errors:", large_ds)

def T_total_err():
    H = []
    Hnew = []
    with open('pyscf_T.mat', 'rb') as f:
        H = np.loadtxt(f)
    with open('bohr_zT.mat', 'rb') as f:
        Hnew = np.loadtxt(f)
    diff = np.abs(H-Hnew).flatten()
    large_ds = len([d for d in diff if d > .00159])
    print("Cumulative error:", diff.sum())
    print("Chemically relevant errors ratio:", float(large_ds/len(Hnew)**2))

if __name__ == "__main__":
    #init_hso()
    #total_err()
    #total_err_veff()
    T_total_err()
