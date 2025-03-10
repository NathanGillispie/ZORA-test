#!/usr/bin/python3

'''
**************************
BOHR [DEVELOPER'S VERSION]
**************************
(c) DR Nascimento Lab 20XX

Usage
----------
As a module:
>>> import bohr
>>> opts = bohr.build_options(input_lines)
>>> opts.someoption = somevalue
>>> pyscf_mol = bohr.build_mol(opts)
>>> bohr.run(pyscf_mol, opts)
'''
import input_parser
import sys
import os
import time
from pyscf import gto

import options
import wavefunction
import real_time
import tdscf
import ccd

def get_input():
  if len(sys.argv) == 1:
    exit("No input file!\n\t./bohr.py input.in")
  input_lines = []
  inputfile = sys.argv[1]
  with open(inputfile, 'r') as f:
    input_lines = f.readlines()
  if input_lines == []:
    exit("Input file empty...\n❌GAME OVER❌\nTotal score: 0 Hartrees")
  return input_lines, inputfile

def build_options(input_lines):
  """Returns a new options object from the provided input lines"""
  opts = options.OPTIONS()
  input_parser.read_input(input_lines,opts)
  if __name__!='__main__': 
    opts.inputfile = os.getcwd()
  return opts

def build_mol(options):
  """Returns a new pyscf mol object from the provided options"""
  pyscf_mol = gto.M(atom = options.pyscf_mol_str, basis = options.basis, unit='B', charge = int(options.charge), spin = int(options.spin), cart=options.cartesian)
  pyscf_mol.set_common_origin(options.com)
  pyscf_mol.verbose = 0
  pyscf_mol.max_memory = options.memory
  pyscf_mol.build()
  return pyscf_mol

def run(pyscf_mol, options):
  tic = time.time()
  match options.method:
    case 'rhf':
      rhf_wfn    = wavefunction.RHF(pyscf_mol)
      rhf_energy = rhf_wfn.compute(options)
      print("Setting method as rhf")
      if options.do_tdhf is True:
        real_time.compute(rhf_wfn)
      elif options.do_cpp is True:
        exit("CPP method under maintenance :(")
        #print("    Complex Polarization Propagator Algorithm requested ...")
        #cpp.compute(rhf_wfn)
    case 'td-rhf':
      rhf_wfn    = wavefunction.RHF(pyscf_mol)
      rhf_energy = rhf_wfn.compute(options)
      if options.do_tda is True:
        tdscf = tdscf.TDSCF(rhf_wfn)
        tdscf.tda()
      else:
        tdrks_energy = tdscf.rpa(rhf_wfn)
    case 'uhf':
      uhf_wfn = wavefunction.UHF(pyscf_mol)
      uhf_energy = uhf_wfn.compute(options)
      if options.do_tdhf is True:
        real_time.compute(uhf_wfn)
    case 'td-uhf':
      uhf_wfn    = wavefunction.UHF(pyscf_mol)
      uhf_energy = uhf_wfn.compute(options)
      if options.do_tda is True:
        tdscf = tdscf.TDSCF(uhf_wfn)
        tdscf.tda()
      else:
        tdscf_energy = tdscf.rpa(uhf_wfn)
    case 'td-ghf':
      ghf_wfn    = wavefunction.GHF(pyscf_mol)
      ghf_energy = ghf_wfn.compute(options)
      if options.do_tda is True:
        tdscf = tdscf.TDSCF(ghf_wfn)
        tdscf.tda()
      else:
        tdscf_energy = tdscf.rpa(ghf_wfn)
    case 'rks':
      rks_wfn    = wavefunction.RKS(pyscf_mol)
      rks_energy = rks_wfn.compute(options)
    #  if options.do_tdhf is True:
    #    real_time.compute(rhf_wfn)
    case 'uks':
      uks_wfn = wavefunction.UKS(pyscf_mol)
      uks_energy = uks_wfn.compute(options)
    case 'roks':
      roks_wfn = wavefunction.ROKS(pyscf_mol)
      roks_energy = roks_wfn.compute(options)
    case 'gks':
      gks_wfn = wavefunction.GKS(pyscf_mol)
      gks_energy = gks_wfn.compute(options)
    case 'rgks':
      rgks_wfn = wavefunction.RGKS(pyscf_mol)
      rgks_energy = rgks_wfn.compute(options)
    case 'td-rks':
      rks_wfn    = wavefunction.RKS(pyscf_mol)
      rks_energy = rks_wfn.compute(options)
      if options.do_tda is True:
        tdscf = tdscf.TDSCF(rks_wfn)
        tdscf.tda()
      else:
        tdrks_energy = tdscf.rpa(rks_wfn)
    case 'td-uks':
      uks_wfn = wavefunction.UKS(pyscf_mol)
      uks_energy = uks_wfn.compute(options)
      if options.do_tda is True:
        tdscf = tdscf.TDSCF(uks_wfn)
        tdscf.tda()
      else:
        tduks_energy = tdscf.rpa(uks_wfn)
    case 'uks-pyscf':
      uks_wfn = wavefunction.UKS_PYSCF(pyscf_mol)
      uks_energy = uks_wfn.compute(options)
    case 'td-gks':
      gks_wfn = wavefunction.GKS(pyscf_mol)
      gks_energy = gks_wfn.compute(options)
      if options.do_tda is True:
        tdscf = tdscf.TDSCF(gks_wfn)
        tdscf.tda()
      else:
        tdgks_energy = tdscf.rpa(gks_wfn)
    case 'td-rgks':
      rgks_wfn = wavefunction.RGKS(pyscf_mol)
      rgks_energy = rgks_wfn.compute(options)
      if options.do_tda is True:
        tdscf = tdscf.TDSCF(rgks_wfn)
        tdscf.tda()
      else:
        tdrgks_energy = tdscf.rpa(rgks_wfn)
    case 'rt-td-gks':
      gks_wfn = wavefunction.GKS(pyscf_mol)
      gks_energy = gks_wfn.compute(options)
      rt = real_time.RTTDSCF(gks_wfn)
      w, fw = rt.propagate_rk4()
    case 'rt-td-rks':
      rks_wfn = wavefunction.RKS(pyscf_mol)
      rks_energy = rks_wfn.compute(options)
      rt = real_time.RTTDSCF(rks_wfn)
      w1, fw1 = rt.propagate_rk4()
    #  w2, fw2 = rt.propagate_simpson()
    case 'uccd':
      uks_wfn = wavefunction.UKS(pyscf_mol)
      uks_energy = uks_wfn.compute(options)
      ccd.uccd_energy(uks_wfn)
    #  if options.do_tdhf is True:
    #    real_time.compute(uhf_wfn)
    case other:
      print("Invalid method")
      exit(1)

  if options.keep_aogridpoints is False:
    import os
    if os.path.isdir("ao_gridpoints") is True:
      import shutil
      shutil.rmtree("ao_gridpoints")

  toc = time.time()
  print("\n    BOHR TIME: %f s"%(toc-tic))

if __name__ == '__main__':
  input_lines, inputfile = get_input()
  options = build_options(input_lines)
  options.inputfile = inputfile ## Temporary hack, should be removed later when we make bohr os independent
  pyscf_mol = build_mol(options)

  run(pyscf_mol, options)
