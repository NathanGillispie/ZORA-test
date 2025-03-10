import os
import sys
import numpy as np
import constants

def read_input(input_lines,options):
    """Parse into options"""
    
    read_mol = False
    count = 0
    atoms = []
    coords = {} ## why not just make it a 2d list and get rid of the dictionary stuff?
    ## this: coords[str(atoms[count])+str(count+1)] just turns into this coords[count]
    ## you could store [Z, [x,y,z]]. Also it doesn't have to be a numpy array.
    units = 'angstroms'

    for line in input_lines:
        if "#" in line:
           continue
        if line.startswith("end molecule"):
            read_mol = False
        if (len(line.split()) > 1) and (read_mol is True):            
            atoms.append(line.split()[0])
            coords[str(atoms[count])+str(count+1)] = np.array((float(line.split()[1]),float(line.split()[2]),float(line.split()[3])))
            count += 1
        if line.startswith("start molecule"):
            read_mol = True
            count = 0
    # TODO: add case to exit if not valid (need to change atoms)
    options.atoms = atoms
    options.coords = coords

    # everything else
    for line in input_lines:
        if "#" in line:
            continue

        if "basis" in line:
            basis_txt = line.split()
            if len(basis_txt) == 2:
              options.basis = basis_txt[1]
              continue
            options.basis = os.path.abspath(os.path.dirname(__file__))+"/basis/"+str(line.split()[2])

        if "method" in line:
            options.method = str(line.split()[1])

        if "xc" in line:
            if len(line.split()) == 3: #read exchange and correlation separately
              options.xc = str(line.split()[1])+","+str(line.split()[2])
              print(options.xc)
            elif len(line.split()) == 2: #read alias to xc functional
              options.xc = str(line.split()[1]) #+","
        if "e_conv" in line:
            options.e_conv   = float(line.split()[1])

        if "d_conv" in line:
            options.d_conv   = float(line.split()[1])

        if "ft_gamma" in line:
            options.gamma   = float(line.split()[1])

        if "maxiter" in line:
            options.maxiter   = int(line.split()[1])

        if "nroots" in line:
            options.nroots = int(line.split()[1])

        if "grid_level" in line:
            options.grid_level = int(line.split()[1])
        
        if "batch_size" in line:
            options.batch_size = int(line.split()[1])

        if "guess_mos" in line:
            options.guess_mos = str(line.split()[1])
            options.guess_mos_provided = True
        if "cosmo" in line:
            if len(line.split()) > 1:
              options.cosmo_epsilon = float(line.split()[1])
            options.cosmo = True

        if "occupied" in line:
            options.occupied = [int(line.split()[1]), int(line.split()[2])]
            options.cvs = True

        if "virtual" in line:
            options.virtual = [int(line.split()[1]), int(line.split()[2])]
            options.reduced_virtual = True

        if "couple_states" in line:
            options.occupied1 = [int(line.split()[1]), int(line.split()[2])]
            options.occupied2 = [int(line.split()[3]), int(line.split()[4])]
            options.couple_states = True

        if "fourier_transform" in line:
            options.fourier_transform = True

        if "frequencies" in line:
            if "eV" in line:
              options.frequencies = np.asarray([float(line.split()[1]), float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])*constants.ev2au
            else: 
              options.frequencies = np.asarray([float(line.split()[1]), float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])

        # set all booleans with the same name
        bools = ["so_scf","write_mos","write_molden","maximum_ovlp","noscf","tdscf_in_core","direct_diagonalization","in_core","do_tda","do_cis","do_tdhf","do_cpp","nocom"]
        for bool_opt in bools:
            if bool_opt in line:
                setattr(options, bool_opt, True)

        if "keep_gridpoints" in line:
            options.keep_aogridpoints = True
        
        if "cartesian" in line:
            options.cartesian = True

        if "spherical" in line:
            options.cartesian = False

        if "sflip" in line:
            options.spin_flip = True

        if "akonly" in line:
            options.akonly = True
            options.nofxc = True
        if "fonly" in line:
            options.fonly = True
            options.nofxc = True
        if "jkonly" in line:
            options.jkonly = True
            options.nofxc = True
        if "jonly" in line:
            options.jonly = True
            options.nofxc = True
        if "nofxc" in line:
            options.nofxc = True

        if "charge" in line:
            options.charge = float(line.split()[1])

        if "spin" in line:
            options.spin = int(line.split()[1])

        if "mult" in line:
            options.mult = int(line.split()[1])
            options.spin = int(line.split()[1])-1 

        if "units" in line:
            units = str(line.split()[1])

        if "swap_mos" in line:
           orig_str = line.split(",")[0].split()[1:]
           swap_str = line.split(",")[1].split()
           if len(orig_str) != len(swap_str):
             exit("Incorrect dimensions for orbital swap!")
           orig = [int(i)-1 for i in orig_str]
           swap = [int(i)-1 for i in swap_str]
  
           options.swap_mos = [orig,swap]

        if "swap_alpha_mos" in line:
           orig_str = line.split(",")[0].split()[1:]
           swap_str = line.split(",")[1].split()
           if len(orig_str) != len(swap_str):
             exit("Incorrect dimensions for orbital swap!")
           orig = [int(i)-1 for i in orig_str]
           swap = [int(i)-1 for i in swap_str]
  
           options.swap_alpha_mos = [orig,swap]

        if "swap_beta_mos" in line:
           orig_str = line.split(",")[0].split()[1:]
           swap_str = line.split(",")[1].split()
           if len(orig_str) != len(swap_str):
             exit("Incorrect dimensions for orbital swap!")
           orig = [int(i)-1 for i in orig_str]
           swap = [int(i)-1 for i in swap_str]
  
           options.swap_beta_mos = [orig,swap]

        if "diis" in line:
            if line.split()[1] == "true":
                options.diis   = True
            elif line.split()[1] == "false":
                options.diis   = False
        if "do_triplet" in line:
            if line.split()[1] == "true":
                options.do_triplet   = True
            elif line.split()[1] == "false":
                options.do_triplet = False

        if "relativistic" in line:
            options.relativistic = line.split()[1]

        if "memory" in line:
            options.memory = float(line.split()[1])

        if "atomic_J" in line:
            options.atomic_J = True
            options.atomic_J_list = [int(i)-1 for i in line.split()[1:]]
        if "perturb_J" in line:
            options.perturb_J = True

        if "B_field_amplitude" in line:
            options.B_field_amplitude = float(line.split()[1])
        if "E_field_amplitude" in line:
            options.E_field_amplitude = float(line.split()[1])
        if "B_field_polarization" in line:
            options.B_field_polarization = int(line.split()[1])
        if "E_field_polarization" in line:
            options.E_field_polarization = int(line.split()[1])

        if "roots_lookup_table" in line:
            low = int(line.split()[1])
            high = int(line.split()[2])+1
            options.roots_lookup_table = np.arange(low,high,1)

    adjust_coords(options, units) ## adjusting after parsing main options because units may change

    gen_pyscf_mol_str(options)
    
    print_molecule_info(options)
    print_method_info(options)
    print_basis_info(options)

def adjust_coords(options, units):
    atoms = options.atoms
    coords = options.coords
    if (units.lower()[0] != 'b'): #following pyscf conventions
        for Ai, A in enumerate(atoms):
            coords[A+str(Ai+1)] *= constants.angs2bohr

    #center of mass
    mass = np.zeros((len(atoms)))
    com  = np.zeros((3))
    for Ai, A in enumerate(atoms):
        xyz = coords[A+str(Ai+1)]
        mass[Ai] = constants.masses[A.upper()]
        com += xyz * mass[Ai] 

    options.com = com

    if not options.nocom:
        for Ai, A in enumerate(atoms):
            coords[A+str(Ai+1)] -= com/np.sum(mass)
    options.coords = coords

def get_nelectrons(atoms):
    nel = 0
    for atom in atoms: 
        nel += constants.Z[atom.upper()]
    return nel

def gen_pyscf_mol_str(options):
    pyscf_mol_str = [] #Format molecule string as required by PySCF: "A1 0 0 0; A2 1 1 1"
    for index, atom in enumerate(options.atoms):
        crds = options.coords[atom+str(index+1)]
        atom_crds = [atom, crds[0], crds[1], crds[2]]
        pyscf_mol_str.append(' '.join(map(str,atom_crds)))
    options.pyscf_mol_str= ';'.join(pyscf_mol_str)

def print_molecule_info(options):
    atoms = options.atoms
    coords = options.coords # molecules["coords"]
    n_electrons = get_nelectrons(atoms) - options.charge
    mult = options.spin+1
    charge = options.charge
    natoms = len(atoms)
    print("    Molecule Info")
    print("    -------------")
    print("")
    print("    Number of atoms    : %i"%(natoms))
    print("    Number of electrons: %i"%(n_electrons))
    print("    Charge             : %i"%(charge))
    print("    Multiplicity       : %i"%(mult))
    print("    Geometry [a0]:")
    for a in range(natoms):
        atom = atoms[a]+str(a+1)
        print("    %5s %20.12f %20.12f %20.12f "%(atoms[a],\
        coords[atom][0],coords[atom][1],coords[atom][2]))
    print("")
    return None

def print_method_info(options):
    print("    Method: %s "%options.method)
    print("    E_conv: %e "%options.e_conv)
    print("    D_conv: %e "%options.d_conv)
    print("    Maxiter: %i "%options.maxiter)
    print("")
    return None

def print_basis_info(options):
    print("    Basis set: %s "%options.basis)
    return None
