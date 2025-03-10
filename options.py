class OPTIONS():
    def __init__(self):
      # Main options
      self.basis = '6-31g'
      self.method  = 'rhf'
      self.molecule = None
      self.grid_level = 3
      self.nocom = False

      # Convergence criteria
      self.e_conv  = 1e-6
      self.d_conv  = 1e-6
      self.maxiter = 200
      self.batch_size = 1024*1024
      self.nroots  = 5
      self.diis    = True
      self.memory = 6000 #MB

      self.do_tdhf = False
      self.do_cpp  = False
      self.do_cis = False
      self.do_tda  = False
      self.do_triplet = True
      self.relativistic = None
      self.so_scf  = False
      self.noscf   = False
      self.cvs     = False #Core valence separation
      self.tdscf_in_core = False
      self.in_core = False
      self.direct_diagonalization = False

      self.frequencies = None
      self.guess_mos_provided = False
      self.fourier_transform = False

      self.cosmo = False
      self.cosmo_epsilon = 78.3553 #water solvent 
      self.couple_states = None
      self.gamma = 0.25
      self.cartesian = False
      self.reduced_virtual = False

      self.plus_tb = False #tight binding

      self.akonly = False
      self.fonly = False
      self.jkonly = False
      self.jonly = False
      self.nofxc = False
      self.spin_flip = False
      self.roots_lookup_table = []
      self.B_field_amplitude = 0.
      self.E_field_amplitude  = 0. 
      self.B_field_polarization = 2
      self.E_field_polarization = 2
      self.swap_mos = []
      self.swap_alpha_mos = []
      self.swap_beta_mos = []
      self.atomic_J = False
      self.atomic_J_list = None
      self.perturb_J = False

      self.keep_aogridpoints = False
      self.write_molden = False
      self.write_mos = False
      self.maximum_ovlp = False

      
      ## Real-time SCF Options
      self.time_step = 0.01
      self.total_time_steps = 5000
      self.do_ft = False
      self.external_field_polarization = 2
      self.external_field_frequency = 0.5
      self.maximum_field_amplitude = 1e-4
      

