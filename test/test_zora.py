import pytest

from sys import path
path.append('/'.join(__file__.split('/')[:-2])+'/')

import bohr

@pytest.fixture(params=['gks-so','gks-sr','rks-sr','rgks-sr','uks-sr'])
def init_bohr(request):
    std_input="""
start molecule
  O      0.00000000     0.00000000     0.11726921
  H      0.75698224     0.00000000    -0.46907685
  H     -0.75698224     0.00000000    -0.46907685
end molecule

charge +1
spin 1
basis 6-31g 
diis true
relativistic zora
grid_level 5
e_convergence 1e-6
d_convergence 1e-6
maxiter 200
nroots 50
xc pbe0 
nocom

method td-gks
""".split('\n')
    opts = bohr.build_options(std_input)
    match request.param:
        case 'gks-so':
            opts.so_scf = True
        case 'rks-sr':
            opts.method = 'rks'
            opts.charge = 0.
            opts.spin = 0
        case 'rgks-sr':
            opts.method = 'rgks'
        case 'uks-sr':
            opts.method = 'uks'

    mol = bohr.build_mol(opts)
    return mol, opts

def test_zora(init_bohr):
    mol, opts = init_bohr
    try:
        bohr.run(mol,opts)
    except:
        assert False
    finally:
        assert True