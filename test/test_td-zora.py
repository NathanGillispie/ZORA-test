import pytest

from sys import path
path.append('/'.join(__file__.split('/')[:-2])+'/')

import bohr

@pytest.fixture(params=['td-gks-so','td-gks-sr','td-rks-sr','td-rgks-sr','td-uks-sr'])
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
        case 'td-gks-so':
            opts.so_scf = True
        case 'td-rks-sr':
            opts.method = 'td-rks'
            opts.charge = 0.
            opts.spin = 0
        case 'td-rgks-sr':
            opts.method = 'td-rgks'
        case 'td-uks-sr':
            opts.method = 'td-uks'

    mol = bohr.build_mol(opts)
    return mol, opts


def test_td_zora(init_bohr):
    mol, opts = init_bohr
    try:
        bohr.run(mol,opts)
    except:
        assert False
    finally:
        assert True