import pytest

from sys import path
path.append('/'.join(__file__.split('/')[:-2])+'/')

import bohr

@pytest.fixture(params=['gks','rks','roks','rgks','uks'])
def init_bohr(request):
    std_input="""
start molecule
  O      0.00000000     0.00000000     0.11726921
  H      0.75698224     0.00000000    -0.46907685
  H     -0.75698224     0.00000000    -0.46907685
end molecule

charge 0
spin 0
basis 6-31g
diis true
grid_level 5
e_convergence 1e-6
d_convergence 1e-6
maxiter 200
nroots 50
xc pbe0 
method gks
""".split('\n')
    opts = bohr.build_options(std_input)
    
    opts.method = request.param
    match request.param:
        case 'roks':
            opts.charge=1.
            opts.spin=1
            opts.xc='hf'
        case 'rgks' | 'uks':
            opts.charge=1.
            opts.spin=1
    mol = bohr.build_mol(opts)
    
    return mol, opts

def test_ks(init_bohr):
    mol, opts = init_bohr
    try:
        bohr.run(mol,opts)
    except:
        assert False
    finally:
        assert True

