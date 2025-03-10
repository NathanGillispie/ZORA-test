import pytest

from sys import path
path.append('/'.join(__file__.split('/')[:-2])+'/')

import bohr

@pytest.fixture(params=['rt-td-rks'])
def init_bohr(request):
    std_input="""
start molecule
  O      0.00000000     0.00000000     0.11726921
  H      0.75698224     0.00000000    -0.46907685
  H     -0.75698224     0.00000000    -0.46907685
end molecule

charge 0
spin 0
basis sto-3g 
diis true
grid_level 5
e_convergence 1e-6
d_convergence 1e-6
maxiter 200
nroots 50
xc hf
nocom

method rt-td-rks
""".split('\n')
    opts = bohr.build_options(std_input)
    opts.method = request.param

    #Change options for each method here

    mol = bohr.build_mol(opts)
    return mol, opts


def test_rt_td_rks():
    mol, opts = init_bohr
    try:
        bohr.run(mol,opts)
    except:
        assert False
    finally:
        assert True