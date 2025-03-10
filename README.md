# Bohr-dev
Density Functional Theory for core-level excitations. Supports time-dependent (TD) and real-time (RT) methods. Includes Hartree-Fock and Kohn-Sham methods for restricted and unrestricted orbitals.

Intended use: cross-referencing calculations across different languages and suites. However, Bohr can be used as a standalone application.

- Initialize environment with `conda install --file requirements.txt`
- Run a calculation with `./bohr.py moleule_basis_method.in`

The `.in` file can contain atoms and options, but must contain atoms. The `options.py` file contains the default options which may be overwritten by the input file.

See [template.in](/template.in) for input formatting.

## Bohr module
Bohr can be ran as a standalone script `./bohr.py input.in`. If you use a virtual environment you might like to change the shebang in `bohr.py`. Here, bohr can be ran as a module. This is useful for testing.

## Supported methods

| Method | Name | Basic | TD ("td-"+name) | RT ("rt-\[td-\]"+name) | Notes |
| --- | --- | --- | --- | --- | --- |
| Restricted Hartree-Fock (HF) | "rhf" | ✅ | ✅ | | 
| Unrestricted HF | "uhf" | ✅ | ✅ | |
| General HF | "ghf" | | ✅ | | |
| Restricted Kohn-Sham (KS) | "rks" | ✅ | ✅ | ✅ | |
| Unrestricted KS | "uks" | ✅ | ✅ | | Includes pyscf version via "uks-pyscf" |
| Restricted Open-shell KS | "roks" | ✅ | | | |
| GKS | "gks" | ✅ | ✅ | ✅ | |
| Restricted GKS | "rgks" | ✅ | ✅ | | |
| Unitary Coupled-Cluster Doubles | "uccd" | ✅ | | | |

List of methods: "rhf", "td-rhf", "uhf", "td-uhf", "td-ghf", "rks", "td-rks", "rt-td-rks", "uks", "td-uks", "uks-pyscf", "roks", "gks", "td-gks", "rt-td-gks", "rgks", "td-rgks", "uccd"

## Tests

Run `pytest` in the root directory.
