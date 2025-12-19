# %%
# This file is part of the gmxtop project.
#
# The gmxtop project is based on or includes code from:
#    kimmdy (https://github.com/graeter-group/kimmdy/tree/main)
#    Copyright (C) graeter-group
#    Licensed under the GNU General Public License v3.0 (GPLv3).
# 
# Modifications and additional code:
#    Copyright (C) 2025 graeter-group
#    Licensed under the GNU General Public License v3.0 (GPLv3).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
from pathlib import Path
from copy import deepcopy

from gmxtop.topology.topology import Topology
from gmxtop.parsing import read_top
from gmxtop.parameterizing import Parameterizer
from gmxtop.topology.atomic import Dihedral
# %% [markdown]
# # Manipulation of force field parameters
# In this example, we demonstrate how to manipulate force field parameters in a topology file using gmxtop.
# We will use a topology file for a hexamer of alanine ('hexala.top') parameterized with the 'amber99sb-star-ildnp.ff' force field.

# %% [markdown]
# To get started, we need to read the topology file.
# %%
top_dict = read_top(Path("hexala.top"), ffdir=Path("amber99sb-star-ildnp.ff"))
original_top = Topology(top_dict)

# %% [markdown]
# # 1. Manipulation of nonbonded parameters
# ## 1.1 Manipulation of partial charges
# Example: Change the charge of the CB atom of the first Ala residue and its attached H atoms
# %% [markdown]
# Inital charges
# %%
top = deepcopy(original_top)
print("Atomic charges before modification:")
for atom_id in ["11", "12", "13", "14"]:
    print(top.atoms[atom_id])

# %% [markdown]
# Modify charges
# %%
top.atoms["11"].charge = "-0.3"  # CB atom of the first ala residue
top.atoms["12"].charge = "0.1"  # HB1 atom of the first ala residue
top.atoms["13"].charge = "0.1"  # HB2 atom
top.atoms["14"].charge = "0.1"  # HB3 atom

print("\nAtomic charges after modification:")
for atom_id in ["11", "12", "13", "14"]:
    print(top.atoms[atom_id])

# %% [markdown]
# Finally, we can write the modified topology to a new file, called "hexala_mod_charges.top". 
# %% 
out_path = Path("hexala_mod_charges.top")
top.to_path(out_path)
# %% [markdown]
# ## 1.2. Manipulation of Lennard-Jones parameters
# Example: Change the Lennard-Jones parameters of the CT atom type. 
# %% [markdown]
# Inital LJ parameters:
# %%
top = deepcopy(original_top)
print(top.ff.atomtypes["CT"])
# %% [markdown]
# Modify LJ parameters:
# %%
top.ff.atomtypes["CT"].sigma = "0.35"  # New sigma value in nm
top.ff.atomtypes["CT"].epsilon = "0.8"  # New epsilon value in kJ mol-1

print(top.ff.atomtypes["CT"])
# %% [markdown]
# Now, let's write the modified topology to a new file, called "hexala_mod_lj.top".
# %%
out_path = Path("hexala_mod_lj.top")
top.to_path(out_path)
# %% [markdown]
# # 2. Manipulation of bonded parameters
# ## 2.1. Local parameter modification
# Simillar to the partial charges, the force field parameters can be modified of a single bond angle or dihedral. <br>
# Example: Change the bond length and force constant of the bond between the CB atom (= atom 11) and the CA (= atom 9) atom of the first Ala residue.

# %% [markdown]
# Initial bond parameters
# %%
top = deepcopy(original_top)
print(
    top.bonds[("9", "11")]
)  # Bond between the CA atom (= atom 9) and the CB atom (= atom 11) of the first Ala residue
# %% [markdown]
# As the equilibrium bond length (c0) [nm] and the force constant (c1) [kJ mol-1 nm-2] are `None`, the bonded parameters for this CA-CB bond are taken from the bond type in the force field. The CA-CB bond parameters in the 'amber99sb-star-ildnp.ff' force field are:
# %%
print(top.ff.bondtypes[("CA", "CB")])
# %% [markdown]
# Modified bond parameters
# %% 
top.bonds[("9", "11")].c0 = "0.15"  # New bond length in nm
top.bonds[("9", "11")].c1 = "500000.0"  # New force constant in kJ mol-1 nm-2

print(top.bonds[("9", "11")])
# %% [markdown]
# Finally, we can write the modified topology to a new file, called "hexala_mod_single_bond.top".
# %%
out_path = Path("hexala_mod_single_bond.top")
top.to_path(out_path)
# %%
# ## 2.2. Global parameter modification
# We can also globally modify force field parameters by overriding the parameters of a specific parameter type in the force field. <br>
# Example: Change the bond length and force constant of all CA-CB bonds
# %% [markdown]
# Initial CA-CB bond parameters from the force field:
# %%
top = deepcopy(original_top)
print(top.ff.bondtypes[("CA", "CB")])

# %% [markdown]
# Modified CA-CB bond parameters in the force field:
# %%
top.ff.bondtypes[("CA", "CB")].c0 = "0.15000"  # New bond length in nm
top.ff.bondtypes[("CA", "CB")].c1 = "500000.0"  # New force constant in kJ mol-1 nm-2

print(top.ff.bondtypes[("CA", "CB")])
# %% [markdown]
# Finally, we can write the modified topology to a new file, called "hexala_mod_all_bond.top".
# %%
out_path = Path("hexala_mod_all_bond.top")
top.to_path(out_path) 
# %% [markdown]
# ## 3. Using a custom Parameterizer
# For more complex modifications of the topology, one can implement a custom `Parameterizer` class. <br>
# Example: Change the N-CA-CB-HBX proper dihedral parameters only for Ala residues
# %% [markdown]
# First the custom `Parameterizer` class, called here `MyParameterizer`, is implemented by inheriting from the base `Parameterizer` class and overriding the `parameterize_topology` method.
# %%
class MyParameterizer(Parameterizer):
    def parameterize_topology(
        self, current_topology: Topology, focus_nrs: set[str] | None = None
    ) -> Topology:
        # iterate over all proper dihedrals
        for proper in current_topology.proper_dihedrals.values():
            # Only modify proper dihedrals in Ala residues
            if current_topology.atoms[proper.ai].residue == "ALA":
                # Get the atom types of the four atoms in the proper dihedral
                atom_names = (
                    current_topology.atoms[proper.ai].atom,
                    current_topology.atoms[proper.aj].atom,
                    current_topology.atoms[proper.ak].atom,
                    current_topology.atoms[proper.al].atom,
                )
                if atom_names in [
                    ("N", "CA", "CB", "HB1"),
                    ("N", "CA", "CB", "HB2"),
                    ("N", "CA", "CB", "HB3"),
                ]:
                    # Change the parameters of the N-CA-CB-HBX proper dihedral
                    proper.dihedrals[""] = Dihedral(
                        proper.ai,
                        proper.aj,
                        proper.ak,
                        proper.al,
                        funct="9",
                        c0="180.0",
                        c1="2.0",
                    )
        return current_topology

# %% [markdown]
# Now, we can apply the custom parameterizer to the topology.
# %%
top = deepcopy(original_top)
# Assign the new parameterizer to the topology
top.parametrizer = MyParameterizer() 
# Please note that the parameters are only updated when needs_parameterization is set to True
top.needs_parameterization = True
# %%
# Finally, we can update the parameters ... <br>
# a) ... directly in the topology instance by calling `update_parameters()` explicitly
# %%
t = deepcopy(top)
t.update_parameters()
# %% [markdown]
# b) ... by writing the topology to dict
# %%
t = deepcopy(top)
top_dict = t.to_dict()
# %% [markdown]
# c) ... by writing the topology to a file
# %%
t = deepcopy(top)
t.to_path("hexala_mod_proper.top")
# %% [markdown]
# When executing `to_dict()` or `to_path()`, the parameters are automatically updated by calling internally `update_parameters()` if `needs_parameterization` is True.
# %%
