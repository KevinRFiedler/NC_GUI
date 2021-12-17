# -*- coding: utf-8 -*-
"""
Testing

@author: Kevin Fiedler
Date created: 1/13/2021
Date last modified: 1/13/2021
"""


from pymatgen.core import Lattice, Structure, Molecule

#structure = Structure.from_file("Al5LiO8.cif")
#print(structure)

structure = Structure.from_file("Si.cif")
a = structure.lattice.a #In real units I think?
b = structure.lattice.b
c = structure.lattice.c
alpha = structure.lattice.alpha
beta = structure.lattice.beta
gamma = structure.lattice.gamma

#Figure out if this is the best way to iterate over the sites?
#print(structure)
#print(structure.as_dict().keys())
df = structure.as_dataframe()
print(df)
#print(structure)
#print(type(structure))

