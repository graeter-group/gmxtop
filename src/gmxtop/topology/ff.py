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

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Optional

from gmxtop.constants import FFFUNC, IMPROPER_DIHEDRAL_FFFUNCS
from gmxtop.parsing import read_top
from gmxtop.topology.atomic import (
    AngleId,
    AngleType,
    AtomId,
    AtomType,
    BondId,
    BondType,
    DihedralType,
    ImproperDihedralId,
    NonbondParamType,
    ProperDihedralId,
    ResidueType,
)
from gmxtop.topology.utils import get_top_section
from gmxtop.utils import get_gmx_dir

logger = logging.getLogger(__name__)


class FF:
    """Container for parsed forcefield data.

    Also see <https://manual.gromacs.org/current/reference-manual/topologies/topology-file-formats.html#topology-file>
    """

    def __init__(
        self,
        top: dict,
        residuetypes_path: Optional[Path] = None,
        gromacs_alias: str = "gmx",
    ):
        self.atomtypes: dict[AtomId, AtomType] = {}
        self.bondtypes: dict[BondId, BondType] = {}
        self.angletypes: dict[AngleId, AngleType] = {}
        self.proper_dihedraltypes: dict[ProperDihedralId, DihedralType] = {}
        self.improper_dihedraltypes: dict[ImproperDihedralId, DihedralType] = {}
        self.residuetypes: dict[str, ResidueType] = {}
        self.nonbond_params: dict[BondId, NonbondParamType] = {}
        # function type of improper dihedrals generated from residuetypes,
        # read from the improper dihedraltypes or else from
        # [ bondedtypes ] of the residuetypes files
        self.improper_funct: str = FFFUNC["mult_improper_dihedral"]

        ffdir: Optional[Path] = top["ffdir"]

        self.gmxdir = get_gmx_dir(gromacs_alias)
        if self.gmxdir is None:
            logger.warning(f"Could not find gromacs data directory for {gromacs_alias}")

        if defaults := get_top_section(top, "defaults"):
            self.defaults = defaults

        atomtypes = get_top_section(top, "atomtypes")
        if atomtypes is not None:
            for l in atomtypes:
                atomtype = AtomType.from_top_line(l)
                self.atomtypes[atomtype.type] = atomtype
        bondtypes = get_top_section(top, "bondtypes")
        if bondtypes is not None:
            for l in bondtypes:
                bondtype = BondType.from_top_line(l)
                self.bondtypes[(bondtype.i, bondtype.j)] = bondtype

        nonbond_params = get_top_section(top, "nonbond_params")
        if nonbond_params is not None:
            for l in nonbond_params:
                nonbond_param = NonbondParamType.from_top_line(l)
                self.nonbond_params[(nonbond_param.i, nonbond_param.j)] = nonbond_param

        angletypes = get_top_section(top, "angletypes")
        if angletypes is not None:
            for l in angletypes:
                angletype = AngleType.from_top_line(l)
                self.angletypes[(angletype.i, angletype.j, angletype.k)] = angletype

        dihedraltypes = get_top_section(top, "dihedraltypes")
        if dihedraltypes is not None:
            for l in dihedraltypes:
                dihedraltype = DihedralType.from_top_line(l)
                # proper dihedrals can be defined multiple times
                # with a different phase
                if dihedraltype.funct in IMPROPER_DIHEDRAL_FFFUNCS:
                    self.improper_dihedraltypes[
                        (dihedraltype.i, dihedraltype.j, dihedraltype.k, dihedraltype.l)
                    ] = dihedraltype
                else:
                    # e.g. proper dihedrals with dihedraltype.funct == "9":
                    if (
                        self.proper_dihedraltypes.get(
                            (
                                dihedraltype.i,
                                dihedraltype.j,
                                dihedraltype.k,
                                dihedraltype.l,
                                dihedraltype.periodicity,
                            )
                        )
                        is None
                    ):
                        self.proper_dihedraltypes[
                            (
                                dihedraltype.i,
                                dihedraltype.j,
                                dihedraltype.k,
                                dihedraltype.l,
                                dihedraltype.periodicity,
                            )
                        ] = dihedraltype

        # grompp takes the improper parameters from the dihedraltypes, so their
        # funct takes precedence over [ bondedtypes ] of the residuetypes files,
        # which may belong to a different forcefield, e.g. for a topology without
        # #include the ffdir is guessed from any *.ff directory next to it.
        dihedraltype_improper_functs = {
            t.funct for t in self.improper_dihedraltypes.values()
        }
        if len(dihedraltype_improper_functs) == 1:
            self.improper_funct = next(iter(dihedraltype_improper_functs))

        if residuetypes_path:
            logger.debug(f"Using specified residuetypes file: {residuetypes_path}")
            residuetypes_paths = [residuetypes_path]
        else:
            logger.debug("Trying to use default amber protein residuetypes file.")
            if ffdir is None:
                logger.warning("ffdir is None. No residuetypes will be parsed.")
                return
            residuetypes_paths = [
                ffdir / x for x in ["rna.rtp", "dna.rtp", "aminoacids.rtp"]
            ]

        improper_functs = []
        for residuetypes_path in residuetypes_paths:
            if not residuetypes_path.exists():
                logger.warning(f"{residuetypes_path} not found in ffdir.")
                continue
            residuetypes_dict = read_top(residuetypes_path, use_gmx_dir=False)
            if bondedtypes := get_top_section(residuetypes_dict, "bondedtypes"):
                # ; bonds angles dihedrals impropers ...
                improper_functs.append(bondedtypes[0][3])
            for k, v in residuetypes_dict.items():
                if k.startswith("BLOCK") or k in ["bondedtypes", "ffdir", "define"]:
                    continue
                if not v.get("subsections"):
                    raise AssertionError(f"key {k} has no subsections, only {v}.")
                self.residuetypes[k] = ResidueType.from_section(k, v["subsections"])

        if improper_functs:
            if len(dihedraltype_improper_functs) != 1:
                self.improper_funct = improper_functs[0]
            if set(improper_functs) != {self.improper_funct}:
                logger.warning(
                    f"Improper dihedral function types {improper_functs} in "
                    f"[ bondedtypes ] of {[str(p) for p in residuetypes_paths]} "
                    f"differ from the one used ({self.improper_funct}). "
                    "The residuetypes may not belong to the forcefield of the topology."
                )

    def __str__(self) -> str:
        return textwrap.dedent(
            f"""
        ForceField parameters with
        {len(self.atomtypes)} atomtypes,
        {len(self.bondtypes)} bondtypes,
        {len(self.angletypes)} angletypes,
        {len(self.proper_dihedraltypes)} proper dihedraltypes
        {len(self.improper_dihedraltypes)} improper dihedraltypes
        {len(self.residuetypes)} residuetypes
        """
        )

    def __repr__(self) -> str:
        return f"FF({self.__dict__})"

    def _repr_pretty_(self, p, _):
        """A __repr__ for ipython.

        This whill be used if just the name of the object is entered in the ipython shell
        or a jupyter notebook.

        p is an instance of [IPython.lib.pretty.RepresentationPrinter](https://ipython.org/ipython-doc/3/api/generated/IPython.lib.pretty.html#IPython.lib.pretty.PrettyPrinter)
        """
        p.text(str(self))
