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

import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from gmxtop.topology.atomic import *
from gmxtop.utils import get_gmx_dir
from gmxtop.parsing import TopologyDict, read_top
from gmxtop.topology.utils import (
    get_residue_by_bonding,
    match_atomic_item_to_atomic_type,
    get_protein_section,
    get_selected_section,
)
from gmxtop.topology.topology import Topology


@pytest.fixture(scope="module")
def filedir() -> Path:
    dirname = "test_topology"
    try:
        file_dir = Path(__file__).parent / "test_files" / dirname
    except NameError:
        file_dir = Path("./tests/test_files") / dirname
    return file_dir


@pytest.fixture(scope="module")
def assetsdir() -> Path:
    return Path(__file__).parent / "test_files" / "assets"


@pytest.fixture()
def raw_hexala_top_fix(filedir) -> TopologyDict:
    return read_top(filedir / "hexala.top")


@pytest.fixture()
def raw_top_a_fix(filedir) -> TopologyDict:
    return read_top(filedir / "topol_stateA.top")


@pytest.fixture()
def raw_top_b_fix(filedir) -> TopologyDict:
    return read_top(filedir / "topol_stateB.top")


@pytest.fixture()
def raw_urea_top_fix(filedir) -> TopologyDict:
    return read_top(filedir / "urea.top")


@pytest.fixture()
def raw_urea_times_two_top_fix(filedir) -> TopologyDict:
    return read_top(filedir / "urea-times-2.top")


@pytest.fixture()
def raw_two_different_ureas_top_fix(filedir) -> TopologyDict:
    return read_top(filedir / "two-different-ureas.top")


@pytest.fixture()
def hexala_rawtop_fix(assetsdir, filedir) -> TopologyDict:
    return read_top(filedir / "hexala.top")


@pytest.fixture()
def hexala_top_fix(assetsdir, filedir) -> Topology:
    hexala_top = read_top(filedir / "hexala.top")
    return Topology(hexala_top)


@st.composite
def random_atomlist(draw):
    n = draw(st.integers(min_value=3, max_value=5))
    elements = st.integers(min_value=1, max_value=n)
    bound_to = st.lists(elements, min_size=2, max_size=4, unique=True)
    atom_nrs = [str(x + 1) for x in range(n)]
    atoms = []
    allowed_text = st.text("COHT1" + "*+", min_size=1, max_size=5)
    for i in atom_nrs:
        type = draw(allowed_text)
        resnr = draw(allowed_text)
        residue = draw(allowed_text)
        a = draw(allowed_text)
        cgnr = draw(allowed_text)
        charge = draw(allowed_text)
        mass = draw(allowed_text)
        atom = Atom(str(i), type, resnr, residue, a, cgnr, charge, mass)
        atom.bound_to_nrs = [str(x) for x in draw(bound_to) if str(x) != i]
        atoms.append(atom)

    return atoms


class TestGMX:
    def test_gmx_dir_is_found(self):
        gmx = get_gmx_dir()
        assert gmx
        assert Path(gmx).is_dir()


class TestMatch:
    @pytest.fixture
    def top_fix(self, filedir) -> Topology:
        hexala_top = read_top(filedir / "hexala.top")
        return Topology(hexala_top)

    def test_match_atomic_item_to_atomic_type(self, top_fix):
        types = top_fix.ff.angletypes

        atomic_id = ["CT", "C_R", "N"]
        want = ("CT", "C", "N")
        types_wanted: dict[AngleId, AngleType] = {want: types[want]}
        item_type = match_atomic_item_to_atomic_type(atomic_id, types_wanted)
        expected = AngleType(
            i="CT",
            j="C",
            k="N",
            id="CT---C---N",
            id_sym="N---C---CT",
            funct="1",
            c0="116.600",
            c1="585.760",
            c2=None,
            c3=None,
        )
        assert item_type == expected

        atomic_id = ["C_R", "CA", "HA"]
        want = ("C", "CA", "HA")
        types_wanted = {want: types[want]}
        item_type = match_atomic_item_to_atomic_type(atomic_id, types_wanted)
        expected = AngleType(
            i="C",
            j="CA",
            k="HA",
            id="C---CA---HA",
            id_sym="HA---CA---C",
            funct="1",
            c0="120.000",
            c1="418.400",
            c2=None,
            c3=None,
        )
        assert item_type == expected


class TestUrea:

    def test_urea(self, raw_urea_top_fix):
        raw = deepcopy(raw_urea_top_fix)
        top = Topology(raw)
        assert len(top.atoms) == 8
        assert len(top.bonds) == 7
        assert len(top.pairs) == 0
        assert len(top.angles) == 0
        assert len(top.proper_dihedrals) == 8
        assert len(top.improper_dihedrals) == 3
        assert top.moleculetypes[top.selected_moleculetype].nrexcl == "3"

    def test_making_molecules_explicit(self, raw_urea_times_two_top_fix):
        raw = deepcopy(raw_urea_times_two_top_fix)
        top = Topology(raw)
        assert len(top.atoms) == 16
        assert len(top.bonds) == 14
        assert len(top.pairs) == 0
        assert len(top.angles) == 0
        assert len(top.proper_dihedrals) == 16
        assert len(top.improper_dihedrals) == 6
        assert top.moleculetypes[top.selected_moleculetype].atoms == top.atoms
        assert top.selected_moleculetype == "Urea(2x)"
        for i in range(8):
            n1 = str(i + 1)
            n2 = str(i + 9)
            a1 = top.atoms[n1]
            a2 = top.atoms[n2]
            assert a1.atom == a2.atom
            assert a1.type == a2.type
            assert a1.mass == a2.mass
            assert int(a2.nr) == int(a1.nr) + 8
            assert int(a2.cgnr) == int(a1.cgnr) + 8

    def test_merging_molecules(self, raw_two_different_ureas_top_fix):
        raw = deepcopy(raw_two_different_ureas_top_fix)
        top = Topology(raw)
        assert len(top.atoms) == 16
        assert len(top.bonds) == 14
        assert len(top.pairs) == 0
        assert len(top.angles) == 0
        assert len(top.proper_dihedrals) == 16
        assert len(top.improper_dihedrals) == 6
        assert top.moleculetypes[top.selected_moleculetype].atoms == top.atoms
        assert top.selected_moleculetype == "Urea1(1x)-Urea2(1x)"
        for i in range(8):
            n1 = str(i + 1)
            n2 = str(i + 9)
            a1 = top.atoms[n1]
            a2 = top.atoms[n2]
            assert a1.atom == a2.atom
            assert a1.type == a2.type
            assert a1.mass == a2.mass
            if i == 7:
                assert a1.residue == "TOTALLYNOTUREA"
                assert a2.residue == "URE"
            else:
                assert a1.residue == "URE"
                assert a2.residue == "URE"
            assert int(a2.nr) == int(a1.nr) + 8
            assert int(a2.cgnr) == int(a1.cgnr) + 8
            assert a1.bound_to_nrs == [str(int(x) - 8) for x in a2.bound_to_nrs]
            if i < 7:
                assert int(a1.resnr) == int(a2.resnr) - 2
            else:
                assert int(a1.resnr) == int(a2.resnr) - 1

    def test_merging_solvent(self, hexala_rawtop_fix):
        top = Topology(
            deepcopy(hexala_rawtop_fix),
            is_selected_moleculetype_f=lambda mol: mol in ["Protein", "SOL"],
        )
        assert len(top.atoms) == 38013
        assert top.atoms["73"].resnr == "9"
        assert top.atoms["74"].resnr == "9"
        assert top.atoms["75"].resnr == "9"
        assert top.atoms["76"].resnr == "10"
        assert top.atoms["77"].resnr == "10"
        assert top.atoms["78"].resnr == "10"
        assert top.atoms["79"].resnr == "11"
        assert top.atoms["80"].resnr == "11"
        assert top.atoms["81"].resnr == "11"

    def test_to_path(self, raw_urea_top_fix):
        top = Topology(deepcopy(raw_urea_top_fix))
        out_path = Path("urea_out.top")
        top.to_path(out_path)


class TestTopAB:
    def test_top_ab(self, raw_top_a_fix, raw_top_b_fix):
        topA = Topology(raw_top_a_fix)
        topB = Topology(raw_top_b_fix)
        assert topA
        assert topB
        assert len(topA.atoms) == 41
        assert len(topA.proper_dihedrals) == 84
        proper_dihedrals_counts = 0
        for dihedral in topA.proper_dihedrals.values():
            proper_dihedrals_counts += len(dihedral.dihedrals)
        assert proper_dihedrals_counts == 88


class TestTopology:
    def test_reindex_no_change(self, hexala_top_fix: Topology):
        org_top: Topology = deepcopy(hexala_top_fix)
        update = hexala_top_fix.reindex_atomnrs()

        # test produced mapping
        assert len(update.keys()) == 12
        for mapping in update.values():
            for k, v in mapping.items():
                assert k == v

        # test topology
        assert org_top.atoms == hexala_top_fix.atoms
        assert org_top.bonds == hexala_top_fix.bonds
        assert org_top.angles == hexala_top_fix.angles
        assert org_top.proper_dihedrals == hexala_top_fix.proper_dihedrals
        assert org_top.improper_dihedrals == hexala_top_fix.improper_dihedrals

    def test_generate_topology_from_bound_to(self, hexala_top_fix):
        og_top = deepcopy(hexala_top_fix)
        newtop = deepcopy(hexala_top_fix)
        newtop.bonds.clear()
        newtop.pairs.clear()
        newtop.angles.clear()
        newtop.proper_dihedrals.clear()

        assert newtop.bonds == {}
        assert newtop.moleculetypes[newtop.selected_moleculetype].bonds == {}

        newtop._regenerate_topology_from_bound_to()

        assert newtop.bonds == og_top.bonds
        assert newtop.pairs == og_top.pairs
        assert newtop.angles == og_top.angles
        assert newtop.proper_dihedrals == og_top.proper_dihedrals


class TestHexalaTopology:

    def test_all_terms_accounted_for(self, raw_hexala_top_fix, hexala_top_fix):
        atoms = get_protein_section(raw_hexala_top_fix, "atoms")
        bonds = get_protein_section(raw_hexala_top_fix, "bonds")
        pairs = get_protein_section(raw_hexala_top_fix, "pairs")
        angles = get_protein_section(raw_hexala_top_fix, "angles")
        dihedrals = get_protein_section(raw_hexala_top_fix, "dihedrals")

        assert atoms
        assert bonds
        assert pairs
        assert angles
        assert dihedrals
        assert len(hexala_top_fix.atoms) == len(atoms)
        assert len(hexala_top_fix.bonds) == len(bonds)
        assert len(hexala_top_fix.pairs) == len(pairs)
        assert len(hexala_top_fix.angles) == len(angles)
        assert len(hexala_top_fix.proper_dihedrals) + len(
            hexala_top_fix.improper_dihedrals
        ) == len(dihedrals)

    def test_find_bondtypes(self, hexala_top_fix):
        top_cp = deepcopy(hexala_top_fix)

        id = ["C", "CT"]
        result = match_atomic_item_to_atomic_type(id, top_cp.ff.bondtypes)
        id = ["CT", "C"]
        result = match_atomic_item_to_atomic_type(id, top_cp.ff.bondtypes)
        assert result is not None

    def test_top_update_dict(self, raw_hexala_top_fix):
        raw = raw_hexala_top_fix
        raw_copy = deepcopy(raw)
        top = Topology(raw_copy)
        top._update_dict()
        assert top.selected_moleculetype == "Protein(1x)"
        assert top.top["dihedraltypes"]["content"] == raw["dihedraltypes"]["content"]
        assert (
            top.top[f"moleculetype_{top.selected_moleculetype}"]["subsections"][
                "dihedrals"
            ]["content"]
            == raw["moleculetype_Protein"]["subsections"]["dihedrals"]["content"]
        )
        # "fix" the expected differences to test the rest
        assert raw["molecules"]["content"][0] == ["Protein", "1"]
        assert top.top["molecules"]["content"][0] == [top.selected_moleculetype, "1"]
        raw["molecules"]["content"][0] = [top.selected_moleculetype, "1"]

        assert raw["moleculetype_Protein"]["content"][0] == ["Protein", "3"]
        assert top.top[f"moleculetype_{top.selected_moleculetype}"]["content"][0] == [
            top.selected_moleculetype,
            "3",
        ]
        for section in ["atoms", "bonds", "angles", "dihedrals", "pairs"]:
            assert (
                top.top[f"moleculetype_{top.selected_moleculetype}"]["subsections"][
                    section
                ]["content"]
                == raw["moleculetype_Protein"]["subsections"][section]["content"]
            )

    def test_top_properties(self, hexala_top_fix):
        top = deepcopy(hexala_top_fix)

        # initial correct number of atoms and bonds
        assert len(top.atoms) == 72
        assert len(top.bonds) == 71

        # order is correct
        val = 0
        for atom_id in top.atoms.keys():
            assert int(atom_id) > val
            val = int(atom_id)

        val = 0
        for bond in top.bonds.keys():
            assert int(bond[0]) >= val
            val = int(bond[0])

        # specific atoms
        for atom in top.atoms.values():
            assert atom.type == "CT"
            assert atom.atom == "CH3"
            assert atom.residue == "ACE"
            break

        # everything is bound to something
        for atom in top.atoms.values():
            assert len(atom.bound_to_nrs) > 0

    def test_find_terms_around_atom(self, hexala_top_fix: Topology):
        top = deepcopy(hexala_top_fix)
        protein = top.moleculetypes[top.selected_moleculetype]

        atomnr = "29"
        bonds = protein._get_atom_bonds(atomnr)
        angles = protein._get_atom_angles(atomnr)
        proper_dihedrals = protein._get_atom_proper_dihedrals(atomnr)
        improper_dihedrals = protein._get_atom_improper_dihedrals(atomnr, top.ff)
        assert len(bonds) == 4
        assert len(angles) == 13
        assert len(proper_dihedrals) == 25
        assert len(improper_dihedrals.keys()) == 3

        atomnr = "25"  # C
        bonds = protein._get_atom_bonds(atomnr)
        angles = protein._get_atom_angles(atomnr)
        proper_dihedrals = protein._get_atom_proper_dihedrals(atomnr)
        improper_dihedrals = protein._get_atom_improper_dihedrals(atomnr, top.ff)
        assert len(bonds) == 3
        assert len(angles) == 8
        assert len(proper_dihedrals) == 18
        assert len(improper_dihedrals.keys()) == 3

        atomnr = "9"
        bonds = protein._get_atom_bonds(atomnr)
        angles = protein._get_atom_angles(atomnr)
        proper_dihedrals_center = protein._get_center_atom_dihedrals(atomnr)

        assert bonds == [("7", "9"), ("9", "10"), ("9", "11"), ("9", "15")]
        assert set(proper_dihedrals_center) == set(
            [
                ("5", "7", "9", "10"),
                ("5", "7", "9", "11"),
                ("5", "7", "9", "15"),
                ("8", "7", "9", "10"),
                ("8", "7", "9", "11"),
                ("8", "7", "9", "15"),
                ("7", "9", "11", "12"),
                ("7", "9", "11", "13"),
                ("7", "9", "11", "14"),
                ("10", "9", "11", "12"),
                ("10", "9", "11", "13"),
                ("10", "9", "11", "14"),
                ("15", "9", "11", "12"),
                ("15", "9", "11", "13"),
                ("15", "9", "11", "14"),
                ("7", "9", "15", "16"),
                ("7", "9", "15", "17"),
                ("10", "9", "15", "16"),
                ("10", "9", "15", "17"),
                ("11", "9", "15", "16"),
                ("11", "9", "15", "17"),
            ]
        )

    def test_ff(self, hexala_top_fix):
        top = deepcopy(hexala_top_fix)

        residues = list(top.ff.residuetypes.keys())
        assert len(residues) == 153

        res = ResidueType(
            "HOH",
            atoms={
                "OW": ResidueAtomSpec("OW", "OW", "-0.834", "0"),
                "HW1": ResidueAtomSpec("HW1", "HW", "0.417", "0"),
                "HW2": ResidueAtomSpec("HW2", "HW", "0.417", "0"),
            },
            bonds={
                ("OW", "HW1"): ResidueBondSpec("OW", "HW1"),
                ("OW", "HW2"): ResidueBondSpec("OW", "HW2"),
            },
            proper_dihedrals={},
            improper_dihedrals={},
        )

        assert top.ff.residuetypes["HOH"] == res

    def test_get_residue_by_bonding(self, hexala_top_fix):
        top = hexala_top_fix
        a = top.atoms["1"]
        res = get_residue_by_bonding(a, top.atoms)
        assert len(res) == 6
        for a in res.values():
            assert a.residue == "ACE"
        a = top.atoms["10"]
        res = get_residue_by_bonding(a, top.atoms)
        assert len(res) == 10
        for a in res.values():
            assert a.residue == "ALA"
        a = top.atoms["25"]
        res = get_residue_by_bonding(a, top.atoms)
        for a in res.values():
            assert a.residue == "ALA"


class TestPolymerFF:
    def test_sections_are_complete(self, filedir):
        path = filedir / "polymer/topol.top"
        raw_top = read_top(path)
        top = Topology(raw_top)
        assert len(top.ff.nonbond_params) == 6
        assert top.ff.nonbond_params[("B0", "B0")] == NonbondParamType(
            i="B0",
            j="B0",
            funct="1",
            c0="2.5",
            c1="2.5",
            id="B0---B0",
            id_sym="B0---B0",
        )
        assert top.ff.atomtypes["B1"] == AtomType(
            type="B1",
            id="B1",
            id_sym="B1",
            at_num="",
            mass="20000.0",
            charge="0.000",
            ptype="A",
            sigma="0.0",
            epsilon="0.0",
        )

    def test_nrexcl_from_the_correct_moleculetype(self, filedir):
        path = filedir / "polymer/topol.top"
        raw_top = read_top(path)
        top = Topology(raw_top)
        assert top.moleculetypes[top.selected_moleculetype].nrexcl == "1"


class TestRadicalAla:
    @pytest.fixture
    def top_noprm_fix(self, filedir) -> Topology:
        hexala_top = read_top(filedir / "Ala_R_noprm.top")
        return Topology(hexala_top)

    @pytest.fixture
    def top_noprm_explicitR_fix(self, filedir) -> Topology:
        hexala_top = read_top(filedir / "Ala_R_noprm.top")
        return Topology(hexala_top, radicals="9")

    def test_is_radical(self, top_noprm_fix):
        assert top_noprm_fix.atoms["9"].is_radical == True
        assert top_noprm_fix.atoms["10"].is_radical == False

    def test_is_radical_explicit(self, top_noprm_explicitR_fix):
        assert top_noprm_explicitR_fix.atoms["9"].is_radical == True
        assert top_noprm_explicitR_fix.atoms["10"].is_radical == False


class TestDimerization:

    @pytest.fixture
    def top_target(self, filedir) -> Topology:
        return Topology.from_path(
            filedir / "TdT_from_pdb2gmx.top", ffdir=filedir / "amber14sb_OL21_mod.ff"
        )

    def test_get_improper_dihedrals_dimer(self, top_target: Topology):
        """

        [ atoms ]
        12 CT 1 DD5 C6 12 0.01612 12.01

        14 CT 1 DD5 C5 14 -0.09999 12.01

        44 CT 2 DD3 C6 44 0.01612 12.01

        46 CT 2 DD3 C5 46 -0.09999 12.01

        from pdb2gmx
        [dihedrals ]
        9 11 12 23 4
        11 21 23 24 4
        14 21 19 20 4
        19 23 21 22 4
        41 43 44 55 4
        43 53 55 56 4
        46 53 51 52 4
        51 55 53 54 4

        from [dd5 ]
         [ impropers ]
            C2    C6    N1   C1'
            N1    N3    C2    O2
            C5    N3    C4    O4
            C4    C2    N3    H3

        from [dd3]
         [ impropers ]
            C2    C6    N1   C1'
            N1    N3    C2    O2
            C5    N3    C4    O4
            C4    C2    N3    H3
        """
        atom_impropers = top_target.selected_molecule._get_atom_improper_dihedrals(
            "12", top_target.ff
        )
        assert list(atom_impropers.keys()) == [("9", "11", "12", "23")]

        atom_impropers = top_target.selected_molecule._get_atom_improper_dihedrals(
            "14", top_target.ff
        )
        assert list(atom_impropers.keys()) == [("14", "21", "19", "20")]

        atom_impropers = top_target.selected_molecule._get_atom_improper_dihedrals(
            "44", top_target.ff
        )
        assert list(atom_impropers.keys()) == [("41", "43", "44", "55")]

        atom_impropers = top_target.selected_molecule._get_atom_improper_dihedrals(
            "46", top_target.ff
        )
        assert list(atom_impropers.keys()) == [("46", "53", "51", "52")]


class TestBAZ:
    def test_no_warning_on_loading(self, filedir, caplog):
        caplog.set_level(logging.WARNING)
        Topology.from_path(filedir / "baz.top", ffdir=filedir / "amber14sb.ff")
        assert (
            not caplog.records
        ), f"Logger.warning was raised during loading baz.top: {[r.message for r in caplog.records]}"
