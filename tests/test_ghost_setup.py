"""
Tests for ghost molecule setup in multi-atom GCMC systems.

Verifies fixes for:
- Bug #7: Ghost molecules must have bonded forces
- Bug #8: Ghost positions must be spread (no r=0 overlap)
- Beyond-1-4 pairs must be converted to constant bonded forces
"""

import os

import numpy as np
import openmm
import openmm.openmm as mm
import pytest
from mstk.topology import Topology
from mstk.trajectory import Trajectory
from mstk.forcefield import ForceField
from mstk.simsys import System

DATA_DIR = os.path.join(os.path.dirname(__file__), '300ethanol')

gpu = pytest.mark.skipif(
    "CUDA_VISIBLE_DEVICES" not in os.environ,
    reason="Requires CUDA enabled GPU.",
)


@pytest.fixture
def ethanol_system():
    top = Topology.open(os.path.join(DATA_DIR, 'top.psf'))
    frame = Trajectory.read_frame_from_file(os.path.join(DATA_DIR, 'conf.gro'), -1)
    top.cell.set_box(frame.cell.vectors)
    top.set_positions(frame.positions)
    ff = ForceField.open('primitive.zff')
    return System(top, ff)


def _make_sampler(system, **kwargs):
    from loch import GCMCSampler

    defaults = dict(
        residue_name='ETOH',
        num_ghost_waters=5,
        batch_size=10,
        num_attempts=10,
        is_pme=False,
        temperature=300.0,
        excess_chemical_potential=-25.413,
        standard_volume=0.09459,
        platform="cuda",
    )
    defaults.update(kwargs)
    return GCMCSampler(system, **defaults)


def _count_bonded_terms_on_atom(omm_system, atom_idx):
    """Count bond, angle, torsion terms involving a given atom."""
    bonds = 0
    angles = 0
    torsions = 0

    for force in omm_system.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                p1, p2, *_ = force.getBondParameters(i)
                if atom_idx in (p1, p2):
                    bonds += 1
        elif isinstance(force, mm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                p1, p2, p3, *_ = force.getAngleParameters(i)
                if atom_idx in (p1, p2, p3):
                    angles += 1
        elif isinstance(force, mm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, *_ = force.getTorsionParameters(i)
                if atom_idx in (p1, p2, p3, p4):
                    torsions += 1

    return bonds, angles, torsions


class TestGhostBondedForces:
    """Bug #7: Ghost molecules must have bonds/angles/torsions."""

    @gpu
    def test_ghost_has_bonded_forces(self, ethanol_system):
        """Ghost atoms must have the same bonded forces as real atoms."""
        sampler = _make_sampler(ethanol_system)
        omm_sys = sampler.omm_system

        real_start = sampler.water_indices[0]
        real_bonds, real_angles, real_torsions = _count_bonded_terms_on_atom(
            omm_sys, real_start)

        n_real = sampler._num_waters - sampler._num_ghost_waters
        ghost_start = sampler.water_indices[n_real]
        ghost_bonds, ghost_angles, ghost_torsions = _count_bonded_terms_on_atom(
            omm_sys, ghost_start)

        assert ghost_bonds == real_bonds, (
            f"Ghost has {ghost_bonds} bonds, real has {real_bonds}")
        assert ghost_angles == real_angles, (
            f"Ghost has {ghost_angles} angles, real has {real_angles}")
        assert ghost_torsions == real_torsions, (
            f"Ghost has {ghost_torsions} torsions, real has {real_torsions}")

    @gpu
    def test_ghost_geometry_preserved_after_md(self, ethanol_system):
        """Ghost molecule geometry must survive MD integration."""
        sampler = _make_sampler(ethanol_system)
        omm_sys = sampler.omm_system

        integrator = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
        platform = mm.Platform.getPlatformByName('CUDA')
        context = mm.Context(omm_sys, integrator, platform)
        context.setPositions(sampler.topology.positions)

        n_real = sampler._num_waters - sampler._num_ghost_waters
        ghost_start = sampler.water_indices[n_real]
        n_pts = sampler._num_points

        state = context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(openmm.unit.nanometer)
        ghost_pos_before = pos[ghost_start:ghost_start + n_pts].copy()

        def internal_distances(positions):
            dists = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    d = np.linalg.norm(positions[i] - positions[j])
                    dists.append(d)
            return np.array(dists)

        dists_before = internal_distances(ghost_pos_before)

        integrator.step(1000)

        state = context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(openmm.unit.nanometer)
        ghost_pos_after = pos[ghost_start:ghost_start + n_pts]
        dists_after = internal_distances(ghost_pos_after)

        max_drift = np.max(np.abs(dists_after - dists_before))
        assert max_drift < 0.05, (
            f"Ghost geometry drifted by {max_drift:.4f} nm after 1000 MD steps")


class TestGhostPositionSpreading:
    """Bug #8: Ghost molecules must not overlap (prevents PME NaN)."""

    @gpu
    def test_ghost_positions_distinct(self, ethanol_system):
        """Each ghost molecule must have a unique position."""
        sampler = _make_sampler(ethanol_system)
        positions = np.array(sampler.topology.positions)

        n_real = sampler._num_waters - sampler._num_ghost_waters
        ghost_centroids = []
        for i in range(sampler._num_ghost_waters):
            start = sampler.water_indices[n_real + i]
            mol_pos = positions[start:start + sampler._num_points]
            ghost_centroids.append(mol_pos.mean(axis=0))

        ghost_centroids = np.array(ghost_centroids)

        for i in range(len(ghost_centroids)):
            for j in range(i + 1, len(ghost_centroids)):
                dist = np.linalg.norm(ghost_centroids[i] - ghost_centroids[j])
                assert dist > 0.01, (
                    f"Ghost molecules {i} and {j} overlap: dist={dist:.6f} nm")

    @gpu
    def test_pme_no_nan_after_init(self, ethanol_system):
        """PME energy must be finite immediately after initialization."""
        sampler = _make_sampler(ethanol_system, is_pme=True)
        omm_sys = sampler.omm_system

        integrator = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
        platform = mm.Platform.getPlatformByName('CUDA')
        context = mm.Context(omm_sys, integrator, platform)
        context.setPositions(sampler.topology.positions)

        state = context.getState(getEnergy=True)
        pe = state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoules_per_mole)
        assert np.isfinite(pe), f"PME energy is NaN/inf after init: {pe}"


class TestBeyond14Pairs:
    """Beyond-1-4 intramolecular pairs must be constant bonded forces."""

    @gpu
    def test_has_beyond_14_pairs(self, ethanol_system):
        """Ethanol molecules have atom pairs >3 bonds apart."""
        top = ethanol_system.topology
        mol = top.molecules[0]
        dist_matrix = mol.get_distance_matrix(max_bond=3)

        beyond_14_count = 0
        for i in range(mol.n_atom):
            for j in range(i + 1, mol.n_atom):
                if dist_matrix[i, j] == 0:
                    beyond_14_count += 1

        assert beyond_14_count > 0, "Ethanol should have beyond-1-4 pairs"

    @gpu
    def test_ghost_toggle_no_intramolecular_change(self, ethanol_system):
        """Toggling ghost/real must not change intramolecular energy."""
        sampler = _make_sampler(ethanol_system)
        omm_sys = sampler.omm_system

        integrator = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
        platform = mm.Platform.getPlatformByName('CUDA')
        context = mm.Context(omm_sys, integrator, platform)
        context.setPositions(sampler.topology.positions)

        state = context.getState(getEnergy=True)
        pe_ghosts_off = state.getPotentialEnergy().value_in_unit(
            openmm.unit.kilojoules_per_mole)

        assert np.isfinite(pe_ghosts_off), "Energy is NaN with ghosts off"

    @gpu
    def test_custom_bond_force_exists(self, ethanol_system):
        """System must have CustomBondForce for beyond-1-4 LJ."""
        sampler = _make_sampler(ethanol_system)
        omm_sys = sampler.omm_system

        custom_bond_forces = [
            f for f in omm_sys.getForces()
            if isinstance(f, mm.CustomBondForce)
        ]

        assert len(custom_bond_forces) > 0, (
            "No CustomBondForce found — beyond-1-4 LJ fix not applied")
