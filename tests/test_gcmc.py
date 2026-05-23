"""
GCMC integration tests.

Covers:
- RF energy audit: kernel dE must match OpenMM energy change
- Ghost state toggling: insertion increases N, deletion decreases N
- MD+GCMC loop stability: no NaN or state corruption
- Platform consistency: CUDA and OpenCL produce valid results
- Acceptance counter: probability in [0, 1]
"""

import math
import os

import numpy as np
import openmm
import openmm.openmm as mm
import pytest
from openmm import unit
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
        num_ghost_waters=20,
        batch_size=100,
        num_attempts=100,
        is_pme=False,
        temperature=300.0,
        excess_chemical_potential=-25.413,
        standard_volume=0.09459,
        reference = [2, 11, 20],
        platform="cuda",
    )
    defaults.update(kwargs)
    return GCMCSampler(system, **defaults)


def _get_rf_energy_from_context(context):
    """Get potential energy in kcal/mol."""
    state = context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)


@gpu
class TestRFEnergyInsertion:
    """After accepted RF insertion, kernel dE must match OpenMM dE."""

    def test_insertion_energy(self, ethanol_system):
        sampler = _make_sampler(ethanol_system)

        integrator = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
        platform = mm.Platform.getPlatformByName('CUDA')
        context = mm.Context(sampler.omm_system, integrator, platform)
        context.setPositions(sampler.topology.positions)

        initial_energy = _get_rf_energy_from_context(context)

        for attempt in range(200):
            moves = sampler.move(context)
            if moves and moves[0] == 0:
                break
        else:
            pytest.skip("No insertion accepted after 200 attempts")

        final_energy = _get_rf_energy_from_context(context)
        openmm_dE = final_energy - initial_energy

        energy_changes = sampler._backend.from_gpu(sampler._energy_change).flatten()
        accepted_arr = sampler._backend.from_gpu(sampler._accepted).flatten()
        idx = np.where(accepted_arr == 1)[0][0]
        kernel_dE = energy_changes[idx]

        assert math.isfinite(openmm_dE), f"OpenMM energy is NaN: {openmm_dE}"
        assert math.isfinite(kernel_dE), f"Kernel energy is NaN: {kernel_dE}"
        assert math.isclose(openmm_dE, kernel_dE, abs_tol=0.1), (
            f"Energy mismatch: OpenMM dE={openmm_dE:.4f}, kernel dE={kernel_dE:.4f}, "
            f"diff={abs(openmm_dE - kernel_dE):.4f} kcal/mol"
        )


@gpu
class TestRFEnergyDeletion:
    """After accepted RF deletion, kernel dE must match OpenMM dE."""

    def test_deletion_energy(self, ethanol_system):
        sampler = _make_sampler(ethanol_system)

        integrator = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
        platform = mm.Platform.getPlatformByName('CUDA')
        context = mm.Context(sampler.omm_system, integrator, platform)
        context.setPositions(sampler.topology.positions)

        initial_energy = _get_rf_energy_from_context(context)

        for attempt in range(200):
            moves = sampler.move(context)
            if moves and moves[0] == 1:
                break
        else:
            pytest.skip("No deletion accepted after 200 attempts")

        final_energy = _get_rf_energy_from_context(context)
        openmm_dE = final_energy - initial_energy

        energy_changes = sampler._backend.from_gpu(sampler._energy_change).flatten()
        accepted_arr = sampler._backend.from_gpu(sampler._accepted).flatten()
        idx = np.where(accepted_arr == 1)[0][0]
        kernel_dE = -energy_changes[idx]

        assert math.isfinite(openmm_dE), f"OpenMM energy is NaN: {openmm_dE}"
        assert math.isfinite(kernel_dE), f"Kernel energy is NaN: {kernel_dE}"
        assert math.isclose(openmm_dE, kernel_dE, abs_tol=0.1), (
            f"Energy mismatch: OpenMM dE={openmm_dE:.4f}, kernel dE={kernel_dE:.4f}, "
            f"diff={abs(openmm_dE - kernel_dE):.4f} kcal/mol"
        )


@gpu
class TestMDGCMCStability:
    """MD+GCMC loop must not produce NaN or corrupt state."""

    def test_10_cycles(self, ethanol_system):
        """10 cycles of MD+GCMC with ethanol, energy stays finite."""
        sampler = _make_sampler(ethanol_system)

        integrator = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
        platform = mm.Platform.getPlatformByName('CUDA')
        context = mm.Context(sampler.omm_system, integrator, platform)
        context.setPositions(sampler.topology.positions)

        for cycle in range(10):
            integrator.step(500)
            sampler.move(context)

        pe = _get_rf_energy_from_context(context)
        assert math.isfinite(pe), f"Energy NaN after 10 cycles: {pe}"

        n = sampler.num_waters()
        assert n > 0, "All molecules deleted"

    def test_acceptance_counter_consistent(self, ethanol_system):
        """Acceptance probability must be in [0, 1]."""
        sampler = _make_sampler(ethanol_system)

        integrator = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
        platform = mm.Platform.getPlatformByName('CUDA')
        context = mm.Context(sampler.omm_system, integrator, platform)
        context.setPositions(sampler.topology.positions)

        for cycle in range(5):
            integrator.step(500)
            sampler.move(context)

        acc = sampler.move_acceptance_probability()
        assert 0.0 <= acc <= 1.0, f"Acceptance probability out of range: {acc}"
        assert sampler.num_insertions + sampler.num_deletions == sampler._num_accepted


@gpu
class TestStateToggling:
    """Insertion increases N, deletion decreases N."""

    def test_insertion_increases_n(self, ethanol_system):
        sampler = _make_sampler(ethanol_system)

        integrator = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
        platform = mm.Platform.getPlatformByName('CUDA')
        context = mm.Context(sampler.omm_system, integrator, platform)
        context.setPositions(sampler.topology.positions)

        sampler.delete_waters(context)
        assert sampler.num_waters() == 0

        for _ in range(100):
            sampler.move(context)
            if sampler.num_insertions > 0:
                break

        assert sampler.num_waters() > 0, "No insertion occurred"

    def test_deletion_decreases_n(self, ethanol_system):
        sampler = _make_sampler(ethanol_system)

        integrator = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
        platform = mm.Platform.getPlatformByName('CUDA')
        context = mm.Context(sampler.omm_system, integrator, platform)
        context.setPositions(sampler.topology.positions)

        n_before = sampler.num_waters()

        for _ in range(100):
            sampler.move(context)
            if sampler.num_deletions > 0:
                break

        assert sampler.num_waters() < n_before, "No deletion occurred"


@gpu
class TestPlatformConsistency:
    """Both CUDA and OpenCL must complete moves without error."""

    @pytest.mark.parametrize("platform", ["cuda", "opencl"])
    def test_move_completes(self, ethanol_system, platform):
        sampler = _make_sampler(ethanol_system, platform=platform)

        integrator = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
        omm_platform = mm.Platform.getPlatformByName(
            'CUDA' if platform == 'cuda' else 'OpenCL')
        context = mm.Context(sampler.omm_system, integrator, omm_platform)
        context.setPositions(sampler.topology.positions)

        moves = sampler.move(context)
        assert isinstance(moves, list)

        pe = _get_rf_energy_from_context(context)
        assert math.isfinite(pe)
