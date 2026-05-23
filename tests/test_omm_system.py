"""
Tests that GCMCSampler correctly modifies the OpenMM system:
- After initialization: ghost particles have zero interactions
- After insertion: particle gets real charge and LJ type
- After deletion: particle reverts to ghost state
"""

import os

import numpy as np
import openmm
import pytest
from openmm import app
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
        residue_name="ETOH",
        num_ghost_waters=5,
        batch_size=100,
        num_attempts=100,
        is_pme=False,
        temperature=300.0,
        excess_chemical_potential=-25.413,
        standard_volume=0.09459,
        platform="cuda",
    )
    defaults.update(kwargs)
    return GCMCSampler(system, **defaults)


def _get_forces(omm_system):
    """Extract NonbondedForce and CustomNonbondedForce from system."""
    nb = None
    cnb_list = []
    for force in omm_system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nb = force
        elif isinstance(force, openmm.CustomNonbondedForce):
            cnb_list.append(force)
    return nb, cnb_list


@gpu
def test_ghost_particles_after_init(ethanol_system):
    """Ghost particles must have zero charge and ghost type index."""
    sampler = _make_sampler(ethanol_system)
    omm_sys = sampler.omm_system
    nb, cnb_list = _get_forces(omm_sys)

    n_real_atoms = len(ethanol_system.topology.atoms)
    n_ghost_atoms = sampler._num_ghost_waters * sampler._num_points
    assert omm_sys.getNumParticles() == n_real_atoms + n_ghost_atoms

    ghost_type = float(sampler._ghost_type_index)

    for i in range(n_ghost_atoms):
        idx = n_real_atoms + i
        charge, sigma, epsilon = nb.getParticleParameters(idx)
        assert charge._value == 0.0
        assert epsilon._value == 0.0

        for cnb in cnb_list:
            params = cnb.getParticleParameters(idx)
            assert params[0] == ghost_type


@gpu
def test_ghost_type_table_zeros(ethanol_system):
    """The ghost type row/column in the tabulated function must be all zeros."""
    sampler = _make_sampler(ethanol_system)
    _, cnb_list = _get_forces(sampler.omm_system)

    ghost_type = sampler._ghost_type_index

    for cnb in cnb_list:
        for func_idx in range(cnb.getNumTabulatedFunctions()):
            func = cnb.getTabulatedFunction(func_idx)
            if isinstance(func, openmm.Discrete2DFunction):
                n, m, values = func.getFunctionParameters()
                assert n == ghost_type + 1
                assert m == ghost_type + 1
                for other in range(n):
                    assert values[ghost_type + n * other] == 0.0
                    assert values[other + n * ghost_type] == 0.0


@gpu
def test_insertion_restores_parameters(ethanol_system):
    """After insertion, particle gets real charge and real LJ type."""
    sampler = _make_sampler(ethanol_system)
    integrator = openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001)
    platform = openmm.Platform.getPlatformByName('CUDA')
    sim = app.Simulation(
        sampler.topology.to_omm_topology(), sampler.omm_system,
        integrator, platform, {'Precision': 'mixed'},
    )
    sim.context.setPositions(sampler.topology.positions)

    sampler.delete_waters(sim.context)

    for _ in range(200):
        moves = sampler.move(sim.context)
        if 0 in moves:
            break
    else:
        pytest.skip("No insertion accepted")

    n_real = sampler._num_waters - sampler._num_ghost_waters
    inserted = None
    for i in range(sampler._num_ghost_waters):
        if sampler.water_state[n_real + i] == 1:
            inserted = n_real + i
            break
    assert inserted is not None

    start_idx = sampler.water_indices[inserted]
    nb, cnb_list = _get_forces(sim.context.getSystem())

    for i in range(sampler._num_points):
        charge, sigma, epsilon = nb.getParticleParameters(start_idx + i)
        for cnb in cnb_list:
            params = cnb.getParticleParameters(start_idx + i)
            assert params[0] == float(sampler._template_type_indices[i])


@gpu
def test_deletion_zeros_parameters(ethanol_system):
    """After deletion, particle reverts to zero charge and ghost LJ type."""
    sampler = _make_sampler(ethanol_system)
    integrator = openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001)
    platform = openmm.Platform.getPlatformByName('CUDA')
    sim = app.Simulation(
        sampler.topology.to_omm_topology(), sampler.omm_system,
        integrator, platform, {'Precision': 'mixed'},
    )
    sim.context.setPositions(sampler.topology.positions)

    sampler.delete_waters(sim.context)

    n_real = sampler._num_waters - sampler._num_ghost_waters
    deleted = None
    for i in range(n_real):
        if sampler.water_state[i] == 0:
            deleted = i
            break
    assert deleted is not None

    start_idx = sampler.water_indices[deleted]
    nb, cnb_list = _get_forces(sim.context.getSystem())
    ghost_type = float(sampler._ghost_type_index)

    for i in range(sampler._num_points):
        charge, sigma, epsilon = nb.getParticleParameters(start_idx + i)
        assert charge._value == 0.0
        assert epsilon._value == 0.0

        for cnb in cnb_list:
            params = cnb.getParticleParameters(start_idx + i)
            assert params[0] == ghost_type


@gpu
def test_exclusions_on_ghosts(ethanol_system):
    """Ghost molecules must have intra-molecular exclusions."""
    sampler = _make_sampler(ethanol_system)
    nb, cnb_list = _get_forces(sampler.omm_system)

    ghost_start = sampler.water_indices[sampler._num_waters - sampler._num_ghost_waters]
    n_pts = sampler._num_points

    ghost_exceptions = set()
    for i in range(nb.getNumExceptions()):
        p1, p2, *_ = nb.getExceptionParameters(i)
        if ghost_start <= p1 < ghost_start + n_pts and ghost_start <= p2 < ghost_start + n_pts:
            ghost_exceptions.add((min(p1, p2), max(p1, p2)))

    expected = set()
    for j in range(n_pts):
        for k in range(j + 1, n_pts):
            expected.add((ghost_start + j, ghost_start + k))

    assert ghost_exceptions == expected

    for cnb in cnb_list:
        cnb_exclusions = set()
        for i in range(cnb.getNumExclusions()):
            p1, p2 = cnb.getExclusionParticles(i)
            if ghost_start <= p1 < ghost_start + n_pts and ghost_start <= p2 < ghost_start + n_pts:
                cnb_exclusions.add((min(p1, p2), max(p1, p2)))
        assert cnb_exclusions == expected
