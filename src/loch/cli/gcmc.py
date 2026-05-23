import os
import sys

import click
import numpy as np
from openmm import openmm as mm, app
from mstk import logger
from mstk.topology import Topology
from mstk.trajectory import Trajectory
from mstk.forcefield import ForceField
from mstk.simsys import System
from mstk import ommhelper as oh
from mstk.ommhelper.unit import kelvin, ps

from .. import GCMCSampler
from .utils import save_real_frame


@click.command(help='Run hybrid MD/GCMC simulation')
@click.option('-p', '--top', 'top_file', type=click.Path(exists=True), required=True,
              help='topology file')
@click.option('-c', '--conf', 'conf_file', type=click.Path(exists=True), required=True,
              help='configuration file')
@click.option('-f', '--ff', 'ff_file', required=True, help='force field file')
@click.option('--resname', required=True, help='residue name of GCMC molecule')
@click.option('--mu', required=True, type=float, help='excess chemical potential in kJ/mol')
@click.option('--vol', required=True, type=float, help='standard molar volume in nm^3')
@click.option('--mol-top', 'mol_top_file', type=click.Path(exists=True), default=None,
              help='topology for inserted molecule (if not in system)')
@click.option('--mol-conf', 'mol_conf_file', type=click.Path(exists=True), default=None,
              help='configuration for inserted molecule')
@click.option('--nghost', default=50, help='number of ghost molecules')
@click.option('--cycle', default=1000, help='number of MD/GCMC cycles')
@click.option('-n', '--nstep', default=1000, help='MD steps per cycle')
@click.option('--dt', default=0.002, help='timestep in ps')
@click.option('-t', '--temp', default=300.0, help='temperature in K')
@click.option('--press', default=1.0, help='pressure in bar')
@click.option('--barostat', default='none', type=click.Choice(['none', 'iso']),
              help='barostat type (none=NVT, iso=isotropic NPT)')
@click.option('--ncyclesave', default=10, help='save XTC and state every N cycles')
@click.option('--nattempt', default=10000, help='GCMC attempts per cycle')
@click.option('--batch', default=1000, help='batch size for GPU trials')
@click.option('--rf', 'use_rf', is_flag=True, help='use reaction field instead of PME')
@click.option('--reference', type=str, default=None,
              help='comma-separated atom indices for GCMC sphere center')
@click.option('--radius', default=0.4, help='GCMC sphere radius in nm')
@click.option('--bulk-prob', default=0.1, help='probability of bulk sampling')
@click.option('--insert-only', is_flag=True, help='only attempt insertions (breaks detailed balance)')
def gcmc(top_file, conf_file, ff_file, resname, mu, vol,
         mol_top_file, mol_conf_file, nghost, cycle, nstep, dt,
         temp, press, barostat, ncyclesave, nattempt, batch,
         use_rf, reference, radius, bulk_prob, insert_only):
    run_gcmc(**locals())


def run_gcmc(top_file, conf_file, ff_file, resname, mu, vol,
             mol_top_file, mol_conf_file, nghost, cycle, nstep, dt,
             temp, press, barostat, ncyclesave, nattempt, batch,
             use_rf, reference, radius, bulk_prob, insert_only):
    for _k, _v in locals().items():
        logger.info(f'Arg: {_k:12s} = {_v}')
    if insert_only:
        logger.warning('insert-only mode: detailed balance is broken, not for production')
    oh.print_omm_info(logger=logger)

    # Load system.
    top = Topology.open(top_file)
    frame = Trajectory.read_frame_from_file(conf_file, -1)
    top.cell.set_box(frame.cell.vectors)
    top.set_positions(frame.positions)

    # For dry systems, add template molecule to topology before building System.
    ghost_existing = False
    if mol_top_file is not None:
        mol_top = Topology.open(mol_top_file)
        if mol_conf_file is not None:
            mol_frame = Trajectory.read_frame_from_file(mol_conf_file, -1)
            mol_top.set_positions(mol_frame.positions)
        top.update_molecules(top.molecules + mol_top.molecules[:1])
        ghost_existing = True

    logger.info(top)
    ff = ForceField.open(ff_file)
    system = System(top, ff)

    # Parse reference atom indices.
    ref_indices = None
    if reference is not None:
        ref_indices = [int(x.strip()) for x in reference.split(',')]

    # Create GCMC sampler.
    sampler = GCMCSampler(
        system,
        residue_name=resname,
        ghost_existing=ghost_existing,
        reference=ref_indices,
        radius=radius,
        is_pme=not use_rf,
        excess_chemical_potential=mu,
        standard_volume=vol,
        temperature=temp,
        num_ghost_waters=nghost,
        batch_size=batch,
        num_attempts=nattempt,
        bulk_sampling_probability=bulk_prob,
        insert_only=insert_only,
    )

    # Create OpenMM simulation.
    if barostat != 'none':
        # use constrained group scaling if the base system is fully bonded
        oh.apply_mc_barostat(sampler.omm_system, barostat, press, temp,
                             nstep=100, rigid_molecule=top.n_molecule >= 100, logger=logger)
    integrator = mm.LangevinMiddleIntegrator(temp * kelvin, 1.0 / ps, dt * ps)
    platform, properties = oh.get_platform_properties()
    omm_top = sampler.topology.to_omm_topology()
    sim = app.Simulation(omm_top, sampler.omm_system, integrator, platform, properties)
    sim.context.setPositions(sampler.topology.positions)
    sim.reporters.append(oh.StateDataReporter(sys.stdout, 1000))

    oh.energy_decomposition(sim, logger=logger)

    # Write full topology (real + ghost) for VMD.
    sampler.topology.write('gcmc_full.psf')
    logger.info(f'Wrote gcmc_full.psf {sampler.topology}')

    # Initialize state file.
    state_file = 'gcmc_state.csv'
    with open(state_file, 'w') as f:
        f.write(f'# resname={resname}\n')
        f.write('cycle,N,n_inserted,n_deleted,ghost_slots\n')

    def write_state(cyc):
        ghost_slots = np.where(sampler.water_state == 0)[0]
        ghosts_str = ' '.join(str(g) for g in ghost_slots)
        N = int(np.sum(sampler.water_state == 1))
        with open(state_file, 'a') as f:
            f.write(f'{cyc},{N},{sampler.num_insertions},{sampler.num_deletions},{ghosts_str}\n')

    write_state(0)
    open('dump.xtc', 'wb').close()
    _xtc = app.XTCFile('dump.xtc', omm_top, dt, interval=ncyclesave*nstep or 1)
    state = sim.context.getState(getPositions=True)
    _xtc.writeModel(state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors())

    # Precompute masses for density calculation.
    non_gcmc_mass = sum(a.mass for mol in top.molecules
                        if mol.name != resname for a in mol.atoms)
    gcmc_mol = next(mol for mol in top.molecules if mol.name == resname)
    gcmc_mol_mass = sum(a.mass for a in gcmc_mol.atoms)

    # GCMC/MD loop.
    logger.info(f'Starting {cycle} cycles: {nattempt} GCMC attempts + {nstep} MD steps')
    for i in range(cycle):
        sampler.move(sim.context)

        if (i + 1) % ncyclesave == 0:
            write_state(i + 1)

        if i == 0 or (i + 1) % 10 == 0 or i == cycle - 1:
            n_w = sampler.num_waters(sim.context)
            box_vol = np.prod(sampler.box_size)
            density = (non_gcmc_mass + n_w * gcmc_mol_mass) / (box_vol * 602.214)
            logger.info(f'Cycle {i + 1:4d} | N={n_w:4d} | ρ={density:.4f} | V={box_vol:.2f} | '
                        f'ins={sampler.num_insertions} del={sampler.num_deletions}')

        if nstep > 0:
            sim.step(nstep)

        if (i + 1) % ncyclesave == 0:
            state = sim.context.getState(getPositions=True)
            _xtc.writeModel(state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors())

        if sampler.ghost_exhausted:
            break

    logger.info(f'Done. Final N={sampler.num_waters(sim.context)}, '
                f'acceptance_ratio={sampler.move_acceptance_probability():.1e}')

    # Write final state with only real atoms.
    state = sim.context.getState(getPositions=True)
    all_pos = state.getPositions(asNumpy=True)._value
    box_vectors = state.getPeriodicBoxVectors(asNumpy=True)._value
    sampler.topology.set_positions(all_pos)
    sampler.topology.cell.set_box(box_vectors)
    save_real_frame(sampler.topology, sampler.water_indices, sampler.water_state,
                    resname, 'gcmc_final.psf', 'gcmc_final.gro')

    os._exit(0)
