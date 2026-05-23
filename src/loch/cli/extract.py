import os

import click
import numpy as np
from mstk import logger
from mstk.topology import Topology
from mstk.trajectory import Trajectory

from .utils import save_real_frame


@click.command(help='Extract a frame from GCMC trajectory with only real atoms')
@click.option('-p', '--top', 'top_file', type=click.Path(exists=True), required=True,
              help='full topology file (gcmc_full.psf, includes ghost molecules)')
@click.option('-c', '--conf', 'conf_file', type=click.Path(exists=True), required=True,
              help='trajectory file (XTC from loch gcmc)')
@click.option('-s', '--state', 'state_file', type=click.Path(exists=True), required=True,
              help='GCMC state file (gcmc_state.csv)')
@click.option('--frame', required=True, type=int,
              help='frame index to extract. 0-based, matches XTC frame and state file row.'
                   ' A negative value has the same meaning as in Python list')
@click.option('-o', '--output', 'output_file', required=True,
              help='output basename (writes <basename>.psf and <basename>.gro)')
def extract(top_file, conf_file, state_file, frame, output_file):
    # Read resname from state file metadata.
    resname = None
    with open(state_file) as f:
        first_line = f.readline()
        if first_line.startswith('# resname='):
            resname = first_line.strip().split('=', 1)[1]
    if resname is None:
        raise ValueError("Cannot determine resname from state file.")

    top = Topology.open(top_file)
    logger.info(top)

    # Derive water_indices and n_points from topology.
    water_indices = []
    n_points = None
    for mol in top.molecules:
        if mol.name == resname:
            water_indices.append(mol.atoms[0].id)
            if n_points is None:
                n_points = len(mol.atoms)
    water_indices = np.array(water_indices)

    if len(water_indices) == 0:
        raise ValueError(f"No molecules with resname '{resname}' found in topology")

    n_total_mols = len(water_indices)
    logger.info(f'{n_total_mols} GCMC molecules ({n_points} atoms each) in topology')

    # Normalize negative frame index using trajectory length.
    trj = Trajectory.open(conf_file)
    logger.info(trj)
    if frame < 0:
        frame += trj.n_frame

    # Parse state file for ghost slots.
    ghost_slots_per_row = []
    with open(state_file) as f:
        for line in f:
            if line.startswith('#') or line.startswith('cycle'):
                continue
            parts = line.strip().split(',', 4)
            if len(parts) >= 5 and parts[4].strip():
                slots = [int(x) for x in parts[4].split()]
            else:
                slots = []
            ghost_slots_per_row.append(slots)

    n_state_rows = len(ghost_slots_per_row)
    if frame < 0 or frame >= n_state_rows:
        raise ValueError(f"Frame {frame} out of range (state file has {n_state_rows} rows)")

    # Reconstruct water_state for this frame.
    water_state = np.ones(n_total_mols, dtype=np.int32)
    for s in ghost_slots_per_row[frame]:
        water_state[s] = 0
    n_real = int(water_state.sum())
    logger.info(f'Frame {frame}: {n_real} real, {n_total_mols - n_real} ghost')

    fr = trj.read_frame(frame)
    top.cell.set_box(fr.cell.vectors)
    top.set_positions(fr.positions)
    base = os.path.splitext(output_file)[0]
    save_real_frame(top, water_indices, water_state, resname, base + '.psf', base + '.gro')
