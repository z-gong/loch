import copy

import numpy as np
from mstk import logger
from mstk.topology import Topology


def save_real_frame(topology, water_indices, water_state,
                    residue_name, psf_file, gro_file):
    """Save a frame containing only real (non-ghost) molecules.

    Parameters
    ----------
    topology : Topology
        Extended topology (real + ghost) with current positions set.
    """
    positions = topology.positions
    template_mol = next(mol for mol in topology.molecules if mol.name == residue_name)
    n_points = len(template_mol.atoms)
    n_non_gcmc_atoms = water_indices[0]
    ghost_slots = set(np.where(water_state == 0)[0].tolist())

    subset = list(range(n_non_gcmc_atoms))
    for mol_idx in range(len(water_indices)):
        if mol_idx not in ghost_slots:
            start = water_indices[mol_idx]
            subset.extend(range(start, start + n_points))

    # Build topology with only real molecules.
    real_mols = [mol for mol in topology.molecules if mol.name != residue_name]
    n_real_gcmc = len(water_indices) - len(ghost_slots)
    for _ in range(n_real_gcmc):
        real_mols.append(copy.deepcopy(template_mol))

    real_top = Topology(real_mols)
    real_top.cell = topology.cell
    real_top.set_positions(positions[subset])
    real_top.write(psf_file)
    real_top.write(gro_file)
    logger.info(f'Wrote {psf_file} and {gro_file} ({real_top.n_atom} atoms, '
                f'{len(real_mols)} molecules)')
