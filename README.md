# Loch

[![License: GPL v3](https://img.shields.io/badge/License-GPL_v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)

CUDA/OpenCL accelerated Grand Canonical Monte Carlo (GCMC) simulation code.
Barostat support enables osmotic ensemble sampling for multi-component systems.
Built on top of [mstk](https://github.com/z-gong/mstk),
[OpenMM](https://github.com/openmm/openmm),
[PyCUDA](https://documen.tician.de/pycuda/index.html#),
and [PyOpenCL](https://documen.tician.de/pyopencl/).

## How it works

OpenMM does not support variable particle counts, so Loch uses a **ghost
molecule** strategy: a pool of non-interacting molecules is pre-allocated,
and insertion/deletion is achieved by toggling their nonbonded parameters.

Instead of computing the energy change for each trial insertion/deletion with
OpenMM, the calculation is performed at the reaction field (RF) level using
a custom CUDA/OpenCL kernel, allowing multiple candidates to be evaluated
simultaneously. Particle mesh Ewald (PME) is handled via the method for
sampling from an approximate potential (in this case the RF potential)
introduced [here](https://doi.org/10.1063/1.1563597). Parallelisation of the
insertion and deletion trials is achieved using the strategy described in
[this](https://doi.org/10.1021/acs.jctc.0c00660) paper.

## Installation

Loch requires OpenMM, mstk and PyCUDA (or PyOpenCL).
A conda environment file is provided:

```bash
conda env update -f conda_env.yaml
pip install -e .
```

## Usage

### Command-line interface

**GCMC simulation:**

```bash
loch gcmc -p top.psf -c conf.gro -f primitive.zff \
    --resname ETOL --mu -25.4 --vol 0.0946 --nghost 50 \
    --cycle 500 -n 1000 -t 300 --nattempt 10000 --batch 1000
```

### Python API

```python
from mstk.topology import Topology
from mstk.trajectory import Trajectory
from mstk.forcefield import ForceField
from mstk.simsys import System
from loch import GCMCSampler
from openmm import openmm

# Build system from mstk.
top = Topology.open('top.psf')
frame = Trajectory.read_frame_from_file('conf.gro', -1)
top.cell.set_box(frame.cell.vectors)
top.set_positions(frame.positions)
ff = ForceField.open('primitive.zff')
system = System(top, ff)

# Create sampler.
sampler = GCMCSampler(
    system,
    residue_name='ETOL',
    excess_chemical_potential=-25.4,  # kJ/mol, from mstk sfe
    standard_volume=0.0946,           # nm^3, molecular volume from NPT
    temperature=300,                  # K
    num_ghost_waters=50,
    num_attempts=10000,
    batch_size=1000,
)

# Access the OpenMM system for dynamics setup.
omm_system = sampler.omm_system
omm_topology = sampler.topology.to_omm_topology()
positions = sampler.topology.positions

# The simulation loop alternates MD integration with GCMC moves:
integrator = openmm.LangevinMiddleIntegrator(300, 1.0, 0.002)
platform = openmm.Platform.getPlatformByName('CUDA')
properties = {'Precision': 'mixed'}
context = openmm.Context(omm_system, integrator, platform, properties)
context.setPositions(positions)

for cycle in range(500):
    sampler.move(context)           # GCMC move
    integrator.step(1000)           # 2 ps MD

print(f"N = {sampler.num_waters()}, "
      f"ins = {sampler.num_insertions}, "
      f"del = {sampler.num_deletions}")
```

## Calibrating GCMC parameters

GCMC requires two thermodynamic inputs for the inserted species:

- **mu_ex** — excess chemical potential (kJ/mol)
- **V_std** — standard molecular volume (nm^3)

mu_ex is computed using `mstk sfe`, which performs alchemical decoupling
of a single molecule from its bulk liquid. The molecular volume is obtained
from the equilibrium volume of the pure liquid (V_std = V_box / N).

## Supported force fields

Loch supports force fields with:

- **LJ 12-6** nonbonded interactions with long range correction.
- **Lorentz-Berthelot** or **geometric** combining rules
- Any bonded functional form supported by mstk (harmonic, quartic, Morse
  bonds; harmonic, SDK, linear angles; OPLS, periodic dihedrals; etc.)

The following are **not** currently supported:

- Mie or Morse VdW potentials
- Explicit pairwise VdW parameters (pair-specific overrides)
- Shifted LJ potential (`vdw_long_range='shift'`)
- Polarizable force fields (Drude)
- Virtual sites (TIP4P, etc.)

## Output files

From `loch gcmc` CLI:

| File | Contents |
|------|----------|
| `gcmc_full.psf` | Full topology (real + ghost), for visualization |
| `dump.xtc` | Trajectory (all atoms including ghosts) |
| `gcmc_state.csv` | Per-cycle: N, insertions, deletions, ghost count |
| `gcmc_final.psf` | Final topology, real atoms only |
| `gcmc_final.gro` | Final coordinates, real atoms only |

To extract a single frame with only real atoms from the trajectory:

```bash
loch extract -p gcmc_full.psf -c dump.xtc -s gcmc_state.csv --frame -1 -o out
```

## Notes

- GCMC supports both **muVT** and **muPT** ensembles. Use
  `--barostat iso` to enable constant-pressure sampling; the acceptance
  criterion adapts to the current volume each cycle.
- When using CUDA, ensure `nvcc` is in your PATH. Set `PYCUDA_NVCC` to
  override the compiler location.
- The GPU platform is auto-detected (CUDA preferred, OpenCL fallback).

## Acknowledgements

This project is a fork of [loch](https://github.com/openbiosim/loch)
by [OpenBioSim](https://github.com/OpenBioSim)

The original acknowledgements from the upstream project:
* We thank the [Essex Lab](https://essexgroup.soton.ac.uk/) and
  [grand](https://github.com/essex-lab/grand) for the inspiration.
* Many thanks to [Gregory Ross](https://github.com/gregoryross) for clarifying
  the parallelisation scheme described [here](https://doi.org/10.1021/acs.jctc.0c00660).
