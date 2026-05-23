######################################################################
# Loch: GPU accelerated GCMC sampling engine.
#
# Copyright: 2025-2026
#
# Authors: The OpenBioSim Team <team@openbiosim.org>
#
# Loch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Loch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Loch. If not, see <http://www.gnu.org/licenses/>.
#####################################################################

__all__ = ["GCMCSampler"]

from typing import Optional as _Optional, Union as _Union

import numpy as _np
import openmm as _openmm
import os as _os

from mstk import logger as _logger

from mstk.simsys import System as _MstkSystem
from mstk.topology import Topology as _MstkTopology
from mstk.forcefield import ForceField as _ForceField
from mstk.forcefield import LJ126Term as _LJ126Term

from ._platforms import create_backend as _create_backend
from ._platforms._rng import RNGManager as _RNGManager


def _as_float32(arr: _np.ndarray) -> _np.ndarray:
    return arr if arr.dtype == _np.float32 else arr.astype(_np.float32)


def _as_int32(arr: _np.ndarray) -> _np.ndarray:
    return arr if arr.dtype == _np.int32 else arr.astype(_np.int32)


# Boltzmann constant in kJ/(mol*K)
_KB_KJMOL = 0.008314462618


class GCMCSampler:
    """
    GPU-accelerated Grand Canonical Monte Carlo sampler.

    Attributes (read-only)
    ----------------------
    topology : mstk.Topology
        Extended topology (real + ghost molecules) with positions.
    omm_system : openmm.System
        OpenMM system with ghost nonbonded parameters zeroed.
    water_state : np.ndarray
        Per-molecule state: 0=ghost, 1=real.
    water_indices : np.ndarray
        First-atom OpenMM index for each GCMC molecule.
    box_size : np.ndarray
        Current box dimensions in nm (shape (3,)).
    num_insertions : int
        Cumulative accepted insertions.
    num_deletions : int
        Cumulative accepted deletions.
    ghost_exhausted : bool
        True if no ghost slots remain for insertion.
    """

    def __init__(
        self,
        system: _MstkSystem,
        residue_name: str,
        ghost_existing: bool = False,
        reference: _Optional[list] = None,
        radius: float = 0.4,
        is_pme: bool = True,
        excess_chemical_potential: float = -25.5,
        standard_volume: float = 0.030543,
        temperature: float = 298.0,
        adams_shift: float = 0.0,
        num_ghost_waters: int = 20,
        batch_size: int = 1000,
        num_attempts: int = 10000,
        num_threads: int = 1024,
        bulk_sampling_probability: float = 0.1,
        insert_only: bool = False,
        device: _Optional[int] = None,
        platform: str = "auto",
        tolerance: float = 0.0,
        seed: _Optional[int] = None,
        nvcc: _Optional[str] = None,
        compiler_optimisations: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        system : mstk.simsys.System
            The molecular system (topology + force field). Must contain at
            least one molecule with matching residue_name.
        residue_name : str
            Residue name of the molecule to insert/delete.
        ghost_existing : bool
            If True, all pre-existing molecules matching residue_name start
            as ghosts (non-interacting). Use for dry systems where a template
            molecule was added to the topology just to complete the force field.
        reference : list of int, optional
            Atom indices defining the GCMC sphere center. If None,
            insertions/deletions occur in the whole box.
        radius : float
            GCMC sphere radius in nm.
        is_pme : bool
            Use PME for electrostatics. If False, use reaction field.
        excess_chemical_potential : float
            Excess chemical potential in kJ/mol.
        standard_volume : float
            Standard molar volume of the molecule in nm^3.
        temperature : float
            Temperature in K.
        adams_shift : float
            Shift applied to the Adams parameter.
        num_ghost_waters : int
            Number of ghost molecules to pre-allocate.
        batch_size : int
            Number of trial moves per GPU batch.
        num_attempts : int
            Total attempts per move() call.
        num_threads : int
            GPU threads per block (multiple of 32).
        bulk_sampling_probability : float
            Probability of sampling in the full box instead of the sphere.
        device : int, optional
            GPU device index.
        platform : str
            "auto", "cuda", or "opencl".
        tolerance : float
            Minimum acceptance probability threshold.
        """

        # Validate system input.
        if not isinstance(system, _MstkSystem):
            raise TypeError("'system' must be of type 'mstk.simsys.System'")
        self._system = system
        self._topology = system.topology
        self._ff = system.ff

        # Detect combining rule from force field.
        if self._ff.lj_mixing_rule == _ForceField.LJ_MIXING_LB:
            self._combining_rule = 0  # arithmetic sigma
        elif self._ff.lj_mixing_rule == _ForceField.LJ_MIXING_GEOMETRIC:
            self._combining_rule = 1  # geometric sigma
        else:
            raise ValueError(
                "Unsupported LJ mixing rule. Must be Lorentz-Berthelot or geometric."
            )

        # Guard: check FF compatibility with GCMC kernel assumptions.
        unsupported_vdw = self._ff.vdw_term_classes - {_LJ126Term}
        if unsupported_vdw:
            raise ValueError(
                f"Unsupported VdW term classes: "
                f"{', '.join(c.__name__ for c in unsupported_vdw)}. "
                f"GCMC kernel only supports LJ126Term."
            )
        if self._ff.pairwise_vdw_terms:
            raise ValueError(
                "Force field contains explicit pairwise VdW terms. "
                "GCMC kernel uses per-atom sigma/epsilon with mixing rules "
                "and cannot reproduce explicit pair parameters."
            )
        if self._ff.vdw_long_range == _ForceField.VDW_LONGRANGE_SHIFT:
            raise ValueError(
                "vdw_long_range='shift' is not supported. "
                "GCMC kernel uses unshifted LJ potential."
            )
        if self._ff.is_polarizable:
            raise ValueError(
                "Polarizable force fields (Drude) are not supported. "
                "GCMC kernel computes fixed-charge pairwise energy only."
            )
        if self._ff.has_virtual_site:
            raise ValueError(
                "Virtual site force fields (e.g. TIP4P) are not supported."
            )

        self._residue_name = residue_name

        # Reference atoms for GCMC sphere.
        if reference is not None:
            if not isinstance(reference, (list, _np.ndarray)):
                raise TypeError("'reference' must be a list of int")
            self._reference_indices = _np.asarray(reference, dtype=_np.int32)
        else:
            self._reference_indices = None

        self._is_pme = bool(is_pme)

        # Store physical parameters (all in nm / kJ/mol / K).
        self._radius = float(radius)
        self._cutoff = float(self._ff.vdw_cutoff)
        self._excess_chemical_potential = float(excess_chemical_potential)
        self._standard_volume = float(standard_volume)
        self._temperature = float(temperature)
        self._adams_shift = float(adams_shift)

        # Validate integer parameters.
        if not isinstance(num_ghost_waters, int) or num_ghost_waters <= 0:
            raise ValueError("'num_ghost_waters' must be a positive int")
        self._num_ghost_waters = num_ghost_waters

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("'batch_size' must be a positive int")
        self._batch_size = batch_size

        if not isinstance(num_attempts, int) or num_attempts <= 0:
            raise ValueError("'num_attempts' must be a positive int")
        if num_attempts < batch_size:
            raise ValueError("'num_attempts' must be >= 'batch_size'")
        self._num_attempts = num_attempts

        if not isinstance(num_threads, int) or num_threads % 32 != 0:
            raise ValueError("'num_threads' must be a positive multiple of 32")
        self._num_threads = num_threads

        self._bulk_sampling_probability = float(bulk_sampling_probability)
        if not 0.0 <= self._bulk_sampling_probability <= 1.0:
            raise ValueError("'bulk_sampling_probability' must be between 0 and 1")

        self._insert_only = bool(insert_only)
        self._ghost_existing = bool(ghost_existing)

        self._tolerance = float(tolerance)

        # Seed.
        if seed is None:
            seed = _np.random.randint(_np.iinfo(_np.int32).max)
        self._seed = seed
        _np.random.seed(seed)
        self._rng = _np.random.default_rng(seed)

        # NVCC path.
        if nvcc is not None:
            if not _os.path.exists(nvcc):
                raise ValueError(f"'nvcc' path does not exist: {nvcc}")
        else:
            from shutil import which
            nvcc = _os.environ.get("PYCUDA_NVCC", which("nvcc"))
        self._nvcc = nvcc

        self._compiler_optimisations = compiler_optimisations

        # Attributes set by _prepare_system().
        self._num_points = 0
        self._water_charge = None
        self._water_sigma = None
        self._water_epsilon = None
        self._template_type_indices = None
        self._use_lrc = False
        self._lrc_w_solute = 0.0
        self._lrc_ww_half = 0.0
        self._nonbonded_force = None
        self._custom_nb_forces = []
        self._ghost_type_index = None
        self._water_indices = None
        self._num_waters = 0
        self._num_atoms = 0
        self._extended_system = None
        self.omm_system = None
        self.topology = None

        # --- Prepare system: find molecules, add ghosts ---
        self._prepare_system()

        # --- Platform and GPU backend ---
        valid_platforms = {"auto", "cuda", "opencl"}
        platform = platform.lower().strip()
        if platform not in valid_platforms:
            raise ValueError(f"Invalid platform '{platform}'. Must be one of {valid_platforms}.")
        self._platform = platform

        if device is not None and not isinstance(device, int):
            raise ValueError("'device' must be of type 'int'")
        self._device = device

        self._backend = _create_backend(
            platform=self._platform,
            device=self._device if self._device is not None else 0,
            num_points=self._num_points,
            num_batch=self._batch_size,
            num_waters=self._num_waters,
            num_atoms=self._num_atoms,
            num_threads=self._num_threads,
            nvcc=self._nvcc,
            compiler_optimisations=self._compiler_optimisations,
        )

        self._kernels = self._backend.compile_kernels()
        self._rng_manager = _RNGManager(self._batch_size, seed=self._seed)

        # Block dimensions.
        self._atom_blocks = self._num_atoms // self._num_threads + 1
        self._batch_blocks = self._batch_size // self._num_threads + 1
        self._water_blocks = self._num_waters // self._num_threads + 1

        # Attributes set by _initialise_gpu_memory().
        self._gpu_charge = None
        self._gpu_sigma = None
        self._gpu_epsilon = None
        self._gpu_charge_water = None
        self._gpu_sigma_water = None
        self._gpu_epsilon_water = None
        self._water_state = None
        self._gpu_is_ghost_water = None
        self._gpu_water_idx = None
        self._gpu_water_state = None
        self._rf_cutoff = None
        self._rf_kappa = None
        self._rf_correction = None
        self._gpu_position = None
        self._water_positions = None
        self._energy_coul = None
        self._energy_lj = None
        self._accepted = None
        self._energy_change = None
        self._probability = None
        self._deletion_candidates = None

        # --- Initialise GPU memory ---
        self._initialise_gpu_memory()

        # --- Box information ---
        box_size = self._topology.cell.get_size()
        self.set_box(box_size=box_size)

        # --- Constants ---
        # beta in mol/kJ
        self._beta = 1.0 / (_KB_KJMOL * self._temperature)
        # beta for OpenMM energy (kJ/mol)
        self._beta_openmm = 1.0 / (
            _openmm.unit.BOLTZMANN_CONSTANT_kB
            * _openmm.unit.AVOGADRO_CONSTANT_NA
            * self._temperature
            * _openmm.unit.kelvin
        )

        # Volume and Adams parameter.
        box_volume = box_size[0] * box_size[1] * box_size[2]  # nm^3
        sphere_volume = (4.0 * _np.pi * self._radius ** 3) / 3.0  # nm^3

        B_sphere = (
            self._beta * self._excess_chemical_potential
            + _np.log(sphere_volume / self._standard_volume)
        ) + self._adams_shift

        B_bulk = (
            self._beta * self._excess_chemical_potential
            + _np.log(box_volume / self._standard_volume)
        ) + self._adams_shift

        if self._reference_indices is not None:
            _logger.info(f"Adams B_sphere = {B_sphere:.6f}, B_bulk = {B_bulk:.6f}")
        else:
            _logger.info(f"Adams B = {B_bulk:.6f}")

        self._exp_B_sphere = _np.exp(B_sphere)
        self._exp_minus_B_sphere = _np.exp(-B_sphere)
        self._exp_B_bulk = _np.exp(B_bulk)
        self._exp_minus_B_bulk = _np.exp(-B_bulk)

        # Coulomb prefactor: 1/(4*pi*eps0) in Angstrom units = 332.0637 kcal*A/(mol*e^2)
        # This is already embedded in the kernel constant `prefactor`.

        # Zero counters.
        self._N = 0
        self._num_moves = 0
        self._num_accepted = 0
        self._num_accepted_attempts = 0
        self._num_insertions = 0
        self._num_deletions = 0
        self._ghost_exhausted = False

        # Bulk sampling flag.
        self._is_bulk = False
        self._openmm_context = None

        import atexit
        atexit.register(self._cleanup)

        # Pre-allocate zero target array for bulk sampling.
        self._zero_target_gpu = self._backend.to_gpu(_np.zeros(3, dtype=_np.float32))

    def _cleanup(self) -> None:
        try:
            self._rng_manager.shutdown()
        except Exception:
            pass
        try:
            self._backend.cleanup()
        except Exception:
            pass


    # --- Properties ---

    @property
    def water_state(self) -> _np.ndarray:
        return self._water_state.copy()

    @property
    def water_indices(self) -> _np.ndarray:
        return self._water_indices

    @property
    def box_size(self) -> _np.ndarray:
        return self._box_size.copy()

    @property
    def num_insertions(self) -> int:
        return self._num_insertions

    @property
    def num_deletions(self) -> int:
        return self._num_deletions

    @property
    def ghost_exhausted(self) -> bool:
        return self._ghost_exhausted

    # --- Public methods ---

    def set_box(self, box_size=None, context=None):
        """
        Set box dimensions.

        Parameters
        ----------
        box_size : array-like of float, optional
            [Lx, Ly, Lz] in nm.
        context : openmm.Context, optional
            Extract box from context state.
        """
        if context is not None:
            box = context.getState().getPeriodicBoxVectors(asNumpy=True) / _openmm.unit.nanometer
            box_size = _np.array([box[0][0], box[1][1], box[2][2]])

        box_size = _np.asarray(box_size, dtype=_np.float64)
        self._box_size = box_size  # nm

        # GPU kernel uses Angstrom.
        box_ang = box_size * 10.0
        cell_matrix = _np.diag(box_ang).flatten().astype(_np.float32)
        cell_matrix_inv = _np.diag(1.0 / box_ang).flatten().astype(_np.float32)
        M = _np.diag(box_ang ** 2).flatten().astype(_np.float32)

        self._gpu_cell_matrix = self._backend.to_gpu(cell_matrix)
        self._gpu_cell_matrix_inverse = self._backend.to_gpu(cell_matrix_inv)
        self._gpu_M = self._backend.to_gpu(M)


    def num_waters(self, context=None) -> int:
        """Return the number of real (non-ghost) molecules in the GCMC region."""
        if self._reference_indices is None:
            return int(_np.sum(self._water_state == 1))

        # After a bulk move, _N reflects box count not sphere count — recompute.
        if context is None and self._is_bulk:
            context = self._openmm_context

        if context is not None:
            # Recompute _N by running deletion kernel on current positions.
            state = context.getState(getPositions=True)
            positions = state.getPositions(asNumpy=True) / _openmm.unit.angstrom

            target = self._backend.to_gpu(
                self._get_target_position(positions).astype(_np.float32)
            )
            self._gpu_position = self._backend.to_gpu(_as_float32(positions).flatten())

            self._kernels["deletion"](
                _np.int32(self._num_waters),
                self._deletion_candidates,
                target,
                _np.float32(self._radius * 10.0),
                self._gpu_position,
                self._gpu_water_idx,
                self._gpu_water_state,
                self._gpu_cell_matrix_inverse,
                self._gpu_M,
                block=(self._num_threads, 1, 1),
                grid=(self._water_blocks, 1, 1),
            )

            candidates = self._backend.from_gpu(self._deletion_candidates).flatten()
            self._N = int(_np.sum(candidates == 1))
            self._is_bulk = False

        # If last move was sphere-targeted, _N is already up to date.
        return self._N


    def move_acceptance_probability(self) -> float:
        total_attempts = self._num_moves * self._num_attempts
        if total_attempts == 0:
            return 0.0
        return self._num_accepted / total_attempts


    # --- Main GCMC move ---

    def move(self, context: _openmm.Context) -> list:
        """
        Perform num_attempts trial insertion/deletion moves.

        Parameters
        ----------
        context : openmm.Context
            The OpenMM context.

        Returns
        -------
        moves : list of int
            Accepted moves (0=insertion, 1=deletion).
        """
        self._num_moves += 1

        num_attempts = 0
        num_batches = 1
        is_accepted = False
        moves = []

        # Decide bulk vs sphere sampling.
        self._is_bulk = True
        if self._reference_indices is not None:
            if self._rng.random() > self._bulk_sampling_probability:
                self._is_bulk = False

        while num_attempts < self._num_attempts:
            _logger.debug(f"Batch {num_batches}, attempts {num_attempts}/{self._num_attempts}")

            if num_batches == 1 or is_accepted:
                if num_batches == 1:
                    state = context.getState(getPositions=True, getEnergy=self._is_pme)
                    positions_openmm = state.getPositions(asNumpy=True)
                    positions_angstrom = positions_openmm / _openmm.unit.angstrom

                    if self._is_pme:
                        initial_energy = state.getPotentialEnergy()
                    else:
                        initial_energy = None

                    self.set_box(context=context)
                    box_volume = self._box_size[0] * self._box_size[1] * self._box_size[2]
                    B_bulk = (
                        self._beta * self._excess_chemical_potential
                        + _np.log(box_volume / self._standard_volume)
                    ) + self._adams_shift
                    self._exp_B_bulk = _np.exp(B_bulk)
                    self._exp_minus_B_bulk = _np.exp(-B_bulk)
                    v_nm3 = box_volume

                    if self._reference_indices is not None and not self._is_bulk:
                        target = self._get_target_position(positions_angstrom).astype(_np.float32)

                    self._gpu_position = self._backend.to_gpu(
                        _as_float32(positions_angstrom).flatten()
                    )

                # Find deletion candidates.
                if not self._is_bulk:
                    self._kernels["deletion"](
                        _np.int32(self._num_waters),
                        self._deletion_candidates,
                        self._backend.to_gpu(_as_float32(target)),
                        _np.float32(self._radius * 10.0),  # nm → Å
                        self._gpu_position,
                        self._gpu_water_idx,
                        self._gpu_water_state,
                        self._gpu_cell_matrix_inverse,
                        self._gpu_M,
                        block=(self._num_threads, 1, 1),
                        grid=(self._water_blocks, 1, 1),
                    )
                    deletion_candidates = self._backend.from_gpu(
                        self._deletion_candidates
                    ).flatten()
                    deletion_candidates = _np.where(deletion_candidates == 1)[0]
                else:
                    _logger.debug("Sampling within the entire simulation box")
                    deletion_candidates = self._get_non_ghost_waters()
                    target = None

                ghost_waters = self._get_ghost_waters()
                if len(ghost_waters) == 0:
                    _logger.error("Ghost molecules exhausted")
                    self._ghost_exhausted = True
                    return moves

                idx_water = self._rng.choice(ghost_waters)

                start_idx = self._water_indices[idx_water]
                template_positions = self._backend.to_gpu(
                    _as_float32(
                        positions_angstrom[start_idx: start_idx + self._num_points]
                    ).flatten()
                )

                self._N = len(deletion_candidates)

            is_accepted = False
            move = None

            _logger.debug(f"N in sampling volume: {self._N}")

            if self._insert_only or len(deletion_candidates) == 0:
                candidates = _np.zeros(self._batch_size, dtype=_np.int32)
                candidates_gpu = self._backend.to_gpu(candidates)
                is_deletion = _np.zeros(self._batch_size, dtype=_np.int32)
                is_deletion_gpu = self._backend.to_gpu(is_deletion)
            else:
                candidates = self._rng.choice(deletion_candidates, size=self._batch_size)
                candidates_gpu = self._backend.to_gpu(_as_int32(candidates))
                is_deletion = self._rng.choice(2, size=self._batch_size)
                is_deletion_gpu = self._backend.to_gpu(_as_int32(is_deletion))

            if target is None:
                target_gpu = self._zero_target_gpu
                is_target = _np.int32(0)
                exp_B = self._exp_B_bulk
                exp_minus_B = self._exp_minus_B_bulk
            else:
                target_gpu = self._backend.to_gpu(_as_float32(target))
                is_target = _np.int32(1)
                exp_B = self._exp_B_sphere
                exp_minus_B = self._exp_minus_B_sphere

            batch_randoms = self._rng_manager.get_batch_randoms()
            randoms_rotation = self._backend.to_gpu(batch_randoms.rotation)
            if is_target:
                randoms_position = self._backend.to_gpu(batch_randoms.direction)
            else:
                randoms_position = self._backend.to_gpu(batch_randoms.position)
            randoms_radius = self._backend.to_gpu(batch_randoms.radius)

            # Generate random positions/orientations.
            self._kernels["water"](
                _np.int32(self._num_points),
                _np.int32(self._batch_size),
                template_positions,
                target_gpu,
                _np.float32(self._radius * 10.0),  # nm → Å
                self._water_positions,
                is_target,
                randoms_rotation,
                randoms_position,
                randoms_radius,
                self._gpu_cell_matrix,
                block=(self._num_threads, 1, 1),
                grid=(self._batch_blocks, 1, 1),
            )

            # Compute energy.
            self._kernels["energy"](
                _np.int32(self._num_points),
                _np.int32(self._batch_size),
                _np.int32(self._num_atoms),
                self._water_positions,
                self._energy_coul,
                self._energy_lj,
                candidates_gpu,
                is_deletion_gpu,
                self._gpu_position,
                self._gpu_charge,
                self._gpu_sigma,
                self._gpu_epsilon,
                self._gpu_is_ghost_water,
                self._gpu_sigma_water,
                self._gpu_epsilon_water,
                self._gpu_charge_water,
                self._gpu_water_idx,
                self._gpu_cell_matrix_inverse,
                self._gpu_M,
                self._rf_cutoff,
                self._rf_kappa,
                self._rf_correction,
                _np.int32(self._combining_rule),
                block=(self._num_threads, 1, 1),
                grid=(self._atom_blocks, self._batch_size, 1),
            )

            randoms_acceptance = self._backend.to_gpu(batch_randoms.acceptance)

            # Check acceptance.
            self._kernels["acceptance"](
                _np.int32(self._batch_size),
                _np.int32(self._num_atoms),
                _np.int32(self._N),
                _np.float32(exp_B),
                _np.float32(exp_minus_B),
                _np.float32(self._beta * 4.184),  # mol/kJ → mol/kcal for kernel
                is_deletion_gpu,
                self._energy_coul,
                self._energy_lj,
                self._energy_change,
                self._probability,
                self._accepted,
                _np.float32(self._tolerance),
                randoms_acceptance,
                block=(self._num_threads, 1, 1),
                grid=(self._batch_blocks, 1, 1),
            )

            accepted = _np.where(self._backend.from_gpu(self._accepted).flatten() == 1)[0]
            num_accepted_attempts = len(accepted)
            self._num_accepted_attempts += num_accepted_attempts

            num_attempts += self._batch_size

            if num_accepted_attempts == 0:
                num_batches += 1
                continue

            if self._is_pme:
                max_accepted = num_accepted_attempts
                energy_changes = self._backend.from_gpu(self._energy_change).flatten()
            else:
                max_accepted = 1
                energy_changes = None

            for i in range(max_accepted):
                idx = accepted[i]

                # Insertion.
                if is_deletion[idx] == 0:
                    if self._use_lrc:
                        n_w_before = float(self._N)

                    self._accept_insertion(
                        idx, idx_water, positions_openmm, positions_angstrom, context
                    )
                    self._num_accepted += 1
                    self._num_insertions += 1
                    is_accepted = True
                    move = 0

                    if self._is_pme:
                        dE_RF = energy_changes[idx] * _openmm.unit.kilocalories_per_mole
                        final_energy = context.getState(getEnergy=True).getPotentialEnergy()

                        acc_prob = _np.exp(
                            min(0.0, -self._beta_openmm * (final_energy - initial_energy - dE_RF))
                        )

                        if acc_prob < self._rng.random():
                            self._accept_deletion(idx_water, context)
                            self._num_accepted -= 1
                            self._num_insertions -= 1
                            is_accepted = False
                            move = None

                    elif self._use_lrc:
                        dLRC = (
                            (self._lrc_w_solute + 2.0 * n_w_before * self._lrc_ww_half)
                            / v_nm3
                        )
                        acc_prob = _np.exp(min(0.0, -self._beta * dLRC))
                        if acc_prob < self._rng.random():
                            self._accept_deletion(idx_water, context)
                            self._num_accepted -= 1
                            self._num_insertions -= 1
                            is_accepted = False
                            move = None

                    if is_accepted:
                        _logger.debug(f"Accepted insertion: water={idx_water}")
                        break

                # Deletion.
                else:
                    if self._use_lrc:
                        n_w_before = float(self._N)

                    self._accept_deletion(candidates[idx], context)
                    self._num_accepted += 1
                    self._num_deletions += 1
                    is_accepted = True
                    move = 1

                    if self._is_pme:
                        dE_RF = energy_changes[idx] * _openmm.unit.kilocalories_per_mole
                        final_energy = context.getState(getEnergy=True).getPotentialEnergy()

                        acc_prob = _np.exp(
                            min(0.0, -self._beta_openmm * (final_energy - initial_energy - dE_RF))
                        )

                        if acc_prob < self._rng.random():
                            self._reject_deletion(candidates[idx], context)
                            self._num_accepted -= 1
                            self._num_deletions -= 1
                            is_accepted = False
                            move = None

                    elif self._use_lrc:
                        dLRC = (
                            -(self._lrc_w_solute + 2.0 * (n_w_before - 1.0) * self._lrc_ww_half)
                            / v_nm3
                        )
                        acc_prob = _np.exp(min(0.0, -self._beta * dLRC))
                        if acc_prob < self._rng.random():
                            self._reject_deletion(candidates[idx], context)
                            self._num_accepted -= 1
                            self._num_deletions -= 1
                            is_accepted = False
                            move = None

                    if is_accepted:
                        _logger.debug(f"Accepted deletion: water={candidates[idx]}")
                        break

            if is_accepted:
                moves.append(move)
                if self._is_pme:
                    initial_energy = final_energy

            num_batches += 1

        if self._reference_indices is not None and self._is_bulk:
            self._openmm_context = context

        return moves

    # --- Private methods ---

    def _get_ghost_waters(self) -> _np.ndarray:
        return _np.where(self._water_state == 0)[0]

    def _get_non_ghost_waters(self) -> _np.ndarray:
        return _np.where(self._water_state != 0)[0]

    def _get_target_position(self, positions_ang):
        """Compute GCMC sphere center using minimum-image convention (rectangular)."""
        ref = positions_ang[self._reference_indices]
        box_ang = self._box_size * 10.0
        delta = ref - ref[0]
        delta -= _np.round(delta / box_ang) * box_ang
        center = ref[0] + delta.mean(axis=0)
        return center.astype(_np.float32)

    def _prepare_system(self):
        """
        Find target molecules, build extended topology with ghost molecules,
        export to OpenMM (generating all bonded forces), then post-process
        nonbonded parameters to create the ghost/real toggle mechanism.
        """
        topology = self._topology

        # Find existing target molecules by residue name.
        target_residues = [r for r in topology.residues if r.name == self._residue_name]
        if len(target_residues) == 0:
            avail = sorted(set(r.name for r in topology.residues))
            raise ValueError(
                f"No residues with name '{self._residue_name}' found in system. "
                f"Available: {avail}. For dry systems, add a template molecule "
                f"to the topology and use ghost_existing=True."
            )

        template_mol = target_residues[0].atoms[0].molecule
        template_atoms = target_residues[0].atoms

        self._num_points = len(template_atoms)

        # Extract template properties.
        template_positions_nm = _np.array([a.position for a in template_atoms])
        template_charges = _np.array([a.charge for a in template_atoms])
        template_sigmas = _np.zeros(self._num_points)
        template_epsilons = _np.zeros(self._num_points)
        for i, atom in enumerate(template_atoms):
            vdw = self._system.atom_vdw_terms[atom]
            template_sigmas[i] = vdw.sigma  # nm
            template_epsilons[i] = vdw.epsilon  # kJ/mol

        # Store template params in kernel units (Å, kcal/mol) for GPU.
        self._water_charge = template_charges  # elementary charge
        self._water_sigma = template_sigmas * 10.0  # Å
        self._water_epsilon = template_epsilons / 4.184  # kcal/mol

        # --- Build extended topology with ghost molecules ---
        n_existing_ghosts = len(target_residues) if self._ghost_existing else 0
        n_extra_ghosts = self._num_ghost_waters - n_existing_ghosts
        if n_extra_ghosts < 0:
            raise ValueError(
                f"num_ghost_waters ({self._num_ghost_waters}) must be >= "
                f"existing target molecules ({n_existing_ghosts}) when ghost_existing=True"
            )
        extended_top = _MstkTopology(
            topology.molecules + [template_mol] * n_extra_ghosts
        )
        extended_top.cell = topology.cell

        # Create new mstk System from extended topology (generates all bonded forces).
        extended_system = _MstkSystem(extended_top, self._ff)
        self.omm_system = extended_system.to_omm_system()

        # --- Identify target residues in extended topology ---
        # All GCMC molecules (real + ghost) in extended topology.
        all_gcmc_residues = [
            r for r in extended_top.residues if r.name == self._residue_name
        ]
        n_real_mols = 0 if self._ghost_existing else len(target_residues)

        # Extract template type indices from the CustomNonbondedForce.
        template_atom_ids = [a.id for a in all_gcmc_residues[0].atoms]
        self._template_type_indices = None
        for force in self.omm_system.getForces():
            if isinstance(force, _openmm.CustomNonbondedForce):
                self._template_type_indices = _np.array(
                    [int(force.getParticleParameters(aid)[0]) for aid in template_atom_ids],
                    dtype=_np.int32,
                )
                break

        # Set NonbondedForce method.
        for force in self.omm_system.getForces():
            if isinstance(force, _openmm.NonbondedForce):
                if self._is_pme:
                    force.setNonbondedMethod(_openmm.NonbondedForce.PME)
                else:
                    force.setNonbondedMethod(_openmm.NonbondedForce.CutoffPeriodic)

        # Patch CustomNonbondedForce expression for NaN prevention with ghost overlaps.
        self._use_lrc = False
        for force in self.omm_system.getForces():
            if isinstance(force, _openmm.CustomNonbondedForce):
                if force.getUseLongRangeCorrection():
                    self._use_lrc = True
                expr = force.getEnergyFunction()
                if 'invR6*invR6' in expr and '1/r^6' in expr:
                    patched = ('select(A(type1,type2)+B(type1,type2),'
                              'A(type1,type2)*invR6*invR6-B(type1,type2)*invR6,0);'
                              'invR6=1/r^6')
                    force.setEnergyFunction(patched)

        # PME: LRC is included in the OpenMM full-energy correction step.
        if self._use_lrc and not self._is_pme:
            self._compute_gcmc_lrc(target_residues, topology)
        else:
            self._lrc_w_solute = 0.0
            self._lrc_ww_half = 0.0

        # --- Find NonbondedForce and CustomNonbondedForce ---
        self._nonbonded_force = None
        self._custom_nb_forces = []
        for force in self.omm_system.getForces():
            if isinstance(force, _openmm.NonbondedForce):
                self._nonbonded_force = force
            elif isinstance(force, _openmm.CustomNonbondedForce):
                self._custom_nb_forces.append(force)

        if self._nonbonded_force is None:
            raise ValueError("No NonbondedForce found in the OpenMM system")
        nonbonded = self._nonbonded_force

        # --- Enlarge Discrete2DFunction tables to add ghost type ---
        self._ghost_type_index = None
        for cnb in self._custom_nb_forces:
            for func_idx in range(cnb.getNumTabulatedFunctions()):
                func = cnb.getTabulatedFunction(func_idx)
                if isinstance(func, _openmm.Discrete2DFunction):
                    xsize, ysize, _ = func.getFunctionParameters()
                    self._ghost_type_index = xsize
                    break
            if self._ghost_type_index is not None:
                break

        if self._ghost_type_index is None:
            raise ValueError("No Discrete2DFunction found in CustomNonbondedForce")

        n_new = self._ghost_type_index + 1
        for cnb in self._custom_nb_forces:
            for func_idx in range(cnb.getNumTabulatedFunctions()):
                func = cnb.getTabulatedFunction(func_idx)
                if isinstance(func, _openmm.Discrete2DFunction):
                    xsize, ysize, old_values = func.getFunctionParameters()
                    new_values = [0.0] * (n_new * n_new)
                    for row in range(xsize):
                        for col in range(ysize):
                            new_values[row + n_new * col] = old_values[row + xsize * col]
                    func.setFunctionParameters(n_new, n_new, new_values)

        # --- Set ghost nonbonded parameters ---
        ghost_residues = all_gcmc_residues[n_real_mols:]
        for ghost_res in ghost_residues:
            for atom in ghost_res.atoms:
                nonbonded.setParticleParameters(atom.id, 0.0, 1.0, 0.0)
                for cnb in self._custom_nb_forces:
                    cnb.setParticleParameters(atom.id, [float(self._ghost_type_index)])

        # --- Fix beyond-1-4 pairs for ALL GCMC molecules ---
        # Makes intramolecular energy constant regardless of ghost/real state.
        dist_matrix = template_mol.get_distance_matrix(max_bond=3)
        beyond_14_local = []
        for i in range(template_mol.n_atom):
            for j in range(i + 1, template_mol.n_atom):
                if dist_matrix[i, j] == 0:
                    beyond_14_local.append((i, j))

        if beyond_14_local:
            _logger.info(
                f"Beyond-1-4 pairs per molecule: {len(beyond_14_local)}"
            )
            # Create CustomBondForce for beyond-1-4 LJ (same expression as mstk 1-4).
            beyond_14_lj = _openmm.CustomBondForce(
                'C*epsilon*((sigma/r)^n-(sigma/r)^m);'
                'C=n/(n-m)*(n/m)^(m/(n-m))'
            )
            beyond_14_lj.addPerBondParameter('epsilon')
            beyond_14_lj.addPerBondParameter('sigma')
            beyond_14_lj.addPerBondParameter('n')
            beyond_14_lj.addPerBondParameter('m')
            beyond_14_lj.setUsesPeriodicBoundaryConditions(True)
            beyond_14_lj.setName('Beyond14LJ')

            for res in all_gcmc_residues:
                for (li, lj) in beyond_14_local:
                    # Get VdW parameters for this pair from the force field.
                    atom_i = res.atoms[li]
                    atom_j = res.atoms[lj]
                    ai = atom_i.id
                    aj = atom_j.id
                    vdw = self._ff.get_vdw_term(
                        self._ff.atom_types[atom_i.type],
                        self._ff.atom_types[atom_j.type]
                    )
                    beyond_14_lj.addBond(
                        ai, aj, [vdw.epsilon, vdw.sigma, 12, 6]
                    )
                    # Add as exception in NonbondedForce (constant chargeProd).
                    chg_prod = atom_i.charge * atom_j.charge
                    nonbonded.addException(ai, aj, chg_prod, 1.0, 0.0)
                    # Add as exclusion in CustomNonbondedForce.
                    for cnb in self._custom_nb_forces:
                        cnb.addExclusion(ai, aj)

            self.omm_system.addForce(beyond_14_lj)

        # --- Spread ghost molecule positions to avoid numerical overlap ---
        box_size = _np.array(topology.cell.get_size())
        rng = _np.random.default_rng(self._seed)
        for ghost_res in ghost_residues:
            origin = rng.random(3) * box_size
            for atom in ghost_res.atoms:
                extended_top.atoms[atom.id].position = origin + template_positions_nm[atom.id_in_mol]

        # --- Build water_indices and positions ---
        water_indices = []
        for res in all_gcmc_residues:
            water_indices.append(res.atoms[0].id)

        self._water_indices = _np.array(water_indices, dtype=_np.int32)
        self._num_waters = len(self._water_indices)
        self._num_atoms = self.omm_system.getNumParticles()

        # Store extended system reference for _initialise_gpu_memory.
        self._extended_system = extended_system
        self.topology = extended_top



    def _initialise_gpu_memory(self):
        """Upload per-atom parameters and allocate GPU buffers."""
        n_total = self._num_atoms

        # Per-atom arrays.
        charges = _np.zeros(n_total, dtype=_np.float32)
        sigmas = _np.zeros(n_total, dtype=_np.float32)
        epsilons = _np.zeros(n_total, dtype=_np.float32)

        # Fill all atoms (real + ghost) from extended system.
        for atom in self.topology.atoms:
            charges[atom.id] = atom.charge
            vdw = self._extended_system.atom_vdw_terms[atom]
            sigmas[atom.id] = vdw.sigma * 10.0  # nm → Å
            epsilons[atom.id] = vdw.epsilon / 4.184  # kJ/mol → kcal/mol

        # Zero ghost atoms' charge and epsilon (sigma kept for insertion).
        n_real_waters = self._num_waters - self._num_ghost_waters
        for i in range(self._num_ghost_waters):
            start = self._water_indices[n_real_waters + i]
            for j in range(self._num_points):
                charges[start + j] = 0.0
                epsilons[start + j] = 0.0

        # Upload to GPU.
        self._gpu_charge = self._backend.to_gpu(charges)
        self._gpu_sigma = self._backend.to_gpu(sigmas)
        self._gpu_epsilon = self._backend.to_gpu(epsilons)

        # Water template parameters (in Å and kcal/mol).
        self._gpu_charge_water = self._backend.to_gpu(
            self._water_charge.astype(_np.float32)
        )
        self._gpu_sigma_water = self._backend.to_gpu(
            self._water_sigma.astype(_np.float32)
        )
        self._gpu_epsilon_water = self._backend.to_gpu(
            self._water_epsilon.astype(_np.float32)
        )

        # Water state: 0=ghost, 1=real.
        water_state = _np.ones(self._num_waters, dtype=_np.int32)
        is_ghost_water = _np.zeros(n_total, dtype=_np.int32)
        for i in range(self._num_ghost_waters):
            idx = n_real_waters + i
            water_state[idx] = 0
            start = self._water_indices[idx]
            for j in range(self._num_points):
                is_ghost_water[start + j] = 1

        self._water_state = water_state

        self._gpu_is_ghost_water = self._backend.to_gpu(is_ghost_water.astype(_np.int32))
        self._gpu_water_idx = self._backend.to_gpu(self._water_indices.astype(_np.int32))
        self._gpu_water_state = self._backend.to_gpu(self._water_state.astype(_np.int32))

        # Reaction field parameters (in Å).
        cutoff_ang = self._cutoff * 10.0
        self._rf_cutoff = _np.float32(cutoff_ang)
        self._rf_kappa = _np.float32(
            (78.3 - 1.0) / ((2.0 * 78.3 + 1.0) * cutoff_ang ** 3)
        )
        self._rf_correction = _np.float32(
            1.0 / cutoff_ang + float(self._rf_kappa) * cutoff_ang ** 2
        )

        # Allocate mutable buffers.
        self._gpu_position = self._backend.empty((1, n_total * 3), _np.float32)
        self._water_positions = self._backend.empty(
            (1, self._batch_size * 3 * self._num_points), _np.float32
        )
        self._energy_coul = self._backend.empty(
            (1, self._batch_size * n_total), _np.float32
        )
        self._energy_lj = self._backend.empty(
            (1, self._batch_size * n_total), _np.float32
        )
        self._accepted = self._backend.empty((1, self._batch_size), _np.int32)
        self._energy_change = self._backend.empty((1, self._batch_size), _np.float32)
        self._probability = self._backend.empty((1, self._batch_size), _np.float32)
        self._deletion_candidates = self._backend.empty((1, self._num_waters), _np.int32)


    def _compute_gcmc_lrc(self, target_residues, topology):
        """
        Precompute GCMC LRC coefficients from atom VdW parameters.

        Computes lrc_w_solute (interaction of one water molecule with all solute
        atoms) and lrc_ww_half (half the interaction between a pair of water
        molecules). Units: kJ/mol*nm^3 (divide by V in nm^3 to get energy).

        The standard LJ tail correction for pair (i,j) is:
          LRC_ij = (8/3)*pi*eps_ij * [sigma_ij^12/(9*rc^9) - sigma_ij^6/(3*rc^3)]
        which equals:
          LRC_ij = (8/3)*pi*eps_ij*sigma_ij^6 * [sigma_ij^6/(9*rc^9) - 1/(3*rc^3)]
        """
        rc = self._cutoff  # nm
        rc3 = rc ** 3
        rc9 = rc3 ** 3

        # Build water atom classes from template (works for both wet and dry systems).
        # Template params are stored in Å / kcal/mol; convert back to nm / kJ/mol.
        water_class_counts = {}
        for i in range(self._num_points):
            sig = self._water_sigma[i] / 10.0  # Å → nm
            eps = self._water_epsilon[i] * 4.184  # kcal/mol → kJ/mol
            if eps > 0:
                key = (sig, eps)
                water_class_counts[key] = water_class_counts.get(key, 0) + 1

        # Identify water atom indices in the topology (for solute classification).
        water_atom_ids = set()
        for res in target_residues:
            for atom in res.atoms:
                water_atom_ids.add(atom.id)

        n_water_mols = max(len(target_residues), 1)
        _logger.info(f"GCMC LRC: {n_water_mols} water mols, {sum(water_class_counts.values())} water atom types")

        # Collect (sigma, epsilon) per atom from mstk — solute only.
        solute_class_counts = {}
        n_solute_atoms = 0
        for atom in topology.atoms:
            if atom.id in water_atom_ids:
                continue
            vdw = self._system.atom_vdw_terms.get(atom)
            if vdw is not None and vdw.epsilon > 0:
                key = (vdw.sigma, vdw.epsilon)
                solute_class_counts[key] = solute_class_counts.get(key, 0) + 1
                n_solute_atoms += 1
        _logger.info(f"GCMC LRC: {n_solute_atoms} solute atoms with eps>0")

        # Combining rule: 0 = arithmetic sigma, 1 = geometric sigma.
        # Epsilon always uses geometric mean.
        def combine(sig_i, eps_i, sig_j, eps_j):
            if self._combining_rule == 0:
                sig_ij = 0.5 * (sig_i + sig_j)
            else:
                sig_ij = (sig_i * sig_j) ** 0.5
            eps_ij = (eps_i * eps_j) ** 0.5
            return sig_ij, eps_ij

        def lrc_pair(sig_ij, eps_ij):
            sig6 = sig_ij ** 6
            return 16.0 * _np.pi * eps_ij * sig6 * (sig6 / (9.0 * rc9) - 1.0 / (3.0 * rc3))

        # lrc_ww_half: half the LRC for one water-molecule pair.
        # water_class_counts is per-molecule (from template).
        water_classes = list(water_class_counts.items())
        lrc_ww = 0.0
        for i, ((sig_i, eps_i), n_i) in enumerate(water_classes):
            for j in range(i, len(water_classes)):
                (sig_j, eps_j), n_j = water_classes[j]
                sig_ij, eps_ij = combine(sig_i, eps_i, sig_j, eps_j)
                pair_lrc = lrc_pair(sig_ij, eps_ij)
                if i == j:
                    lrc_ww += n_i * n_j * pair_lrc
                else:
                    lrc_ww += 2.0 * n_i * n_j * pair_lrc
        self._lrc_ww_half = 0.5 * lrc_ww

        # lrc_w_solute: LRC of one water molecule with all solute atoms.
        lrc_w_solute = 0.0
        for (sig_w, eps_w), n_w in water_classes:
            for (sig_s, eps_s), n_s in solute_class_counts.items():
                sig_ij, eps_ij = combine(sig_w, eps_w, sig_s, eps_s)
                pair_lrc = lrc_pair(sig_ij, eps_ij)
                lrc_w_solute += n_w * n_s * pair_lrc
        self._lrc_w_solute = lrc_w_solute

        _logger.info(
            f"GCMC LRC: lrc_w_solute={self._lrc_w_solute:.6f}, "
            f"lrc_ww_half={self._lrc_ww_half:.6f}"
        )

    def _accept_insertion(self, idx, idx_water, positions_openmm, positions_angstrom, context):
        """Accept an insertion move."""
        water_positions = self._backend.from_gpu(self._water_positions).reshape(
            (self._batch_size, self._num_points, 3)
        )[idx]

        self._water_state[idx_water] = 1

        start_idx = self._water_indices[idx_water]

        for i in range(self._num_points):
            positions_openmm[start_idx + i] = _openmm.unit.Quantity(
                water_positions[i], _openmm.unit.angstrom
            )
            positions_angstrom[start_idx + i] = water_positions[i]
            # Restore charge (NonbondedForce handles Coulomb only).
            self._nonbonded_force.setParticleParameters(
                start_idx + i,
                self._water_charge[i] * _openmm.unit.elementary_charge,
                1.0 * _openmm.unit.nanometer,
                0.0 * _openmm.unit.kilojoules_per_mole,
            )
        self._nonbonded_force.updateParametersInContext(context)

        for cnb in self._custom_nb_forces:
            for i in range(self._num_points):
                cnb.setParticleParameters(
                    start_idx + i, [float(self._template_type_indices[i])]
                )
            cnb.updateParametersInContext(context)

        context.setPositions(positions_openmm)

        # Update GPU arrays.
        self._kernels["update_water"](
            _np.int32(self._num_points),
            _np.int32(idx_water),
            _np.int32(1),
            _np.int32(1),
            self._backend.to_gpu(water_positions.flatten().astype(_np.float32)),
            self._gpu_position,
            self._gpu_charge,
            self._gpu_epsilon,
            self._gpu_is_ghost_water,
            self._gpu_water_state,
            self._gpu_water_idx,
            self._gpu_charge_water,
            self._gpu_epsilon_water,
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )

        self._N += 1


    def _accept_deletion(self, idx, context):
        """Accept a deletion move."""
        self._water_state[idx] = 0

        start_idx = self._water_indices[idx]

        # Zero charge (NonbondedForce handles Coulomb only).
        for i in range(self._num_points):
            self._nonbonded_force.setParticleParameters(
                start_idx + i, 0.0, 1.0 * _openmm.unit.nanometer,
                0.0 * _openmm.unit.kilojoules_per_mole,
            )
        self._nonbonded_force.updateParametersInContext(context)

        # Set ghost type in CustomNonbondedForces (LJ).
        for cnb in self._custom_nb_forces:
            for i in range(self._num_points):
                cnb.setParticleParameters(
                    start_idx + i, [float(self._ghost_type_index)]
                )
            cnb.updateParametersInContext(context)

        self._kernels["update_water"](
            _np.int32(self._num_points),
            _np.int32(idx),
            _np.int32(0),
            _np.int32(0),
            self._backend.to_gpu(
                _np.zeros((self._num_points, 3), dtype=_np.float32).flatten()
            ),
            self._gpu_position,
            self._gpu_charge,
            self._gpu_epsilon,
            self._gpu_is_ghost_water,
            self._gpu_water_state,
            self._gpu_water_idx,
            self._gpu_charge_water,
            self._gpu_epsilon_water,
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )

        self._N -= 1


    def _reject_deletion(self, idx, context):
        """Reject a deletion move (restore parameters)."""
        self._water_state[idx] = 1

        start_idx = self._water_indices[idx]

        # Restore charge (NonbondedForce handles Coulomb only).
        for i in range(self._num_points):
            self._nonbonded_force.setParticleParameters(
                start_idx + i,
                self._water_charge[i] * _openmm.unit.elementary_charge,
                1.0 * _openmm.unit.nanometer,
                0.0 * _openmm.unit.kilojoules_per_mole,
            )
        self._nonbonded_force.updateParametersInContext(context)

        # Restore real type in CustomNonbondedForces (LJ).
        for cnb in self._custom_nb_forces:
            for i in range(self._num_points):
                cnb.setParticleParameters(
                    start_idx + i, [float(self._template_type_indices[i])]
                )
            cnb.updateParametersInContext(context)

        self._kernels["update_water"](
            _np.int32(self._num_points),
            _np.int32(idx),
            _np.int32(1),
            _np.int32(0),
            self._backend.to_gpu(
                _np.zeros((self._num_points, 3), dtype=_np.float32).flatten()
            ),
            self._gpu_position,
            self._gpu_charge,
            self._gpu_epsilon,
            self._gpu_is_ghost_water,
            self._gpu_water_state,
            self._gpu_water_idx,
            self._gpu_charge_water,
            self._gpu_epsilon_water,
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )

        self._N += 1
