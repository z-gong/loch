"""
Tests for force field compatibility guards.

Verifies that GCMCSampler rejects unsupported force field features
at construction time.
"""

import os

import pytest
from mstk.topology import Topology
from mstk.trajectory import Trajectory
from mstk.forcefield import ForceField, LJ126Term, MieTerm
from mstk.simsys import System

DATA_DIR = os.path.join(os.path.dirname(__file__), '300ethanol')


@pytest.fixture
def ethanol_system():
    top = Topology.open(os.path.join(DATA_DIR, 'top.psf'))
    frame = Trajectory.read_frame_from_file(os.path.join(DATA_DIR, 'conf.gro'), -1)
    top.cell.set_box(frame.cell.vectors)
    top.set_positions(frame.positions)
    ff = ForceField.open('primitive.zff')
    return System(top, ff)


def _sampler_kwargs():
    return dict(
        residue_name="ETOH",
        num_ghost_waters=5,
        batch_size=10,
        num_attempts=10,
        is_pme=False,
        temperature=300.0,
        excess_chemical_potential=-25.413,
        standard_volume=0.09459,
        platform="cuda",
    )


class TestGCMCGuards:
    def test_reject_mie_term(self, ethanol_system):
        from loch import GCMCSampler

        ff = ethanol_system.ff
        mie = MieTerm('fake1', 'fake1', 0.5, 0.35, 14, 7)
        ff.vdw_terms[mie.name] = mie

        with pytest.raises(ValueError, match="Unsupported VdW term classes"):
            GCMCSampler(ethanol_system, **_sampler_kwargs())

    def test_reject_pairwise_vdw(self, ethanol_system):
        from loch import GCMCSampler

        ff = ethanol_system.ff
        pair = LJ126Term('c_4o2', 'o_2', 0.5, 0.35)
        ff.pairwise_vdw_terms[pair.name] = pair

        with pytest.raises(ValueError, match="explicit pairwise VdW terms"):
            GCMCSampler(ethanol_system, **_sampler_kwargs())

    def test_reject_vdw_shift(self, ethanol_system):
        from loch import GCMCSampler

        ethanol_system.ff.vdw_long_range = 'shift'

        with pytest.raises(ValueError, match="vdw_long_range='shift'"):
            GCMCSampler(ethanol_system, **_sampler_kwargs())

    def test_reject_polarizable(self, ethanol_system):
        from loch import GCMCSampler
        from mstk.forcefield import DrudePolarTerm

        ff = ethanol_system.ff
        pterm = DrudePolarTerm('fake', alpha=0.001, thole=1.3)
        ff.polar_terms[pterm.name] = pterm

        with pytest.raises(ValueError, match="Polarizable"):
            GCMCSampler(ethanol_system, **_sampler_kwargs())

    def test_reject_virtual_site(self, ethanol_system):
        from loch import GCMCSampler
        from mstk.forcefield import TIP4PSiteTerm

        ff = ethanol_system.ff
        vterm = TIP4PSiteTerm('fake', 'fake_O', 'fake_H', d=0.015)
        ff.virtual_site_terms[vterm.name] = vterm

        with pytest.raises(ValueError, match="Virtual site"):
            GCMCSampler(ethanol_system, **_sampler_kwargs())

    def test_accept_valid_ff(self, ethanol_system):
        """Valid LJ126 + LB/geometric FF should not raise."""
        from loch import GCMCSampler

        try:
            GCMCSampler(ethanol_system, **_sampler_kwargs())
        except ValueError:
            pytest.fail("Guard rejected a valid force field")
        except Exception:
            pass  # GPU/platform errors are fine — we only test the guard
