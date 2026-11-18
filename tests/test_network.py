"""Tests for the IEEE 33-bus network model."""

import pandapower as pp
import pandapower.networks as pn
import pytest

from src.grid.network import (
    MV_BUSBAR, add_bess, add_ev_loads, add_pv_generators, build_network, run_power_flow,
)


@pytest.fixture(scope="module")
def net(config):
    return build_network(config)


def test_standard_case_matches_literature():
    """Unmodified case33bw reproduces the published Baran & Wu results."""
    ref = pn.case33bw()
    pp.runpp(ref)
    assert ref.res_line.pl_mw.sum() * 1000 == pytest.approx(202.7, abs=0.5)  # kW
    assert ref.res_bus.vm_pu.min() == pytest.approx(0.9131, abs=1e-3)
    assert ref.res_bus.vm_pu.idxmin() == 17  # IEEE bus 18


def test_network_topology(net):
    mv_buses = net.bus[net.bus.vn_kv < 20]
    assert len(mv_buses) == 33
    assert len(net.line) == 32  # radial: n_bus - 1
    assert len(net.trafo) == 1 and len(net.ext_grid) == 1
    assert net.trafo.lv_bus.iloc[0] == MV_BUSBAR
    assert net.load.p_mw.sum() == pytest.approx(3.715)
    assert set(net.load.profile) == {"residential", "commercial"}


def test_line_ratings_assigned(net, config):
    allowed = {a / 1000 for a in config["network"]["line_ratings"]["standard_ampacities_a"]}
    assert set(net.line.max_i_ka.round(6)) <= allowed


def test_baseline_power_flow_converges(net, config):
    run_power_flow(net)
    assert net.converged
    assert 0.98 <= net.res_bus.vm_pu[MV_BUSBAR] <= 1.03  # OLTC regulates the busbar
    assert net.res_line.loading_percent.max() < 100.0
    assert net.res_bus.vm_pu.idxmin() == 17


def test_der_elements_added(config):
    net = build_network(config)
    pv = add_pv_generators(net, total_capacity_mw=1.0)
    ev = add_ev_loads(net, [5, 5, 12])
    storage = add_bess(net, bus=17, capacity_kwh=500, max_power_kw=250)
    assert net.sgen.loc[pv, "sn_mva"].sum() == pytest.approx(1.0)
    assert set(ev) == {5, 12}
    assert net.storage.loc[storage, "max_e_mwh"] == pytest.approx(0.5)
    run_power_flow(net)
    assert net.converged
