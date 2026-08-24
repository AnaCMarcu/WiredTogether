"""Per-channel, credit-masked co-firing (Experiment 2) — Hebbian layer.

Pins the gated path's channel semantics added for the cofire ablations:

- ``social_events=[(i, j, "comm")]`` is exactly equivalent to the legacy
  ``comm_events=[(i, j)]``;
- with the DEFAULT credit mask ``("comm",)`` obs/imit events are invisible —
  the historical rule is reproduced bit-for-bit (the non-regression proof);
- obs/imit credit is DIRECTED (initiator only) and, critically, an imitation
  event within the interaction radius earns nonzero credit — the comm formula
  ``δ·(1 − spatial_gate)`` would have zeroed every legal replay step, since
  the act gate requires proximity (the regression the first design draft
  would have shipped);
- per-channel growth attribution sums to total realized growth;
- the new config fields default to legacy behavior (pinned HERE so
  tests/test_paper_defaults.py stays byte-identical).
"""

import numpy as np

from hebbian import HebbianConfig, HebbianSocialGraph

NEAR = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (50.0, 50.0, 50.0)]
FAR = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (200.0, 0.0, 0.0)]


def _graph(hcfg, **over):
    return HebbianSocialGraph(hcfg(**over))


# ── config defaults (legacy reproduction) ───────────────────────────────────

def test_new_config_fields_default_to_legacy_behavior():
    cfg = HebbianConfig()
    assert cfg.social_act_channels == ("comm",)
    assert cfg.social_coactivity_bonus is None  # None → δ_comm is the one δ


# ── comm equivalence ────────────────────────────────────────────────────────

def test_social_comm_event_equals_comm_event(hcfg):
    ga = _graph(hcfg)
    gb = _graph(hcfg)
    for _ in range(3):
        ga.update(FAR, comm_events=[(0, 1)],
                  bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
        gb.update(FAR, social_events=[(0, 1, "comm")],
                  bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
    np.testing.assert_array_equal(ga.W, gb.W)


def test_default_mask_ignores_obs_and_imit(hcfg):
    """Historical rule bit-for-bit: obs/imit events do not exist for it."""
    ga = _graph(hcfg)
    gb = _graph(hcfg)
    for _ in range(3):
        ga.update(NEAR, comm_events=[(0, 1)],
                  bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
        gb.update(NEAR, comm_events=[(0, 1)],
                  social_events=[(0, 2, "obs"), (1, 0, "imit")],
                  bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
    np.testing.assert_array_equal(ga.W, gb.W)


# ── directed obs credit ─────────────────────────────────────────────────────

def test_obs_grows_initiator_row_only(hcfg):
    g = _graph(hcfg, social_act_channels=("obs",))
    w0 = g.W.copy()
    g.update(FAR, social_events=[(0, 1, "obs")],
             bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
    # c_obs = δ = 0.5 ≥ ε → coeff η0 = 0.01: ΔW = 0.01·0.5·(1−0.1) = 0.0045
    assert np.isclose(g.W[0, 1] - w0[0, 1], 0.01 * 0.5 * 0.9)
    assert g.W[1, 0] == w0[1, 0]           # target earned nothing
    assert g.W[0, 2] == w0[0, 2]


def test_obs_credit_masked_out(hcfg):
    g = _graph(hcfg, social_act_channels=("comm",))
    w0 = g.W.copy()
    g.update(FAR, social_events=[(0, 1, "obs")],
             bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
    np.testing.assert_array_equal(g.W, w0)


# ── imitation credit within radius (THE regression guard) ───────────────────

def test_imit_within_radius_earns_credit(hcfg):
    """Agents 0/1 are 1 block apart — exactly the situation the act gate
    requires — and the imit term must NOT be suppressed by co-location."""
    g = _graph(hcfg, social_act_channels=("imit",))
    w0 = g.W.copy()
    g.update(NEAR, social_events=[(0, 1, "imit")],
             bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
    # g_i = 0.5 (initiator socially engaged), g_j = 0 → c_spat = 0;
    # c_imit = 0.5 carries the whole credit: ΔW[0,1] = 0.01·0.5·0.9
    assert np.isclose(g.W[0, 1] - w0[0, 1], 0.01 * 0.5 * 0.9)
    assert g.W[1, 0] == w0[1, 0]           # directed


def test_custom_social_delta_used_for_obs_imit(hcfg):
    g = _graph(hcfg, social_act_channels=("imit",),
               social_coactivity_bonus=0.8)
    g.update(NEAR, social_events=[(0, 1, "imit")],
             bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
    assert np.isclose(g.W[0, 1] - 0.1, 0.01 * 0.8 * 0.9)


# ── empty mask = proximity+reward floor ─────────────────────────────────────

def test_empty_mask_is_social_silent(hcfg):
    g = _graph(hcfg, social_act_channels=())
    w0 = g.W.copy()
    g.update(NEAR, comm_events=[(0, 1)],
             social_events=[(0, 1, "obs"), (1, 0, "imit")],
             bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
    # No credited social act, zero reward → g_i = 0 → nothing anywhere.
    np.testing.assert_array_equal(g.W, w0)


def test_empty_mask_reward_proximity_still_wires(hcfg):
    g = _graph(hcfg, social_act_channels=())
    g.update(NEAR, bond_rewards=[300.0, 300.0, 0.0],
             total_rewards=[300.0, 300.0, 0.0],
             chambers=[2, 2, 2])
    # Both near agents fully reward-engaged: g = 0.5 → c_spat = 0.25.
    assert g.W[0, 1] > 0.11 and g.W[1, 0] > 0.11
    # Far pair untouched (decay=0 in fixture): still at init_weight.
    assert g.W[0, 2] == np.float32(0.1)


# ── attribution ─────────────────────────────────────────────────────────────

def test_attribution_sums_to_total_growth(hcfg):
    g = _graph(hcfg, social_act_channels=("comm", "obs", "imit"))
    total = np.zeros((3, 3), dtype=np.float64)
    for _ in range(4):
        g.update(NEAR, comm_events=[(0, 2)],
                 social_events=[(0, 1, "imit"), (1, 0, "obs")],
                 bond_rewards=[10.0, 0.0, 0.0], total_rewards=[10.0, 0.0, 0.0],
                 chambers=[2, 2, 2])
        total += g._last_growth
    attr = g.get_channel_attribution()
    assert set(attr) == {"spat", "comm", "obs", "imit", "total"}
    by_channel = sum(np.array(attr[ch]) for ch in ("spat", "comm", "obs", "imit"))
    np.testing.assert_allclose(by_channel, total, atol=1e-5)
    assert np.isclose(attr["total"], total.sum(), atol=1e-5)


def test_attribution_channels_are_the_right_ones(hcfg):
    g = _graph(hcfg, social_act_channels=("obs",))
    g.update(FAR, social_events=[(0, 1, "obs")],
             bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
    attr = g.get_channel_attribution()
    assert np.array(attr["obs"])[0][1] > 0.0
    assert np.array(attr["comm"]).sum() == 0.0
    assert np.array(attr["imit"]).sum() == 0.0
    assert np.array(attr["spat"]).sum() == 0.0


def test_attribution_disabled_and_reset(hcfg):
    g_off = HebbianSocialGraph(HebbianConfig())      # disabled no-op
    assert g_off.get_channel_attribution() == {}
    g = _graph(hcfg, social_act_channels=("obs",))
    g.update(FAR, social_events=[(0, 1, "obs")],
             bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
    assert g.get_channel_attribution()["total"] > 0.0
    g.reset()
    assert g.get_channel_attribution()["total"] == 0.0


# ── directed engagement ─────────────────────────────────────────────────────

def test_obs_initiator_only_engagement(hcfg):
    """comm engages both parties; obs engages the initiator only."""
    g = _graph(hcfg)
    eng_comm = g._engagement(np.zeros(3, dtype=np.float32), {0, 1})
    assert eng_comm[0] == eng_comm[1] == 0.5
    eng_obs = g._engagement(np.zeros(3, dtype=np.float32), {0})
    assert eng_obs[0] == 0.5 and eng_obs[1] == 0.0


# ── delivery-symmetric obs/imit (--social-bidirectional) ────────────────────

def test_bidirectional_defaults_off():
    """Non-regression pin: the flag defaults to the directed legacy terms."""
    assert HebbianConfig().social_bidirectional is False


def test_bidirectional_obs_credits_both_directions(hcfg):
    g = _graph(hcfg, social_act_channels=("obs",), social_bidirectional=True)
    w0 = g.W.copy()
    g.update(FAR, social_events=[(0, 1, "obs")],
             bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
    # Same per-cell arithmetic as the directed test — now BOTH directions.
    assert np.isclose(g.W[0, 1] - w0[0, 1], 0.01 * 0.5 * 0.9)
    assert np.isclose(g.W[1, 0] - w0[1, 0], 0.01 * 0.5 * 0.9)
    assert g.W[0, 2] == w0[0, 2]           # uninvolved pair untouched


def test_bidirectional_imit_credits_both_directions(hcfg):
    """Co-located adoption (the act gate's own situation) credits the pair
    symmetrically; c_spat stays 0 (target is not socially engaged)."""
    g = _graph(hcfg, social_act_channels=("imit",), social_bidirectional=True)
    w0 = g.W.copy()
    g.update(NEAR, social_events=[(0, 1, "imit")],
             bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
    assert np.isclose(g.W[0, 1] - w0[0, 1], 0.01 * 0.5 * 0.9)
    assert np.isclose(g.W[1, 0] - w0[1, 0], 0.01 * 0.5 * 0.9)


def test_bidirectional_attribution_matrices_symmetric(hcfg):
    g = _graph(hcfg, social_act_channels=("obs", "imit"),
               social_bidirectional=True)
    g.update(FAR, social_events=[(0, 1, "obs"), (2, 0, "imit")],
             bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
    np.testing.assert_array_equal(g._last_c_obs, g._last_c_obs.T)
    np.testing.assert_array_equal(g._last_c_imit, g._last_c_imit.T)


def test_bidirectional_leaves_comm_and_engagement_unchanged(hcfg):
    """The flag touches ONLY the obs/imit terms: comm events produce the
    identical W trajectory with the flag on and off (comm was already
    symmetric), and the target of an obs event still earns no engagement."""
    ga = _graph(hcfg)
    gb = _graph(hcfg, social_bidirectional=True)
    for _ in range(3):
        ga.update(FAR, comm_events=[(0, 1)],
                  bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
        gb.update(FAR, comm_events=[(0, 1)],
                  bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
    np.testing.assert_array_equal(ga.W, gb.W)


def test_observed_and_imitated_notices_name_the_initiator():
    from mindforge.agent_modules.social_acts import (
        render_imitated_notice, render_observed_notice,
    )
    obs = render_observed_notice("agent_2")
    assert "agent_2" in obs and "observed you" in obs
    imit = render_imitated_notice("agent_4", "Dig")
    assert "agent_4" in imit and "'Dig'" in imit
    # Missing action falls back to generic phrasing, never "None".
    assert "None" not in render_imitated_notice("agent_4", None)


# ── distance-free comm (--comm-distance-free) ───────────────────────────────

def test_comm_distance_free_defaults_off():
    assert HebbianConfig().comm_distance_free is False


def test_legacy_comm_reduced_when_colocated(hcfg):
    """Pin the legacy rule this flag relaxes: co-located messaging earns only
    the engagement-gated spatial credit (comm itself is zeroed by the
    (1-spatial) factor). Both parties of a comm event count as socially
    engaged — unlike obs, which is initiator-only — so c_spat = 0.5·0.5 =
    0.25 < the 0.5 the same message earns at distance: the 'penalty for
    talking while together'."""
    g = _graph(hcfg)
    w0 = g.W.copy()
    g.update(NEAR, comm_events=[(0, 1)],
             bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
    assert np.isclose(g.W[0, 1] - w0[0, 1], 0.01 * 0.25 * 0.9)
    assert np.isclose(g._last_c_comm[0, 1], 0.0)


def test_comm_distance_free_credits_when_colocated(hcfg):
    g = _graph(hcfg, comm_distance_free=True)
    w0 = g.W.copy()
    g.update(NEAR, comm_events=[(0, 1)],
             bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
    # comm (0.5) now stacks on the mutual-engagement spatial credit (0.25):
    # c = clip(0.25 + 0.5) = 0.75 → ΔW = 0.01·0.75·0.9 each way.
    assert np.isclose(g.W[0, 1] - w0[0, 1], 0.01 * 0.75 * 0.9)
    assert np.isclose(g.W[1, 0] - w0[1, 0], 0.01 * 0.75 * 0.9)
    assert np.isclose(g._last_c_comm[0, 1], 0.5)


def test_comm_distance_free_far_pair_unchanged(hcfg):
    """At distance the factor was already 1, so the flag is a no-op there."""
    ga = _graph(hcfg)
    gb = _graph(hcfg, comm_distance_free=True)
    for _ in range(3):
        ga.update(FAR, comm_events=[(0, 1)],
                  bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
        gb.update(FAR, comm_events=[(0, 1)],
                  bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
    np.testing.assert_array_equal(ga.W, gb.W)


def test_colocated_engaged_chat_stacks_and_clips(hcfg):
    """Both agents messaging while co-located: spatial (g_i·g_j = 0.25) +
    comm (0.5) stack to c = 0.75 under the flag — the amplified
    proximity+chatter regime the config comment warns about."""
    g = _graph(hcfg, comm_distance_free=True)
    w0 = g.W.copy()
    g.update(NEAR, comm_events=[(0, 1), (1, 0)],
             bond_rewards=[0.0] * 3, total_rewards=[0.0] * 3)
    assert np.isclose(g.W[0, 1] - w0[0, 1], 0.01 * 0.75 * 0.9)
