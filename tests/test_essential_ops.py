"""Unit tests for research/essential_ops/canonical_ops.py.

Covers:
  - OP_PATTERNS regex matching (each pattern, each alternative)
  - find_ops() composition rules (sweep+helix, polyline⊃lineTo,
    closed-polyline⊃polygon, cylinder+cut→hole)
  - essential_pass() AND-of-OR-tuples logic
  - feature_f1() symmetric difference

Design principle (per review): the metric is the regex + family-spec
combination. Global aliases in find_ops() are ONLY true syntactic sugar
(polyline IS a chain of lineTos; closed polyline IS a polygon outline).
Per-family equivalences (e.g. loft accepted for taper_pin) belong in
canonical_ops.yaml, NOT as global aliases — because loft and revolve
are different ops with overlapping but non-identical use-cases.

Run:
    pytest tests/test_canonical_ops.py -v
"""
from __future__ import annotations

import os
import sys
import textwrap

import pytest

sys.path.insert(0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'research', 'essential_ops'))

from common.essential_ops import (
    OP_PATTERNS, FEATURE_CLASS, ESSENTIAL_BY_FAMILY,
    find_ops, essential_pass, essential_score, feature_f1,
)


# ── pattern matching ──────────────────────────────────────────────────────

class TestPatterns:
    """Each OP_PATTERNS entry must match the canonical syntax + alternatives."""

    def test_revolve(self):
        assert find_ops(".revolve(360)") == {"revolve"}
        assert find_ops("result.revolve(180, (0,0,0), (0,1,0))") == {"revolve"}

    def test_loft(self):
        assert find_ops(".loft()") == {"loft"}
        assert find_ops(".circle(5).workplane(offset=10).circle(3).loft()") == {"loft"}

    def test_sweep_no_helix(self):
        # Plain sweep (no makeHelix in code) → just sweep
        ops = find_ops("result.sweep(path)")
        assert ops == {"sweep"}

    def test_sweep_plus_helix(self):
        # When both .sweep( and makeHelix are present anywhere → sweep+helix,
        # plain sweep is dropped (composite rule).
        code = textwrap.dedent('''
            path = cq.Wire.makeHelix(pitch, height, radius)
            result = profile.sweep(path)
        ''')
        ops = find_ops(code)
        assert "sweep+helix" in ops
        assert "sweep" not in ops

    def test_sphere(self):
        assert find_ops(".sphere(10)") == {"sphere"}
        assert find_ops("cq.Workplane().makeSphere(5)") == {"sphere"}

    def test_cut_basic(self):
        assert find_ops(".cut(other)") == {"cut"}
        assert find_ops(".cutBlind(-5)") == {"cut"}

    def test_cut_via_mode_s(self):
        # Sketch subtract: `mode='s'` is the in-sketch carve op,
        # semantically equivalent to a cut.
        code = ".rect(10, 10).circle(2, mode='s')"
        ops = find_ops(code)
        assert "cut" in ops

    def test_cut_via_mode_s_double_quote(self):
        ops = find_ops('.circle(2, mode="s")')
        assert "cut" in ops

    def test_polyline_explicit(self):
        ops = find_ops(".polyline([(0,0),(1,0),(1,1),(0,1)])")
        assert "polyline" in ops
        assert "lineTo" in ops, "polyline IS a chain of lineTo (sugar)"

    def test_polyline_via_segment(self):
        # BenchCAD shell-style: chain of .segment() calls forms a polyline.
        code = ".sketch().segment((0,0),(1,0)).segment((1,1)).segment((0,1)).close()"
        ops = find_ops(code)
        assert "polyline" in ops, "segment chain should match polyline pattern"
        assert "lineTo" in ops, "polyline should imply lineTo"

    def test_closed_polyline_implies_polygon(self):
        code = ".polyline([(0,0),(1,0),(1,1),(0,1)]).close()"
        ops = find_ops(code)
        assert "polygon" in ops, "closed polyline IS a polygon outline"

    def test_open_polyline_does_NOT_imply_polygon(self):
        # No .close() → not a polygon
        ops = find_ops(".polyline([(0,0),(1,0),(1,1),(0,1)])")
        assert "polygon" not in ops

    def test_sketch_class(self):
        assert find_ops("cq.Sketch()") == {"Sketch"}
        assert find_ops(".placeSketch(s)") == {"Sketch"}

    def test_sketch_lowercase_instance_method(self):
        # BenchCAD shell-style `.sketch()` should also match.
        ops = find_ops(".sketch()")
        assert "Sketch" in ops

    def test_polygon(self):
        assert find_ops(".polygon(6, 10)") == {"polygon"}

    def test_lineTo_explicit(self):
        ops = find_ops(".lineTo(5, 0)")
        assert "lineTo" in ops

    def test_chamfer_fillet_hole(self):
        ops = find_ops(".chamfer(1.0)")
        assert "chamfer" in ops
        ops = find_ops(".fillet(0.5)")
        assert "fillet" in ops
        ops = find_ops(".hole(3.0)")
        assert "hole" in ops
        ops = find_ops(".cutThruAll()")
        assert "hole" in ops, "cutThruAll is a hole-class op"

    def test_polarArray_rarray(self):
        assert find_ops(".polarArray(10, 0, 360, 6)") == {"polarArray"}
        assert find_ops(".rarray(10, 10, 4, 4)") == {"rarray"}

    def test_makeTorus(self):
        assert find_ops("cq.Solid.makeTorus(20, 5)") == {"makeTorus"}

    def test_shell(self):
        assert find_ops(".shell(0.5)") == {"shell"}

    def test_taper_kwarg(self):
        assert find_ops(".extrude(10, taper=5)") == {"taper="}


class TestSemanticAliases:
    """Inclusion-style aliases — true syntactic sugar only."""

    def test_polyline_implies_lineTo(self):
        """polyline IS a chain of lineTos by definition."""
        assert "lineTo" in find_ops(".polyline([(0,0),(1,1)])")
        assert "lineTo" in find_ops(".sketch().segment((0,0),(1,1))")

    def test_lineTo_alone_does_NOT_imply_polyline(self):
        # The reverse is asymmetric — a single lineTo isn't a polyline.
        ops = find_ops(".moveTo(0,0).lineTo(1,0)")
        assert "lineTo" in ops
        assert "polyline" not in ops

    def test_cylinder_plus_cut_implies_hole(self):
        """`cut(cylinder(...))` is what `.hole()` does internally."""
        code = "result.cut(cq.Workplane().cylinder(10, 2))"
        ops = find_ops(code)
        assert "hole" in ops

    def test_cylinder_alone_does_NOT_imply_hole(self):
        ops = find_ops("cq.Workplane().cylinder(10, 5)")
        assert "hole" not in ops

    def test_loft_does_NOT_imply_revolve(self):
        """Per review: loft and revolve are DIFFERENT ops, not inclusion."""
        ops = find_ops(".circle(5).workplane(offset=10).circle(3).loft()")
        assert "loft" in ops
        assert "revolve" not in ops, "loft is NOT revolve — per-family alternative belongs in spec"
        assert "sweep" not in ops, "loft is NOT sweep"

    def test_pushPoints_does_NOT_imply_rarray(self):
        """Per review: pushPoints can be irregular; per-family in spec."""
        code = ".pushPoints([(1,1),(2,2),(3,3),(4,4)]).hole(1)"
        ops = find_ops(code)
        assert "rarray" not in ops


# ── essential_pass logic ──────────────────────────────────────────────────

class TestEssentialPass:

    def test_na_when_family_unknown(self):
        assert essential_pass("not_a_family", {"cut", "hole"}) is None

    def test_simple_string_spec_pass(self):
        # propeller spec is [loft] — single op required
        assert essential_pass("propeller", {"loft", "polyline"}) is True

    def test_simple_string_spec_fail(self):
        assert essential_pass("propeller", {"revolve"}) is False

    def test_or_tuple_pass_first(self):
        # ball_knob: [(revolve | sphere)]
        assert essential_pass("ball_knob", {"revolve"}) is True

    def test_or_tuple_pass_second(self):
        assert essential_pass("ball_knob", {"sphere"}) is True

    def test_or_tuple_fail(self):
        assert essential_pass("ball_knob", {"cut", "hole"}) is False

    def test_and_of_or_pass(self):
        # hex_nut: [(polygon|lineTo), (cut|hole)] — AND of two OR-tuples
        assert essential_pass("hex_nut", {"polygon", "cut"}) is True
        assert essential_pass("hex_nut", {"lineTo", "hole"}) is True

    def test_and_of_or_partial_fail(self):
        # missing the cut/hole element
        assert essential_pass("hex_nut", {"polygon"}) is False
        # missing the polygon/lineTo element
        assert essential_pass("hex_nut", {"cut"}) is False


class TestPerFamilyAlternatives:
    """Family specs accept multiple OPs in OR-tuples; loft/revolve/sweep
    interchangeability is encoded per-family, NOT globally."""

    def test_taper_pin_accepts_revolve(self):
        # 圆台 / truncated cone — revolve a tapered profile
        assert essential_pass("taper_pin", {"revolve"}) is True

    def test_taper_pin_accepts_loft(self):
        # 圆台 / truncated cone — loft between two circles is equivalent
        # for THIS family (not globally; see canonical_ops.yaml)
        assert essential_pass("taper_pin", {"loft"}) is True

    def test_taper_pin_accepts_sweep(self):
        assert essential_pass("taper_pin", {"sweep"}) is True

    def test_taper_pin_rejects_cut_alone(self):
        assert essential_pass("taper_pin", {"cut"}) is False

    def test_propeller_accepts_loft_only(self):
        # propeller spec is [loft] — strict; revolve does NOT substitute
        assert essential_pass("propeller", {"loft"}) is True
        assert essential_pass("propeller", {"revolve"}) is False
        assert essential_pass("propeller", {"sweep"}) is False

    def test_helical_gear_needs_loft_AND_profile(self):
        # helical_gear: [[loft|twistExtrude], [polyline|spline|threePointArc|Sketch]]
        assert essential_pass("helical_gear",
                              {"loft", "polyline"}) is True
        assert essential_pass("helical_gear",
                              {"twistExtrude", "Sketch"}) is True
        # missing profile element
        assert essential_pass("helical_gear", {"loft"}) is False
        # missing structural element
        assert essential_pass("helical_gear", {"polyline"}) is False


# ── essential_score (fractional / partial credit) ────────────────────────

class TestEssentialScore:
    """Fractional score in [0, 1] = #satisfied_AND_elements / #total."""

    def test_na_when_family_unknown(self):
        assert essential_score("not_a_family", {"cut"}) is None

    def test_single_element_pass(self):
        # propeller: [loft] — 1 element
        assert essential_score("propeller", {"loft"}) == 1.0

    def test_single_element_fail(self):
        assert essential_score("propeller", {"revolve"}) == 0.0

    def test_two_elements_partial(self):
        # helical_gear: [(loft|twistExtrude), (polyline|spline|threePointArc|Sketch)]
        assert essential_score("helical_gear", {"polyline"}) == 0.5
        assert essential_score("helical_gear", {"loft"}) == 0.5
        assert essential_score("helical_gear", {"loft", "polyline"}) == 1.0
        assert essential_score("helical_gear", set()) == 0.0
        assert essential_score("helical_gear", {"cut", "hole"}) == 0.0

    def test_or_tuple_alternatives_dont_double_count(self):
        # Each AND-element contributes at most 1 to the numerator,
        # regardless of how many OR-tuple alternatives are matched.
        assert essential_score("ball_knob", {"revolve", "sphere"}) == 1.0
        assert essential_score("ball_knob", {"revolve"}) == 1.0
        assert essential_score("ball_knob", {"sphere"}) == 1.0

    def test_pass_iff_score_one(self):
        # essential_pass and essential_score must agree at the boundaries.
        for fam in ESSENTIAL_BY_FAMILY:
            spec = ESSENTIAL_BY_FAMILY[fam]
            full = set()
            for el in spec:
                if isinstance(el, str): full.add(el)
                else: full.add(el[0])  # pick first alt
            assert essential_pass(fam, full) is True
            assert essential_score(fam, full) == 1.0
            assert essential_pass(fam, set()) is False
            assert essential_score(fam, set()) == 0.0


# ── feature_f1 ────────────────────────────────────────────────────────────

class TestFeatureF1:

    def test_perfect_match(self):
        assert feature_f1({"chamfer", "hole"}, {"chamfer", "hole"}) == 1.0

    def test_no_features_either_side(self):
        # both empty for the {chamfer, fillet, hole} keys → F1 = 1.0
        assert feature_f1({"cut", "polyline"}, {"cut", "revolve"}) == 1.0

    def test_pure_miss(self):
        # GT has hole, pred has nothing → precision=N/A, recall=0
        ops = feature_f1({"cut"}, {"cut", "hole"})
        assert ops == 0.0

    def test_pure_spurious(self):
        # GT has nothing, pred has hole → precision=0, recall=N/A
        assert feature_f1({"cut", "hole"}, {"cut"}) == 0.0


# ── end-to-end real-world cases (regression tests) ────────────────────────

class TestRealCases:
    """Cases pulled from cad_bench_722 audit — verify metric verdict
    matches the documented expectation."""

    BATTERY_HOLDER_CODE = textwrap.dedent('''\
        wp1 = cq.Workplane('ZX').sketch()
        wp2 = wp1.rect(181, 200)
        wp3 = wp2.push([(5.5, 0)]).rect(23, 192, mode='s')
        wp4 = wp3.finalize().extrude(-36)
    ''')

    def test_battery_holder_passes_via_mode_s(self):
        ops = find_ops(self.BATTERY_HOLDER_CODE)
        # spec [(cut, hole)] satisfied by mode='s' → cut
        assert essential_pass("battery_holder", ops) is True

    HEX_NUT_CADEVOLVE_CODE = textwrap.dedent('''\
        wp1 = cq.Workplane('ZX').sketch()
        wp2 = wp1.segment((-93,-39), (-80,-61))
        wp3 = wp2.segment((-13,-100))
        wp4 = wp3.segment((13,-100))
        wp5 = wp4.segment((80,-61))
        wp6 = wp5.segment((93,-39))
        wp7 = wp6.segment((93,39))
        wp8 = wp7.close().assemble().circle(57, mode='s')
        wp9 = wp8.finalize().extrude(-97)
    ''')

    def test_hex_nut_passes_via_segment_chain_plus_mode_s(self):
        ops = find_ops(self.HEX_NUT_CADEVOLVE_CODE)
        # spec [(polygon|lineTo), (cut|hole)]
        # polygon ← polyline+close, cut ← mode='s'
        assert "polygon" in ops, "closed segment chain should imply polygon"
        assert "cut" in ops, "mode='s' should imply cut"
        assert essential_pass("hex_nut", ops) is True

    TAPER_PIN_CADEVOLVE_CODE = textwrap.dedent('''\
        wp1 = cq.Workplane('XY').circle(10)
        wp2 = wp1.workplane(offset=200).circle(12)
        wp3 = wp2.loft()
    ''')

    def test_taper_pin_passes_via_loft(self):
        # spec was originally [(revolve, sweep)] → fail
        # after adding `loft` per-family alternative → pass
        ops = find_ops(self.TAPER_PIN_CADEVOLVE_CODE)
        assert "loft" in ops
        assert "revolve" not in ops, "loft does NOT globally imply revolve"
        assert essential_pass("taper_pin", ops) is True, \
            "taper_pin spec should accept loft per-family"

    GUSSETED_BRACKET_CODE = textwrap.dedent('''\
        wp1 = cq.Workplane('XY').sketch()
        wp2 = wp1.segment((-100,-50), (100,-50))
        wp3 = wp2.segment((100,-7))
        wp4 = wp3.segment((43,-2))
        wp5 = wp4.close().assemble().finalize().extrude(-125)
    ''')

    def test_gusseted_bracket_legit_fail(self):
        # spec [(cut, hole)] — model draws final outline directly,
        # never uses cut OR hole. This is a LEGITIMATE fail.
        ops = find_ops(self.GUSSETED_BRACKET_CODE)
        assert "cut" not in ops, "no .cut(/.cutBlind(/mode='s' in code"
        assert "hole" not in ops, "no .hole(/.cutThruAll/etc in code"
        assert essential_pass("gusseted_bracket", ops) is False
