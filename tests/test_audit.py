"""端到端与单元测试。使用 SQLite 覆盖数据库，无需 PostgreSQL。"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest

os.environ["DATABASE_URL"] = "sqlite:///" + tempfile.mktemp(suffix=".db")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app import gcode as gc  # noqa: E402
from app.clearance import minimum_clearance  # noqa: E402
from app.collision import Zone, check_path, initial_collision  # noqa: E402
from app.geometry import GeometryError, normalize_polygon  # noqa: E402
from app.main import app  # noqa: E402
from app.database import Base, engine  # noqa: E402

Base.metadata.create_all(bind=engine)
client = TestClient(app)

SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]]
SAFE = (-50.0, -50.0, 0.0)


def parse(text, start=SAFE):
    return gc.parse_gcode(text, start)


class GeometryTests(unittest.TestCase):
    def test_bad_polygons(self):
        with self.assertRaises(GeometryError):
            normalize_polygon([[0, 0], [1, 1]])
        with self.assertRaises(GeometryError):
            normalize_polygon([[0, 0], [1, 0], [0.5, 0]])  # 零面积
        with self.assertRaises(GeometryError):  # 蝴蝶结自相交
            normalize_polygon([[0, 0], [10, 10], [10, 0], [0, 10]])

    def test_closing_duplicate_point_accepted(self):
        poly = normalize_polygon([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])
        self.assertEqual(len(poly), 4)


class LineCollisionTests(unittest.TestCase):
    def setUp(self):
        # 10x10 方块禁区，z ∈ [-5, 5]
        self.zone = Zone(1, normalize_polygon(SQUARE), -5.0, 5.0)

    def test_mid_segment_hit_not_endpoints(self):
        # 两端都在方块外侧，但路径横穿方块内部（仅采样端点会漏检）
        motions = parse("G90 G21 G17\nG1 X-5 Y5\nG1 X15 Y5\n").motions
        hit = check_path(motions, [self.zone], 0.0)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.line_no, 3)

    def test_pass_above_zone(self):
        motions = parse("G1 X-5 Y20\nG1 X15 Y20\n").motions
        self.assertIsNone(check_path(motions, [self.zone], 0.5))

    def test_tool_radius_touch_at_distance(self):
        # 路径距边恰为 1mm，刀半径 1 => 边界接触算碰撞
        motions = parse("G1 X-5 Y11\nG1 X15 Y11\n").motions
        self.assertIsNone(check_path(motions, [self.zone], 0.99))
        hit = check_path(motions, [self.zone], 1.0)
        self.assertIsNotNone(hit)

    def test_z_out_of_range_no_hit(self):
        motions = parse("G1 X-5 Y5 Z10\nG1 X15 Y5 Z10\n").motions
        self.assertIsNone(check_path(motions, [self.zone], 0.0))
        # 斜线 Z 从范围外穿入范围内：途中碰撞
        motions2 = parse("G1 X-5 Y5 Z20\nG1 X15 Y5 Z0\n").motions
        hit = check_path(motions2, [self.zone], 0.0)
        self.assertIsNotNone(hit)

    def test_diagonal_through_corner(self):
        motions = parse("G1 X-2 Y-2\nG1 X12 Y12\n").motions
        hit = check_path(motions, [self.zone], 0.0)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.line_no, 2)

    def test_grazing_tangent_contact(self):
        # 与边重合贴边（从边外沿 X 轴进入）
        motions = parse("G1 X-5 Y0\nG1 X10 Y0\n").motions
        hit = check_path(motions, [self.zone], 0.0)
        self.assertIsNotNone(hit)

    def test_line_completely_inside(self):
        motions = gc.parse_gcode("G1 X2 Y2\nG1 X8 Y8\n",
                                 (2.0, 2.0, 0.0)).motions
        hit = check_path(motions, [self.zone], 0.0)
        self.assertIsNotNone(hit)


class ArcCollisionTests(unittest.TestCase):
    def setUp(self):
        # 弧：圆心 (0,0) 半径 10，从 (10,0) 逆时针整圆；禁区跨在圆顶 (0,10)
        self.zone = Zone(7, normalize_polygon(
            [[-1, 9], [1, 9], [1, 11], [-1, 11]]), -5.0, 5.0)

    def test_full_circle_hits_mid_arc(self):
        text = "G90 G17 G21\nG1 X10 Y0 Z0\nG3 X10 Y0 I-10 J0\n"
        motions = gc.parse_gcode(text, SAFE).motions
        hit = check_path(motions, [self.zone], 0.0)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.line_no, 3)
        self.assertEqual(hit.zone_no, 7)
        # 首个碰撞约在 1/4 圆处（进度 ~0.25）
        self.assertAlmostEqual(hit.progress, 0.25, delta=0.02)

    def test_clockwise_full_circle(self):
        text = "G1 X10 Y0\nG2 X10 Y0 I-10 J0\n"
        motions = parse(text).motions
        hit = check_path(motions, [self.zone], 0.0)
        self.assertIsNotNone(hit)

    def test_arc_misses(self):
        # 禁区在圆外 (0,20) 附近
        zone = Zone(1, normalize_polygon([[19, -1], [21, -1], [21, 1], [19, 1]]),
                    -5, 5)
        text = "G1 X10 Y0\nG3 X10 Y0 I-10 J0\n"
        motions = parse(text).motions
        self.assertIsNone(check_path(motions, [zone], 0.0))

    def test_arc_contact_with_radius(self):
        # 弧半径 10，禁区边在半径 10.5 处：刀半径 0.5 恰好接触
        zone = Zone(1, normalize_polygon(
            [[-1, 10.5], [1, 10.5], [1, 12], [-1, 12]]), -5, 5)
        text = "G1 X10 Y0\nG3 X10 Y0 I-10 J0\n"
        motions = parse(text).motions
        self.assertIsNone(check_path(motions, [zone], 0.4))
        self.assertIsNotNone(check_path(motions, [zone], 0.5))

    def test_semicircle_inside_zone(self):
        # 整个半圆都在大禁区内
        zone = Zone(1, normalize_polygon(
            [[-15, -15], [15, -15], [15, 15], [-15, 15]]), -5, 5)
        text = "G90\nG1 X10 Y0 Z50\nG1 Z0\nG3 X-10 Y0 I-10 J0\n"
        motions = gc.parse_gcode(text, (10.0, 0.0, 50.0)).motions
        hit = check_path(motions, [zone], 0.0)
        self.assertIsNotNone(hit)
        # 第 3 行 Z 向下降进入高度范围时已位于禁区内
        self.assertEqual(hit.line_no, 3)


class ParserModalTests(unittest.TestCase):
    def test_incremental_and_unit(self):
        # G20 英寸：X 走 1 英寸 = 25.4mm；G91 增量
        r = gc.parse_gcode("G20 G91\nG1 X1 Y0\n", (0, 0, 0))
        self.assertAlmostEqual(r.final_position[0], 25.4, places=6)

    def test_missing_axis_inherits(self):
        r = gc.parse_gcode("G90\nG1 X5 Y7 Z3\nG1 X8\n", (0, 0, 0))
        self.assertEqual(r.final_position, (8.0, 7.0, 3.0))

    def test_arc_z_change_rejected(self):
        with self.assertRaises(gc.GCodeError):
            gc.parse_gcode("G1 X10 Y0 Z0\nG3 X0 Y10 Z1 I-10 J0\n", (0, 0, 0))

    def test_arc_zero_ij_rejected(self):
        with self.assertRaises(gc.GCodeError):
            gc.parse_gcode("G1 X10 Y0\nG3 X0 Y10\n", (0, 0, 0))

    def test_arc_radius_mismatch_rejected(self):
        with self.assertRaises(gc.GCodeError):
            gc.parse_gcode("G1 X10 Y0\nG3 X0 Y9 I-10 J0\n", (0, 0, 0))

    def test_unsupported_g_commands(self):
        for bad in ("G54 G1 X1", "G18 G1 X1", "G81 X1 Y1 Z-1", "G28"):
            with self.assertRaises(gc.GCodeError):
                gc.parse_gcode(bad, (0, 0, 0))

    def test_r_arc_rejected(self):
        with self.assertRaises(gc.GCodeError):
            gc.parse_gcode("G1 X0 Y0\nG2 X10 Y0 R5\n", (0, 0, 0))

    def test_comments_and_line_numbers(self):
        text = "G90 ; start\n(hole path)\nN10 G1 X5 Y5\n"
        r = gc.parse_gcode(text, (0, 0, 0))
        # 第 1 行 G90 是有效非空行；第 2 行纯注释；第 3 行运动
        self.assertEqual(r.nonblank_count, 2)
        self.assertEqual(r.motions[0].line_no, 3)

    def test_unbalanced_paren_rejected(self):
        with self.assertRaises(gc.GCodeError):
            gc.parse_gcode("G1 X1 (oops\n", (0, 0, 0))

    def test_arc_is_constant_z_incremental_z0(self):
        r = gc.parse_gcode("G90\nG1 X10 Y0 Z2\nG91\nG3 X0 Y0 I-10 J0\n",
                           (0, 0, 0))
        self.assertEqual(len(r.motions), 2)
        self.assertAlmostEqual(abs(r.motions[1].sweep), 6.2831853, places=5)

    def test_ij_relative_to_start(self):
        r = gc.parse_gcode("G1 X10 Y0\nG3 X0 Y10 I-10 J0\n", (0, 0, 0))
        arc = r.motions[1]
        self.assertAlmostEqual(arc.center[0], 0.0)
        self.assertAlmostEqual(arc.center[1], 0.0)


class InitialPointTests(unittest.TestCase):
    def test_initial_inside(self):
        zone = Zone(1, normalize_polygon(SQUARE), -5, 5)
        self.assertEqual(initial_collision((5, 5, 0), [zone], 0.0), 1)
        self.assertIsNone(initial_collision((5, 5, 9), [zone], 0.0))


class ClearanceEngineTests(unittest.TestCase):
    """全程最小水平净间隙（毫米）。"""

    def setUp(self):
        # 10x10 方块 [0,10]^2，z ∈ [-5, 5]
        self.zone = Zone(1, normalize_polygon(SQUARE), -5.0, 5.0)

    def clr(self, text, start, radius=0.0, zones=None):
        motions = gc.parse_gcode(text, start).motions
        return minimum_clearance(motions, zones or [self.zone], start, radius)

    def test_line_mid_positive(self):
        # 路径 y=11，距 y=10 边 1mm，r=0.5 => 0.5；最小值可在段中（不只是端点）
        r = self.clr("G1 X-5 Y11 Z0\nG1 X15 Y11 Z0\n", SAFE, 0.5)
        self.assertAlmostEqual(r.min_clearance_mm, 0.5, places=9)
        self.assertEqual(r.line_no, 2)
        self.assertEqual(r.zone_no, 1)

    def test_line_boundary_contact_is_zero(self):
        r = self.clr("G1 X-5 Y10.5 Z0\nG1 X15 Y10.5 Z0\n", SAFE, 0.5)
        self.assertEqual(r.min_clearance_mm, 0.0)

    def test_line_negative_inside(self):
        # 横穿填充区：净间隙 -r，取最早进入点
        r = self.clr("G1 X-5 Y5 Z0\nG1 X15 Y5 Z0\n", SAFE, 1.0)
        self.assertEqual(r.min_clearance_mm, -1.0)
        self.assertAlmostEqual(r.tip_position_mm[0], 0.0, places=9)
        self.assertAlmostEqual(r.progress, 0.25, places=9)

    def test_line_nearest_foot_not_endpoint(self):
        # 与边平行、不到边：垂足段内部极小（x=12 距 x=10 边 2mm）
        r = self.clr("G1 X12 Y-5 Z0\nG1 X12 Y15 Z0\n", SAFE, 0.0)
        self.assertAlmostEqual(r.min_clearance_mm, 2.0, places=9)
        self.assertAlmostEqual(r.tip_position_mm[1], 0.0, places=9)

    def test_z_window_mid_entry(self):
        # 单段斜线 (15,5,20)->(-5,5,-20)；z∈[15,20] 对应 t∈[0,.125]，
        # 路径全在 x∈[12.5,15]，距 x=10 边最近 2.5（在 z=15 边界进入点）
        motions = gc.parse_gcode("G1 X-5 Y5 Z-20\n", (15.0, 5.0, 20.0)).motions
        z = Zone(1, normalize_polygon(SQUARE), 15.0, 20.0)
        r = minimum_clearance(motions, [z], (15.0, 5.0, 20.0), 0.0)
        self.assertAlmostEqual(r.min_clearance_mm, 2.5, places=9)
        self.assertAlmostEqual(r.tip_position_mm[0], 12.5, places=9)
        self.assertAlmostEqual(r.tip_position_mm[2], 15.0, places=9)

    def test_z_window_partial_enter(self):
        # z∈[8,12] 窗口内路径部分进入多边形 => 0，点 (10,5,10)
        motions = gc.parse_gcode("G1 X-5 Y5 Z-20\n", (15.0, 5.0, 20.0)).motions
        z = Zone(1, normalize_polygon(SQUARE), 8.0, 12.0)
        r = minimum_clearance(motions, [z], (15.0, 5.0, 20.0), 0.0)
        self.assertEqual(r.min_clearance_mm, 0.0)
        self.assertAlmostEqual(r.tip_position_mm[0], 10.0, places=9)
        self.assertAlmostEqual(r.tip_position_mm[2], 10.0, places=9)

    def test_never_in_height_returns_none(self):
        motions = gc.parse_gcode("G1 X-5 Y5 Z-20\n", (15.0, 5.0, 20.0)).motions
        z = Zone(1, normalize_polygon(SQUARE), 100.0, 200.0)
        self.assertIsNone(minimum_clearance(motions, [z],
                                            (15.0, 5.0, 20.0), 0.0))

    def test_degenerate_single_point_z_window(self):
        # z 窗口退化为单点 z=5（t=.375），该点 (7.5,5) 在多边形内 => 0
        motions = gc.parse_gcode("G1 X-5 Y5 Z-20\n", (15.0, 5.0, 20.0)).motions
        z = Zone(1, normalize_polygon(SQUARE), 5.0, 5.0)
        r = minimum_clearance(motions, [z], (15.0, 5.0, 20.0), 0.0)
        self.assertEqual(r.min_clearance_mm, 0.0)
        self.assertAlmostEqual(r.tip_position_mm[0], 7.5, places=9)
        self.assertAlmostEqual(r.tip_position_mm[2], 5.0, places=9)

    def test_vertical_plunge_xy_static(self):
        # XY 静止 (5,5) 在区内，Z 从 20 降到 -20：进入高度范围 z=5 即 -r
        motions = gc.parse_gcode("G1 X5 Y5 Z20\nG1 Z-20\n",
                                 (-50.0, -50.0, 0.0)).motions
        r = minimum_clearance(motions, [self.zone], (5.0, 5.0, 20.0), 1.0)
        self.assertEqual(r.min_clearance_mm, -1.0)
        self.assertEqual(r.line_no, 2)
        self.assertEqual(r.tip_position_mm, (5.0, 5.0, 5.0))
        self.assertAlmostEqual(r.progress, 0.375, places=9)

    def test_full_circle_entry(self):
        # 整圆 r=10；圆顶小禁区 [-1,9]×[1,11]；入口在右边 x=1 处
        z = Zone(7, normalize_polygon(
            [[-1, 9], [1, 9], [1, 11], [-1, 11]]), -5.0, 5.0)
        motions = gc.parse_gcode("G3 X10 Y0 I-10 J0\n",
                                 (10.0, 0.0, 0.0)).motions
        r = minimum_clearance(motions, [z], (10.0, 0.0, 0.0), 0.0)
        self.assertEqual(r.min_clearance_mm, 0.0)
        self.assertAlmostEqual(r.tip_position_mm[0], 1.0, places=9)
        self.assertAlmostEqual(r.tip_position_mm[1], math.sqrt(99), places=6)

    def test_full_circle_enclosing_zone_vertex_radial(self):
        # 圆完全包围中心小方块：最近点在 45° 径向上的角点（非弧端点）
        z = Zone(1, normalize_polygon(
            [[-1, -1], [1, -1], [1, 1], [-1, 1]]), -5.0, 5.0)
        motions = gc.parse_gcode("G3 X10 Y0 I-10 J0\n",
                                 (10.0, 0.0, 0.0)).motions
        r = minimum_clearance(motions, [z], (10.0, 0.0, 0.0), 0.0)
        self.assertAlmostEqual(r.min_clearance_mm,
                               10.0 - math.sqrt(2), places=9)
        self.assertAlmostEqual(r.progress, 0.125, places=9)

    def test_arc_contact_and_gap(self):
        # 圆顶 (0,10) 到禁区底边 y=10.5：r=.5 接触=0；r=.4 间隙 .1
        z = Zone(1, normalize_polygon(
            [[-1, 10.5], [1, 10.5], [1, 12], [-1, 12]]), -5.0, 5.0)
        motions = gc.parse_gcode("G3 X10 Y0 I-10 J0\n",
                                 (10.0, 0.0, 0.0)).motions
        self.assertEqual(minimum_clearance(
            motions, [z], (10.0, 0.0, 0.0), 0.5).min_clearance_mm, 0.0)
        r = minimum_clearance(motions, [z], (10.0, 0.0, 0.0), 0.4)
        self.assertAlmostEqual(r.min_clearance_mm, 0.1, places=9)
        self.assertAlmostEqual(r.tip_position_mm[1], 10.0, places=9)

    def test_arc_z_out_of_range_none(self):
        z = Zone(7, normalize_polygon(
            [[-1, 9], [1, 9], [1, 11], [-1, 11]]), 50.0, 60.0)
        motions = gc.parse_gcode("G3 X10 Y0 I-10 J0\n",
                                 (10.0, 0.0, 0.0)).motions
        self.assertIsNone(minimum_clearance(
            motions, [z], (10.0, 0.0, 0.0), 0.0))

    def test_static_no_motions_uses_start(self):
        r = minimum_clearance([], [self.zone], (12.0, 5.0, 0.0), 0.5)
        self.assertAlmostEqual(r.min_clearance_mm, 1.5, places=9)
        self.assertIsNone(r.line_no)
        self.assertEqual(r.tip_position_mm, (12.0, 5.0, 0.0))
        r2 = minimum_clearance([], [self.zone], (5.0, 5.0, 0.0), 2.0)
        self.assertEqual(r2.min_clearance_mm, -2.0)
        self.assertIsNone(minimum_clearance(
            [], [self.zone], (5.0, 5.0, 99.0), 0.0))

    def test_tie_progress_before_zone(self):
        # 对称两禁区净间隙同为 5：左禁区最近点在 t=0（更早）优先
        za = Zone(3, normalize_polygon(
            [[20, -5], [30, -5], [30, 5], [20, 5]]), -5.0, 5.0)
        zb = Zone(5, normalize_polygon(
            [[-30, -5], [-20, -5], [-20, 5], [-30, 5]]), -5.0, 5.0)
        motions = gc.parse_gcode("G1 X15 Y0 Z0\n",
                                 (-15.0, 0.0, 0.0)).motions
        r = minimum_clearance(motions, [za, zb], (-15.0, 0.0, 0.0), 0.0)
        self.assertEqual(r.zone_no, 5)
        self.assertEqual(r.tip_position_mm[0], -15.0)

    def test_tie_same_point_smaller_zone_wins(self):
        z1 = Zone(8, normalize_polygon(
            [[20, 0], [30, 0], [30, 5], [20, 5]]), -5.0, 5.0)
        z2 = Zone(9, normalize_polygon(
            [[20, -5], [30, -5], [30, 0], [20, 0]]), -5.0, 5.0)
        motions = gc.parse_gcode("G1 X15 Y0 Z0\n",
                                 (-15.0, 0.0, 0.0)).motions
        r = minimum_clearance(motions, [z1, z2], (-15.0, 0.0, 0.0), 0.0)
        self.assertEqual(r.zone_no, 8)

    def test_tie_motion_order(self):
        # 第一段最远仅距 4，第二段才进入：最小点在第二段
        z = Zone(1, normalize_polygon(
            [[20, -5], [30, -5], [30, 5], [20, 5]]), -5.0, 5.0)
        motions = gc.parse_gcode("G1 X16 Y0 Z0\nG1 X22 Y0 Z0\n",
                                 (15.0, 0.0, 0.0)).motions
        r = minimum_clearance(motions, [z], (15.0, 0.0, 0.0), 0.0)
        self.assertEqual(r.order, 1)
        self.assertEqual(r.line_no, 2)
        self.assertAlmostEqual(r.tip_position_mm[0], 20.0, places=9)


class ApiTests(unittest.TestCase):
    def test_01_full_flow(self):
        # 登记禁区
        r = client.post("/zones", json={
            "zone_no": 100, "polygon": SQUARE, "z_min": -5, "z_max": 5,
            "description": "基座"})
        self.assertEqual(r.status_code, 201, r.text)

        # 非法几何拒绝
        r = client.post("/zones", json={
            "polygon": [[0, 0], [10, 10], [10, 0], [0, 10]],
            "z_min": -5, "z_max": 5})
        self.assertEqual(r.status_code, 422)

        # 横穿禁区 => 碰撞
        gcode = "G90 G21 G17\nG1 X-5 Y5 Z0\nG1 X15 Y5 Z0\n"
        r = client.post("/reviews", json={
            "zones": [{"zone_no": 100, "version": 1}],
            "start": [-5, 5, 0], "tool_radius": 0.0, "gcode": gcode})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["has_collision"])
        self.assertEqual(body["collision_line_no"], 3)
        self.assertEqual(body["collision_zone_no"], 100)

        # 无碰撞刀路
        gcode_ok = "G1 X-5 Y50 Z0\nG1 X15 Y50 Z0\n"
        r = client.post("/reviews", json={
            "zones": [{"zone_no": 100, "version": 1}],
            "start": [-5, 50, 0], "tool_radius": 0.0, "gcode": gcode_ok})
        body = r.json()
        self.assertFalse(body["has_collision"])
        self.assertIsNone(body["collision_line_no"])

        # 非法 gcode 拒绝，且不留报告
        existing = client.get("/zones/100").json()
        r = client.post("/reviews", json={
            "zones": [{"zone_no": 100, "version": 1}],
            "start": [0, 0, 0], "tool_radius": 0.0,
            "gcode": "G54 G1 X1\n"})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(client.get("/zones/100").json(), existing)

        # 不存在的禁区/版本
        r = client.post("/reviews", json={
            "zones": [{"zone_no": 999, "version": 1}],
            "start": [0, 0, 0], "tool_radius": 0.0, "gcode": "G1 X1\n"})
        self.assertEqual(r.status_code, 404)
        r = client.post("/reviews", json={
            "zones": [{"zone_no": 100, "version": 9}],
            "start": [0, 0, 0], "tool_radius": 0.0, "gcode": "G1 X1\n"})
        self.assertEqual(r.status_code, 404)

        # 刀具半径非法
        r = client.post("/reviews", json={
            "zones": [{"zone_no": 100, "version": 1}],
            "start": [0, 0, 0], "tool_radius": -1, "gcode": "G1 X1\n"})
        self.assertEqual(r.status_code, 422)

    def test_02_versions_and_immutable_report(self):
        r = client.post("/zones", json={
            "polygon": SQUARE, "z_min": -5, "z_max": 5})
        no = r.json()["zone_no"]
        r = client.post("/reviews", json={
            "zones": [{"zone_no": no, "version": 1}],
            "start": [-5, 5, 0], "tool_radius": 0.0,
            "gcode": "G1 X15 Y5 Z0\n"})
        self.assertTrue(r.json()["has_collision"])
        report_v1 = r.json()["report_id"]

        # 修改 => 不可变新版本（方块平移到 (100,100) 附近）
        far = [[100, 100], [110, 100], [110, 110], [100, 110]]
        r = client.post(f"/zones/{no}/versions",
                        json={"polygon": far, "z_min": -5, "z_max": 5})
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.json()["version"], 2)

        # 旧报告不随禁区更新改变
        old = client.get(f"/reports/{report_v1}").json()
        self.assertEqual(old["used_zone_versions"], {str(no): 1})
        self.assertTrue(old["has_collision"])

        # 同一刀路按 v2 审查 => 无碰撞
        r = client.post("/reviews", json={
            "zones": [{"zone_no": no, "version": 2}],
            "start": [-5, 5, 0], "tool_radius": 0.0,
            "gcode": "G1 X15 Y5 Z0\n"})
        self.assertFalse(r.json()["has_collision"])

        detail = client.get(f"/zones/{no}").json()
        self.assertEqual([v["version"] for v in detail["versions"]], [1, 2])

    def test_03_multiple_zones_first_one_wins(self):
        client.post("/zones", json={"zone_no": 200, "polygon": SQUARE,
                                   "z_min": -5, "z_max": 5})
        poly2 = [[11, 0], [20, 0], [20, 10], [11, 10]]
        client.post("/zones", json={"zone_no": 300, "polygon": poly2,
                                   "z_min": -5, "z_max": 5})
        gcode = "G1 X30 Y5 Z0\n"
        r = client.post("/reviews", json={
            "zones": [{"zone_no": 300, "version": 1},
                      {"zone_no": 200, "version": 1}],
            "start": [-10, 5, 0], "tool_radius": 0.0, "gcode": gcode})
        body = r.json()
        self.assertTrue(body["has_collision"])
        # 路径先进入 200 号禁区（x=10 边早于 300 的 x=11 边）
        self.assertEqual(body["collision_zone_no"], 200)

    def test_04_min_clearance_api(self):
        client.post("/zones", json={"zone_no": 400, "polygon": SQUARE,
                                    "z_min": -5, "z_max": 5})

        # 未启用：响应不含 min_clearance 字段
        r = client.post("/reviews", json={
            "zones": [{"zone_no": 400, "version": 1}],
            "start": [-5, 11, 0], "tool_radius": 0.5,
            "gcode": "G1 X15 Y11 Z0\n"})
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("min_clearance", r.json())
        old_id = r.json()["report_id"]
        self.assertNotIn("min_clearance",
                         client.get(f"/reports/{old_id}").json())

        # 启用：正净间隙 0.5（距边 1 - 半径 .5）
        r = client.post("/reviews", json={
            "zones": [{"zone_no": 400, "version": 1}],
            "start": [-5, 11, 0], "tool_radius": 0.5,
            "gcode": "G1 X15 Y11 Z0\n",
            "include_min_clearance": True})
        body = r.json()
        self.assertEqual(body["min_clearance"], {
            "min_clearance_mm": 0.5, "line_no": 1,
            "tip_position_mm": [0.0, 11.0, 0.0], "zone_no": 400})
        rid = body["report_id"]

        # 查询一致（固化）
        got = client.get(f"/reports/{rid}").json()
        self.assertEqual(got["min_clearance"], body["min_clearance"])

        # 边界接触净间隙为 0
        r = client.post("/reviews", json={
            "zones": [{"zone_no": 400, "version": 1}],
            "start": [-5, 10.5, 0], "tool_radius": 0.5,
            "gcode": "G1 X15 Y10.5 Z0\n",
            "include_min_clearance": True})
        self.assertEqual(r.json()["min_clearance"]["min_clearance_mm"], 0.0)

        # 净间隙可为负，且首个碰撞结论/行号不变
        r = client.post("/reviews", json={
            "zones": [{"zone_no": 400, "version": 1}],
            "start": [-5, 5, 0], "tool_radius": 1.0,
            "gcode": "G1 X15 Y5 Z0\n",
            "include_min_clearance": True})
        body = r.json()
        self.assertTrue(body["has_collision"])
        self.assertEqual(body["collision_line_no"], 1)
        self.assertEqual(body["min_clearance"]["min_clearance_mm"], -1.0)

        # 全程不在任何禁区高度：min_clearance 为 null（字段仍在）
        r = client.post("/reviews", json={
            "zones": [{"zone_no": 400, "version": 1}],
            "start": [-5, 5, 99], "tool_radius": 0.0,
            "gcode": "G1 X15 Y5 Z99\n",
            "include_min_clearance": True})
        body = r.json()
        self.assertIn("min_clearance", body)
        self.assertIsNone(body["min_clearance"])
        rid_null = body["report_id"]
        self.assertIsNone(client.get(f"/reports/{rid_null}")
                          .json()["min_clearance"])

        # 无运动段：按初始刀位，line_no 为 null
        r = client.post("/reviews", json={
            "zones": [{"zone_no": 400, "version": 1}],
            "start": [12, 5, 0], "tool_radius": 0.5,
            "gcode": "G90 ; only modal\n",
            "include_min_clearance": True})
        self.assertEqual(r.json()["min_clearance"], {
            "min_clearance_mm": 1.5, "line_no": None,
            "tip_position_mm": [12.0, 5.0, 0.0], "zone_no": 400})

        # 非法布尔输入：422，且不落报告
        before = client.get("/zones/400").status_code
        r = client.post("/reviews", json={
            "zones": [{"zone_no": 400, "version": 1}],
            "start": [0, 0, 0], "tool_radius": 0.0,
            "gcode": "G1 X1\n", "include_min_clearance": "yes"})
        self.assertEqual(r.status_code, 422)

        # 非法 gcode：422 且不落报告（启用时同样）
        r = client.post("/reviews", json={
            "zones": [{"zone_no": 400, "version": 1}],
            "start": [0, 0, 0], "tool_radius": 0.0,
            "gcode": "G54 G1 X1\n", "include_min_clearance": True})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(before, 200)

        # 旧报告仍无该字段
        self.assertNotIn("min_clearance",
                         client.get(f"/reports/{old_id}").json())

    def test_05_min_clearance_arc_api(self):
        # 整圆 + 圆顶小禁区：入口在弧上
        client.post("/zones", json={"zone_no": 500,
                                    "polygon": [[-1, 9], [1, 9], [1, 11],
                                                [-1, 11]],
                                    "z_min": -5, "z_max": 5})
        gcode = "G90 G21 G17\nG1 X10 Y0 Z0\nG3 X10 Y0 I-10 J0\n"
        r = client.post("/reviews", json={
            "zones": [{"zone_no": 500, "version": 1}],
            "start": SAFE, "tool_radius": 0.0, "gcode": gcode,
            "include_min_clearance": True})
        body = r.json()
        self.assertEqual(body["min_clearance"]["min_clearance_mm"], 0.0)
        self.assertEqual(body["min_clearance"]["line_no"], 3)
        self.assertAlmostEqual(body["min_clearance"]["tip_position_mm"][0],
                               1.0, places=6)
        self.assertAlmostEqual(body["min_clearance"]["tip_position_mm"][1],
                               math.sqrt(99), places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
