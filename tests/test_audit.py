"""端到端与单元测试。使用 SQLite 覆盖数据库，无需 PostgreSQL。"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

os.environ["DATABASE_URL"] = "sqlite:///" + tempfile.mktemp(suffix=".db")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app import gcode as gc  # noqa: E402
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
