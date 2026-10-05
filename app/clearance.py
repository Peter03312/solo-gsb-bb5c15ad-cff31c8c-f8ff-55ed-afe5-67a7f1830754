"""全程最小水平净间隙（可选审查项）。

对每段运动，**仅取刀尖 Z 位于某禁区闭合高度范围 [z_min, z_max] 内的部分**，
求连续 XY 路径到该禁区**填充多边形**的最小距离再减去刀具半径：

    clearance = min over P∈路径∩高度窗口 : d(P, polygon_filled) - tool_radius

- 多边形内部/边界的点距离为 0，故净间隙可为负（穿越禁区时为 -tool_radius）；
- 刀具轮廓恰好接触禁区边界（路径距填充区恰为刀半径）=> 净间隙为 0；
- 直线、圆弧、整圆均连续求解（非端点采样），Z 中途进入高度范围同样覆盖；
- 全程不在任何禁区高度内 => 无结果；无运动段 => 按初始刀位计算。

等值（毫米，绝对容差）时按
“运动顺序 → 段内进度（直线 t / 圆弧角进度 s）→ 禁区编号”选取。

本模块**只计算最小净间隙**，不影响既有首个碰撞结论与行号规则。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .collision import (
    Zone, _intersect, _line_inside_intervals, _merge,
    _vertex_passthrough_events, _z_window,
)
from .geometry import (
    CONTACT_EPS, EPS, XY,
    arc_segment_distance_candidates, point_in_polygon,
    point_polygon_distance, point_segment_distance,
)
from .gcode import Motion

# 圆弧入口检测：均匀采样的区间划分数（再二分定位）
_ENTRY_SAMPLES = 4096
# 边界净间隙数值归零阈值（毫米）
ZERO_TOL = CONTACT_EPS


@dataclass(frozen=True)
class ClearanceResult:
    """最小净间隙结果（坐标/净间隙单位：毫米）。"""
    min_clearance_mm: float
    line_no: int | None          # 对应 G-code 原行号；仅初始刀位时为 None
    tip_position_mm: tuple[float, float, float]
    zone_no: int
    order: int                   # 运动段序号（-1 表示初始刀位），仅用于排序
    progress: float              # 段内进度（直线 t / 圆弧角进度 s），仅用于排序


# ---------- 通用 ----------

def _clamp_zero(v: float) -> float:
    """边界接触数值归零：|v| 小于接触容差时按 0 返回。"""
    return 0.0 if abs(v) <= ZERO_TOL else v


def _better(a: ClearanceResult, b: ClearanceResult | None) -> bool:
    """a 是否优先于当前最优 b：净间隙更小；等值按顺序/进度/禁区号。"""
    if b is None:
        return True
    if abs(a.min_clearance_mm - b.min_clearance_mm) > EPS:
        return a.min_clearance_mm < b.min_clearance_mm
    if a.order != b.order:
        return a.order < b.order
    if abs(a.progress - b.progress) > EPS:
        return a.progress < b.progress
    return a.zone_no < b.zone_no


# ---------- 线段运动 ----------

def _line_inside_window(p0: XY, p1: XY, m: Motion,
                        zone: Zone) -> list[tuple[float, float]]:
    """高度窗口与路径“位于填充多边形内/边界”参数区间的交。"""
    zwin = _z_window(m, zone)
    if not zwin:
        return []
    inside = list(_line_inside_intervals(p0, p1, zone.polygon))
    inside.extend(_vertex_passthrough_events(p0, p1, zone.polygon))
    return _intersect(_merge(inside), zwin)


def _line_edge_min_t(p0: XY, p1: XY, a: XY, b: XY,
                     window: list[tuple[float, float]]) -> list[float]:
    """运动参数 t 上，P(t) 到固定边 a-b 的点-线段距离达到极小的候选 t。

    在高度窗口的每个子区间内，距离平方按“垂足在边上 / 靠近 a 端 / 靠近 b 端”
    分段为至多两段二次式，收集：子区间端点、各段内部驻点（夹到段内）。
    """
    d = (p1[0] - p0[0], p1[1] - p0[1])
    ab = (b[0] - a[0], b[1] - a[1])
    length2 = ab[0] ** 2 + ab[1] ** 2
    ts: list[float] = []
    for lo, hi in window:
        ts.extend((lo, hi))
        if length2 <= EPS * EPS:
            # 退化边：P 到固定点距离的二次式
            denom = d[0] ** 2 + d[1] ** 2
            if denom > EPS * EPS:
                t = ((a[0] - p0[0]) * d[0] + (a[1] - p0[1]) * d[1]) / denom
                if lo - EPS <= t <= hi + EPS:
                    ts.append(min(hi, max(lo, t)))
            continue
        ap = (p0[0] - a[0], p0[1] - a[1])
        s0 = (ab[0] * ap[0] + ab[1] * ap[1]) / length2
        sd = (ab[0] * d[0] + ab[1] * d[1]) / length2
        # 垂足参数 s(t) 到达 0 / 1 的分界点
        cuts: list[float] = []
        if abs(sd) > EPS:
            for target in (0.0, 1.0):
                tc = (target - s0) / sd
                if lo + EPS < tc < hi - EPS:
                    cuts.append(tc)
        cuts.sort()
        bounds = [lo] + cuts + [hi]
        for i in range(len(bounds) - 1):
            qlo, qhi = bounds[i], bounds[i + 1]
            s_mid = s0 + sd * 0.5 * (qlo + qhi)
            if 0.0 - EPS <= s_mid <= 1.0 + EPS:
                # 垂足段：w(t)=ap-s0 ab + t(d-sd ab)，驻点 w·wd=0
                w0 = (ap[0] - s0 * ab[0], ap[1] - s0 * ab[1])
                wd = (d[0] - sd * ab[0], d[1] - sd * ab[1])
                ww = wd[0] ** 2 + wd[1] ** 2
                if ww > EPS * EPS:
                    t = -(w0[0] * wd[0] + w0[1] * wd[1]) / ww
                    if qlo - EPS <= t <= qhi + EPS:
                        ts.append(min(qhi, max(qlo, t)))
            else:
                # 端点段：到近端顶点的距离驻点
                e = a if s_mid < 0.0 else b
                denom = d[0] ** 2 + d[1] ** 2
                if denom > EPS * EPS:
                    t = ((e[0] - p0[0]) * d[0] + (e[1] - p0[1]) * d[1]) / denom
                    if qlo - EPS <= t <= qhi + EPS:
                        ts.append(min(qhi, max(qlo, t)))
    return ts


def _line_xyz(m: Motion, t: float) -> tuple[float, float, float]:
    return (m.start[0] + (m.end[0] - m.start[0]) * t,
            m.start[1] + (m.end[1] - m.start[1]) * t,
            m.start[2] + (m.end[2] - m.start[2]) * t)


def _line_min_clearance(m: Motion, zone: Zone, order: int,
                        tool_r: float) -> ClearanceResult | None:
    """单段直线运动对单个禁区的最小净间隙。"""
    zwin = _z_window(m, zone)
    if not zwin:
        return None
    p0 = (m.start[0], m.start[1])
    p1 = (m.end[0], m.end[1])
    window = _merge(zwin)
    poly = zone.polygon

    # 路径有位于填充多边形内部/边界的部分 => 净间隙 -tool_r（负值，最先点胜出）
    inside = _line_inside_window(p0, p1, m, zone)
    if inside:
        t0 = min(a for a, _ in inside)
        px, py, pz = _line_xyz(m, t0)
        return ClearanceResult(_clamp_zero(-tool_r), m.line_no,
                               (px, py, pz), zone.zone_no, order, t0)

    # 路径全程在多边形外：距离 = 到各边距离的逐点最小，收集解析极小候选
    best: ClearanceResult | None = None
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for t in _line_edge_min_t(p0, p1, a, b, window):
            t = min(1.0, max(0.0, t))
            px, py, pz = _line_xyz(m, t)
            d = min(point_segment_distance((px, py), poly[j],
                                           poly[(j + 1) % n])
                    for j in range(n))
            cand = ClearanceResult(_clamp_zero(d - tool_r), m.line_no,
                                   (px, py, pz), zone.zone_no, order, t)
            if _better(cand, best):
                best = cand
    return best


# ---------- 圆弧运动 ----------

def _arc_xyz(m: Motion, s: float) -> tuple[float, float, float]:
    theta = m.theta0 + m.sweep * s
    return (m.center[0] + m.radius * math.cos(theta),
            m.center[1] + m.radius * math.sin(theta),
            m.start[2])


def _arc_entry_progress(m: Motion, poly: list[XY]) -> float | None:
    """弧从多边形外进入多边形（含边界）的最早进度 s；全程在外返回 None。

    均匀采样定位首个含入口的小区间，再二分边界，避免漏掉“经过单个顶点”
    等零测入口。
    """
    step = 1.0 / _ENTRY_SAMPLES

    def inside_at(s: float) -> bool:
        s = min(1.0, max(0.0, s))
        x, y, _ = _arc_xyz(m, s)
        return point_in_polygon((x, y), poly)

    if inside_at(0.0):
        return 0.0
    k_first: int | None = None
    for k in range(_ENTRY_SAMPLES):
        if inside_at((k + 1) * step) or inside_at((k + 0.5) * step):
            k_first = k
            break
    if k_first is None:
        return None

    lo, hi = k_first * step, min(1.0, (k_first + 1) * step)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if inside_at(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _arc_min_clearance(m: Motion, zone: Zone, order: int,
                       tool_r: float) -> ClearanceResult | None:
    """单段圆弧运动（恒定 Z）对单个禁区的最小净间隙。"""
    z0 = m.start[2]
    if not (zone.z_min - EPS <= z0 <= zone.z_max + EPS):
        return None
    poly = zone.polygon

    # 弧进入多边形内部/边界 => 净间隙 -tool_r，入口即最小点
    entry = _arc_entry_progress(m, poly)
    if entry is not None:
        return ClearanceResult(_clamp_zero(-tool_r), m.line_no,
                               _arc_xyz(m, entry), zone.zone_no, order, entry)

    # 全程在多边形外：逐边收集解析最近进度候选，统一用“弧点到填充多边形
    # 真实距离”打分（到单条边的端点径向候选可能落在其他边上/多边形内）。
    ss: set[float] = {0.0, 1.0}
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for s, _d in arc_segment_distance_candidates(
                m.center, m.radius, m.theta0, m.sweep, a, b):
            ss.add(round(min(1.0, max(0.0, s)), 12))

    best: ClearanceResult | None = None
    for s in ss:
        x, y, _z = _arc_xyz(m, s)
        d = point_polygon_distance((x, y), poly)
        cand = ClearanceResult(_clamp_zero(d - tool_r), m.line_no,
                               _arc_xyz(m, s), zone.zone_no, order, s)
        if _better(cand, best):
            best = cand
    return best


# ---------- 总入口 ----------

def minimum_clearance(motions: list[Motion], zones: list[Zone],
                      start: tuple[float, float, float],
                      tool_radius: float) -> ClearanceResult | None:
    """全程最小水平净间隙；全程不在任何禁区高度内返回 None。

    motions 按程序顺序，zones 按 zone_no 升序；无运动段按初始刀位计算。
    """
    best: ClearanceResult | None = None

    if not motions:
        x, y, z = start
        for zone in zones:
            if not (zone.z_min - EPS <= z <= zone.z_max + EPS):
                continue
            d = point_polygon_distance((x, y), zone.polygon)
            cand = ClearanceResult(_clamp_zero(d - tool_radius), None,
                                   (float(x), float(y), float(z)),
                                   zone.zone_no, -1, 0.0)
            if _better(cand, best):
                best = cand
        return best

    for order, m in enumerate(motions):
        for zone in zones:
            if m.kind == "line":
                cand = _line_min_clearance(m, zone, order, tool_radius)
            else:
                cand = _arc_min_clearance(m, zone, order, tool_radius)
            if cand is not None and _better(cand, best):
                best = cand
    return best
