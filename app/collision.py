"""刀路-禁入区碰撞引擎。

判定条件（连续运动、按程序顺序）：
刀位点 Z 位于禁区闭合高度范围 [z_min, z_max] 内，
且水平路径与填充多边形的距离不大于刀具半径时 => 碰撞；
边界接触同样计为碰撞。

线段运动不做端点采样：对每条多边形边建立
“运动点到边距离 ≤ 刀半径”的二次不等式，精确求解参数 t 区间，
再与 Z 高度区间求交，取全程最早 t。
圆弧运动：弧-边距离精确求值 + 子弧二分定位首个碰撞进度。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import (
    EPS, CONTACT_EPS, XY,
    cross, dist, point_in_polygon, point_polygon_distance,
    earliest_arc_hit, subarc_hits,
)
from .gcode import Motion


@dataclass(frozen=True)
class Zone:
    zone_no: int
    polygon: list[XY]
    z_min: float
    z_max: float


@dataclass
class Collision:
    line_no: int
    zone_no: int
    progress: float          # 该段运动上的碰撞进度（t 或角度 s），仅用于排序
    order: int = 0           # 运动段序号


# ---------- 区间工具 ----------

def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """合并重叠/相接区间。"""
    if not intervals:
        return []
    intervals = sorted((max(0.0, a), min(1.0, b)) for a, b in intervals
                       if b >= a - EPS)
    out: list[list[float]] = []
    for a, b in intervals:
        if out and a <= out[-1][1] + EPS:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _intersect(iv1: list[tuple[float, float]],
               iv2: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    i = j = 0
    while i < len(iv1) and j < len(iv2):
        a = max(iv1[i][0], iv2[j][0])
        b = min(iv1[i][1], iv2[j][1])
        if a <= b + EPS:
            out.append((a, b))
        if iv1[i][1] < iv2[j][1]:
            i += 1
        else:
            j += 1
    return out


def _solve_quad_le(aa: float, bb: float, cc: float,
                   k: float) -> list[tuple[float, float]]:
    """求解 aa t^2 + bb t + cc <= k 的实数区间。"""
    A = aa
    B = bb
    C = cc - k
    if abs(A) <= 1e-18:
        if abs(B) <= 1e-18:
            return [(-math.inf, math.inf)] if C <= CONTACT_EPS else []
        t = -C / B
        if B > 0:
            return [(-math.inf, t)]
        return [(t, math.inf)]
    disc = B * B - 4.0 * A * C
    if disc < -EPS:
        if A < 0:
            return [(-math.inf, math.inf)]
        return []
    disc = max(0.0, disc)
    sq = math.sqrt(disc)
    t1 = (-B - sq) / (2.0 * A)
    t2 = (-B + sq) / (2.0 * A)
    lo, hi = sorted((t1, t2))
    if A > 0:
        return [(lo, hi)]
    return [(-math.inf, lo), (hi, math.inf)]


# ---------- 线段运动 ----------

def _z_window(m: Motion, zone: Zone) -> list[tuple[float, float]]:
    """运动参数 t∈[0,1] 中，刀尖 Z 位于禁区闭合高度内的区间。"""
    z0, z1 = m.start[2], m.end[2]
    dz = z1 - z0
    if abs(dz) <= EPS:
        return [(0.0, 1.0)] if zone.z_min - EPS <= z0 <= zone.z_max + EPS else []
    t_lo = (zone.z_min - z0) / dz
    t_hi = (zone.z_max - z0) / dz
    lo, hi = sorted((t_lo, t_hi))
    lo = max(0.0, lo)
    hi = min(1.0, hi)
    # 闭合区间：端点含边界
    if lo <= hi + EPS:
        return [(lo, hi)]
    return []


def _line_capsule_interval(p0: XY, p1: XY, a: XY, b: XY,
                           tool_r: float) -> list[tuple[float, float]]:
    """运动点 P(t)=p0+t(p1-p0) 到固定边 a-b 的距离 ≤ tool_r 的 t 区间。

    P 到无限直线的垂足参数 tc 落在 [0,1] 时用二次不等式；
    否则距离由近端点决定（|P-e| ≤ r 的圆域也是二次不等式）。
    两种情况合并即可精确覆盖整个线段的膨胀区域（胶囊）。
    """
    d = (p1[0] - p0[0], p1[1] - p0[1])
    ab = (b[0] - a[0], b[1] - a[1])
    length2 = ab[0] * ab[0] + ab[1] * ab[1]
    if length2 <= EPS * EPS:
        # 退化边：按点处理
        dx = p0[0] - a[0]
        dy = p0[1] - a[1]
        return _merge(_solve_quad_le(dot2(d, d), 2.0 * (d[0] * dx + d[1] * dy),
                                     dx * dx + dy * dy, tool_r ** 2))
    # P(t) 在边无限线上的投影位置 s(t)
    ap = (p0[0] - a[0], p0[1] - a[1])
    # s(t) = (ab·(ap + t d)) / |ab|^2 = s0 + sd t
    s0 = (ab[0] * ap[0] + ab[1] * ap[1]) / length2
    sd = (ab[0] * d[0] + ab[1] * d[1]) / length2

    intervals: list[tuple[float, float]] = []
    r2 = (tool_r + CONTACT_EPS) ** 2

    if sd != 0.0:
        t_a = -s0 / sd
        t_b = (1.0 - s0) / sd
        t0, t1 = sorted((t_a, t_b))
        slab = [(max(0.0, t0), min(1.0, t1))] if t0 <= 1.0 + EPS and t1 >= -EPS else []
    else:
        slab = [(0.0, 1.0)] if 0.0 - EPS <= s0 <= 1.0 + EPS else []

    if slab:
        # 垂足段：垂直距离平方 = |ap + t d - s(t) ab|^2
        # w(t) = ap - s0 ab + t (d - sd ab)
        w0 = (ap[0] - s0 * ab[0], ap[1] - s0 * ab[1])
        wd = (d[0] - sd * ab[0], d[1] - sd * ab[1])
        ivs = _solve_quad_le(dot2(wd, wd),
                             2.0 * (w0[0] * wd[0] + w0[1] * wd[1]),
                             dot2(w0, w0), r2)
        for iv in ivs:
            cut = _intersect([iv], slab)
            intervals.extend(cut)

    # 两端点圆域（处理垂足在边外的情形）
    for e in (a, b):
        dx = p0[0] - e[0]
        dy = p0[1] - e[1]
        intervals.extend(
            _solve_quad_le(dot2(d, d), 2.0 * (d[0] * dx + d[1] * dy),
                           dx * dx + dy * dy, r2))
    return _merge(intervals)


def dot2(a: XY, b: XY) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _line_inside_intervals(p0: XY, p1: XY,
                           poly: list[XY]) -> list[tuple[float, float]]:
    """水平路径（参数 t）严格位于填充多边形内部/边界的区间。

    收集运动线段与各边的真相交参数作为事件点，事件点之间内外状态恒定；
    事件点本身（边界接触）也并入结果。
    """
    d = (p1[0] - p0[0], p1[1] - p0[1])
    if dot2(d, d) <= EPS * EPS:
        # 水平方向静止
        return [(0.0, 1.0)] if point_in_polygon(p0, poly) else []

    events: set[float] = {0.0, 1.0}
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = (b[0] - a[0], b[1] - a[1])
        denom = cross(d, ab)
        if abs(denom) <= EPS:
            continue
        t = cross((a[0] - p0[0], a[1] - p0[1]), ab) / denom
        u = cross((a[0] - p0[0], a[1] - p0[1]), d) / denom
        if -EPS <= t <= 1.0 + EPS and -EPS <= u <= 1.0 + EPS:
            events.add(max(0.0, min(1.0, t)))

    evs = sorted(events)
    result: list[tuple[float, float]] = []
    for k in range(len(evs) - 1):
        t_lo, t_hi = evs[k], evs[k + 1]
        if t_hi - t_lo <= EPS:
            # 事件点自身（边界）计入
            result.append((t_lo, t_lo))
            continue
        t_mid = (t_lo + t_hi) * 0.5
        pm = (p0[0] + d[0] * t_mid, p0[1] + d[1] * t_mid)
        if point_in_polygon(pm, poly):
            result.append((t_lo, t_hi))
    # 端点边界检查（point_in_polygon 已在采样点覆盖，此处显式加入事件点）
    for t in (0.0, 1.0):
        p = (p0[0] + d[0] * t, p0[1] + d[1] * t)
        if point_in_polygon(p, poly):
            result.append((t, t))
    return _merge(result)


def _vertex_passthrough_events(p0: XY, p1: XY,
                               poly: list[XY]) -> list[tuple[float, float]]:
    """运动路径恰好经过多边形顶点的参数点（顶点属于多边形，必为碰撞点）。

    覆盖“折线穿单个顶点”这类退化为单个参数点的接触，
    该点可能被一般区间运算当作零测事件漏掉。
    """
    d = (p1[0] - p0[0], p1[1] - p0[1])
    if dot2(d, d) <= EPS * EPS:
        return [(0.0, 0.0)] if point_in_polygon(p0, poly) else []
    events: list[tuple[float, float]] = []
    for v in poly:
        ap = (v[0] - p0[0], v[1] - p0[1])
        # ap = t d 共线
        if abs(cross(ap, d)) > EPS:
            continue
        t = (ap[0] * d[0] + ap[1] * d[1]) / dot2(d, d)
        if -EPS <= t <= 1.0 + EPS:
            t = max(0.0, min(1.0, t))
            events.append((t, t))
    return events


def _line_earliest(m: Motion, zone: Zone, tool_r: float) -> float | None:
    """线段运动对单个禁区的最早碰撞参数 t，无碰撞返回 None。"""
    zwin = _z_window(m, zone)
    if not zwin:
        return None
    p0 = (m.start[0], m.start[1])
    p1 = (m.end[0], m.end[1])

    poly = zone.polygon
    hit_t: list[tuple[float, float]] = list(_line_inside_intervals(p0, p1, poly))
    hit_t.extend(_vertex_passthrough_events(p0, p1, poly))

    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        hit_t.extend(_line_capsule_interval(p0, p1, a, b, tool_r))

    hit_t = _merge(hit_t)
    hit_t = _intersect(hit_t, zwin)
    if not hit_t:
        return None
    return min(a for a, _ in hit_t)


# ---------- 圆弧运动 ----------

def _arc_earliest(m: Motion, zone: Zone, tool_r: float) -> float | None:
    z0 = m.start[2]
    if not (zone.z_min - EPS <= z0 <= zone.z_max + EPS):
        return None
    if not subarc_hits(m.center, m.radius, m.theta0, m.sweep,
                       zone.polygon, tool_r):
        return None
    return earliest_arc_hit(m.center, m.radius, m.theta0, m.sweep,
                            zone.polygon, tool_r)


# ---------- 总入口 ----------

def check_path(motions: list[Motion], zones: list[Zone],
               tool_radius: float) -> Collision | None:
    """按程序顺序检查全部连续运动，返回首个碰撞。

    首个碰撞：运动段序号最小；同一段内碰撞进度最小；
    进度相同取禁区编号最小。
    """
    best: Collision | None = None
    for order, m in enumerate(motions):
        for zone in zones:  # zones 已按 zone_no 排序
            if m.kind == "line":
                prog = _line_earliest(m, zone, tool_radius)
            else:
                prog = _arc_earliest(m, zone, tool_radius)
            if prog is None:
                continue
            cand = Collision(m.line_no, zone.zone_no, prog, order)
            if best is None or _earlier(cand, best):
                best = cand
    return best


def _earlier(a: Collision, b: Collision) -> bool:
    if a.order != b.order:
        return a.order < b.order
    if abs(a.progress - b.progress) > EPS:
        return a.progress < b.progress
    return a.zone_no < b.zone_no


def initial_collision(pos: tuple[float, float, float],
                      zones: list[Zone], tool_radius: float) -> int | None:
    """初始刀位点本身是否已在某禁区（用于静止/首点判定），返回禁区号。"""
    x, y, z = pos
    for zone in zones:
        if zone.z_min - EPS <= z <= zone.z_max + EPS:
            if point_polygon_distance((x, y), zone.polygon) <= tool_radius + CONTACT_EPS:
                return zone.zone_no
    return None
