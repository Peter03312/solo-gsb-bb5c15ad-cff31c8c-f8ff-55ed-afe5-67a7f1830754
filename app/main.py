"""FastAPI 应用：数控刀路禁入区审查服务。"""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import Session

from .clearance import minimum_clearance
from .collision import Zone, check_path, initial_collision
from .database import Base, engine, get_db
from .gcode import GCodeError, parse_gcode
from .geometry import GeometryError, normalize_polygon, validate_z_range
from . import models
from .models import Report, ZoneVersion
from .schemas import (MinClearanceOut, ReviewRequest, ZoneCreate, ZoneDetailOut,
                      ZoneOut, ZoneVersionCreate, ZoneVersionOut)

app = FastAPI(
    title="数控刀路禁入区审查服务",
    version="1.1.0",
    description="登记毫米制二维多边形禁区（含闭合 Z 高度范围、不可变版本），"
                "对 G-code 刀路做连续碰撞审查并出具不可变报告；"
                "可选计算全程最小水平净间隙（include_min_clearance）。",
)


@app.on_event("startup")
def on_startup() -> None:
    # 容器内由应用直接完成数据库初始化（表结构创建）
    Base.metadata.create_all(bind=engine)
    # 轻量补列：旧库（可选最小净间隙功能上线前）自动补齐新列，不影响既有数据
    _ensure_report_columns()


def _ensure_report_columns() -> None:
    inspector = inspect(engine)
    if "reports" not in inspector.get_table_names():
        return
    existing = {c["name"] for c in inspector.get_columns("reports")}
    dialect = engine.dialect.name
    with engine.begin() as conn:
        if "min_clearance_enabled" not in existing:
            default = "0" if dialect == "sqlite" else "false"
            conn.execute(text(
                "ALTER TABLE reports ADD COLUMN min_clearance_enabled "
                f"BOOLEAN NOT NULL DEFAULT {default}"))
        if "min_clearance" not in existing:
            conn.execute(text(
                "ALTER TABLE reports ADD COLUMN min_clearance JSON"))


@app.exception_handler(GeometryError)
async def geometry_error_handler(request: Request, exc: GeometryError):  # noqa: ARG001
    return JSONResponse(status_code=422,
                        content={"error": "invalid_geometry", "detail": str(exc)})


# ---------- 禁区登记 ----------

@app.post("/zones", response_model=ZoneOut, status_code=201,
          summary="登记新禁区（生成版本 1）")
def create_zone(payload: ZoneCreate, db: Session = Depends(get_db)) -> ZoneOut:
    polygon = normalize_polygon(payload.polygon)
    validate_z_range(payload.z_min, payload.z_max)

    if payload.zone_no is not None:
        exists = db.scalar(select(models.Zone).where(
            models.Zone.zone_no == payload.zone_no))
        if exists:
            raise HTTPException(status_code=409,
                                detail=f"禁区编号 {payload.zone_no} 已存在")
        zone_no = payload.zone_no
    else:
        max_no = db.scalar(select(func.max(models.Zone.zone_no))) or 0
        zone_no = max_no + 1

    zone = models.Zone(zone_no=zone_no, description=payload.description)
    db.add(zone)
    db.flush()
    db.add(ZoneVersion(zone_id=zone.id, version=1, polygon=[list(p) for p in polygon],
                       z_min=payload.z_min, z_max=payload.z_max))
    db.commit()
    db.refresh(zone)
    return ZoneOut(zone_id=zone.id, zone_no=zone.zone_no,
                   description=zone.description, current_version=1,
                   created_at=zone.created_at)


@app.get("/zones", response_model=list[ZoneOut], summary="列出全部禁区")
def list_zones(db: Session = Depends(get_db)) -> list[ZoneOut]:
    zones = db.scalars(select(models.Zone).order_by(models.Zone.zone_no)).all()
    out: list[ZoneOut] = []
    for z in zones:
        cur = db.scalar(select(func.max(ZoneVersion.version)).where(
            ZoneVersion.zone_id == z.id))
        out.append(ZoneOut(zone_id=z.id, zone_no=z.zone_no,
                           description=z.description, current_version=cur or 0,
                           created_at=z.created_at))
    return out


@app.get("/zones/{zone_no}", response_model=ZoneDetailOut,
         summary="查询禁区及其全部不可变版本")
def get_zone(zone_no: int, db: Session = Depends(get_db)) -> ZoneDetailOut:
    zone = db.scalar(select(models.Zone).where(models.Zone.zone_no == zone_no))
    if zone is None:
        raise HTTPException(status_code=404, detail=f"禁区 {zone_no} 不存在")
    versions = db.scalars(
        select(ZoneVersion).where(ZoneVersion.zone_id == zone.id)
        .order_by(ZoneVersion.version)).all()
    return ZoneDetailOut(
        zone_id=zone.id, zone_no=zone.zone_no, description=zone.description,
        current_version=versions[-1].version if versions else 0,
        created_at=zone.created_at,
        versions=[ZoneVersionOut(
            version_id=v.id, version=v.version,
            polygon=[tuple(p) for p in v.polygon],
            z_min=v.z_min, z_max=v.z_max, created_at=v.created_at)
            for v in versions])


@app.post("/zones/{zone_no}/versions", response_model=ZoneVersionOut,
          status_code=201, summary="为禁区登记不可变新版本（修改不改旧版本）")
def add_zone_version(zone_no: int, payload: ZoneVersionCreate,
                     db: Session = Depends(get_db)) -> ZoneVersionOut:
    zone = db.scalar(select(models.Zone).where(models.Zone.zone_no == zone_no))
    if zone is None:
        raise HTTPException(status_code=404, detail=f"禁区 {zone_no} 不存在")

    polygon = normalize_polygon(payload.polygon)
    validate_z_range(payload.z_min, payload.z_max)

    # 行锁保证并发下版本号单调
    db.scalar(
        select(models.Zone).where(models.Zone.id == zone.id).with_for_update())
    latest = db.scalar(select(func.max(ZoneVersion.version)).where(
        ZoneVersion.zone_id == zone.id)) or 0
    ver = ZoneVersion(zone_id=zone.id, version=latest + 1,
                      polygon=[list(p) for p in polygon],
                      z_min=payload.z_min, z_max=payload.z_max)
    db.add(ver)
    db.commit()
    db.refresh(ver)
    return ZoneVersionOut(version_id=ver.id, version=ver.version,
                          polygon=[tuple(p) for p in ver.polygon],
                          z_min=ver.z_min, z_max=ver.z_max,
                          created_at=ver.created_at)


# ---------- 审查 ----------

def _load_zones(db: Session, refs) -> tuple[list[Zone], dict[str, int], dict[int, int]]:
    zones: list[Zone] = []
    used: dict[str, int] = {}
    version_ids: dict[int, int] = {}
    seen: set[int] = set()
    for ref in refs:
        if ref.zone_no in seen:
            raise HTTPException(status_code=422,
                                detail=f"禁区 {ref.zone_no} 在请求中重复")
        seen.add(ref.zone_no)
        zone = db.scalar(select(models.Zone).where(
            models.Zone.zone_no == ref.zone_no))
        if zone is None:
            raise HTTPException(status_code=404,
                                detail=f"禁区 {ref.zone_no} 不存在")
        zv = db.scalar(select(ZoneVersion).where(
            ZoneVersion.zone_id == zone.id,
            ZoneVersion.version == ref.version))
        if zv is None:
            raise HTTPException(
                status_code=404,
                detail=f"禁区 {ref.zone_no} 不存在版本 {ref.version}")
        polygon = normalize_polygon(zv.polygon)  # 防御性再校验
        zones.append(Zone(ref.zone_no, polygon, zv.z_min, zv.z_max))
        used[str(ref.zone_no)] = ref.version
        version_ids[ref.zone_no] = zv.id
    zones.sort(key=lambda z: z.zone_no)
    return zones, used, version_ids


@app.post("/reviews",
          summary="提交刀路审查并生成不可变报告",
          description="碰撞结论始终返回。请求体 `include_min_clearance=true` 时，"
                      "额外计算并在响应 `min_clearance` 中返回全程最小水平净间隙"
                      "（毫米：路径到填充多边形最小距离减刀半径，仅计刀尖 Z 位于"
                      "禁区闭合高度范围的部分；直线/圆弧/整圆连续覆盖，可为负，"
                      "接触为 0；全程不在高度内为 null；无运动段按初始刀位、"
                      "line_no=null）。该字段随报告固化，且不影响首个碰撞结论；"
                      "未启用时请求与报告均不含该字段。")
def review(payload: ReviewRequest,
           db: Session = Depends(get_db)) -> dict:
    if payload.tool_radius < 0:
        raise HTTPException(status_code=422, detail="刀具半径不得为负")

    # 先加载并固化所用禁区版本；任何非法输入都在写报告之前拒绝
    zones, used_versions, _ = _load_zones(db, payload.zones)

    # 解析 G-code；语法/语义错误明确拒绝，不留报告
    try:
        parsed = parse_gcode(payload.gcode, tuple(payload.start))  # type: ignore[arg-type]
    except GCodeError as exc:
        raise HTTPException(status_code=422, detail={
            "error": "invalid_gcode",
            "line_no": exc.line_no,
            "message": str(exc),
        })

    summary = {
        "gcode_lines": parsed.line_count,
        "nonblank_lines": parsed.nonblank_count,
        "motion_count": len(parsed.motions),
        "start_position_mm": list(payload.start),
        "tool_radius_mm": payload.tool_radius,
        "final_position_mm": list(parsed.final_position),
        "gcode_sha256": __sha(payload.gcode),
    }

    collision = check_path(parsed.motions, zones, payload.tool_radius)

    line_no: int | None = None
    zone_no: int | None = None

    if collision is not None:
        line_no = collision.line_no
        zone_no = collision.zone_no
    else:
        # 无运动段碰撞时检查初始刀位点；有运动段时归属首个运动行号
        static_zone = initial_collision(tuple(payload.start),  # type: ignore[arg-type]
                                        zones, payload.tool_radius)
        if static_zone is not None:
            zone_no = static_zone
            line_no = parsed.motions[0].line_no if parsed.motions else None

    has_coll = zone_no is not None

    # 可选：全程最小水平净间隙。仅在请求启用时计算并随报告固化；
    # 结果不影响既有首个碰撞结论与行号。
    min_clr_payload = None
    if payload.include_min_clearance:
        mc = minimum_clearance(parsed.motions, zones,
                               tuple(payload.start),  # type: ignore[arg-type]
                               payload.tool_radius)
        if mc is not None:
            min_clr_payload = MinClearanceOut(
                min_clearance_mm=mc.min_clearance_mm,
                line_no=mc.line_no,
                tip_position_mm=mc.tip_position_mm,
                zone_no=mc.zone_no,
            ).model_dump()

    if has_coll:
        conclusion = "collision"
        message = (f"发现碰撞：G-code 第 {line_no} 行与禁区编号 {zone_no} 冲突") \
            if line_no is not None else \
            f"初始刀位点位于禁区编号 {zone_no} 范围内"
    else:
        conclusion = "no_collision"
        message = "刀路与全部指定禁区版本无碰撞"

    report = Report(
        zone_versions=used_versions,
        input_summary=summary,
        has_collision=has_coll,
        collision_line_no=line_no,
        collision_zone_no=zone_no,
        conclusion=conclusion,
        min_clearance_enabled=payload.include_min_clearance,
        min_clearance=min_clr_payload,
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    body: dict = {
        "report_id": report.id,
        "used_zone_versions": used_versions,
        "input_summary": summary,
        "has_collision": has_coll,
        "collision_line_no": line_no,
        "collision_zone_no": zone_no,
        "conclusion": conclusion,
        "message": message,
    }
    # 仅启用该可选项的请求/报告返回该字段（旧请求与既有报告保持兼容）
    if payload.include_min_clearance:
        body["min_clearance"] = min_clr_payload
    return body


def _report_message(report: Report) -> str:
    if report.has_collision and report.collision_line_no is not None:
        return (f"发现碰撞：G-code 第 {report.collision_line_no} 行与禁区编号 "
                f"{report.collision_zone_no} 冲突")
    if report.has_collision:
        return "初始刀位点位于禁区范围内"
    return "刀路与全部指定禁区版本无碰撞"


@app.get("/reports/{report_id}", summary="查询不可变审查报告")
def get_report(report_id: int, db: Session = Depends(get_db)) -> dict:
    report = db.get(Report, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="报告不存在")
    body = {
        "report_id": report.id,
        "created_at": report.created_at,
        "used_zone_versions": report.zone_versions,
        "input_summary": report.input_summary,
        "has_collision": report.has_collision,
        "collision_line_no": report.collision_line_no,
        "collision_zone_no": report.collision_zone_no,
        "conclusion": report.conclusion,
        "message": _report_message(report),
    }
    # 仅当报告生成时启用了最小净间隙计算，才固化回传该字段；
    # 旧报告（未启用）永不出现该字段，查询结果与生成时一致。
    if getattr(report, "min_clearance_enabled", False):
        body["min_clearance"] = report.min_clearance
    return body


@app.get("/health", summary="健康检查")
def health() -> dict:
    return {"status": "ok"}


def __sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
