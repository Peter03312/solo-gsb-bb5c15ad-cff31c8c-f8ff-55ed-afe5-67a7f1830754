#!/usr/bin/env bash
# 数控刀路禁入区审查服务调用示例
# 用法: bash examples/curl_examples.sh
set -euo pipefail

BASE="${BASE:-http://localhost:8080}"

echo "== 健康检查 =="
curl -s "$BASE/health"; echo

echo "== 1) 登记禁区 1：10x10 mm 方块，Z 闭合范围 [-5, 5] =="
curl -s -X POST "$BASE/zones" \
  -H 'Content-Type: application/json' \
  -d '{
        "zone_no": 1,
        "polygon": [[0,0],[10,0],[10,10],[0,10]],
        "z_min": -5,
        "z_max": 5,
        "description": "基座（毫米）"
      }'; echo

echo "== 2) 查看禁区及版本 =="
curl -s "$BASE/zones/1"; echo

echo "== 3) 提交审查：横穿禁区（应当碰撞，首个碰撞在第 2 行） =="
curl -s -X POST "$BASE/reviews" \
  -H 'Content-Type: application/json' \
  -d '{
        "zones": [{"zone_no": 1, "version": 1}],
        "start": [-5, 5, 0],
        "tool_radius": 0.0,
        "gcode": "G21 G90 G17\nG1 X15 Y5 Z0\n"
      }'; echo

echo "== 4) 提交审查：从禁区上方经过（无碰撞） =="
curl -s -X POST "$BASE/reviews" \
  -H 'Content-Type: application/json' \
  -d '{
        "zones": [{"zone_no": 1, "version": 1}],
        "start": [-5, 50, 0],
        "tool_radius": 0.0,
        "gcode": "G1 X15 Y50 Z0\n"
      }'; echo

echo "== 5) 非法指令（G54 坐标系）必须被拒绝，且不产生报告 =="
curl -s -o /dev/null -w "HTTP %{http_code}\n" -X POST "$BASE/reviews" \
  -H 'Content-Type: application/json' \
  -d '{
        "zones": [{"zone_no": 1, "version": 1}],
        "start": [0,0,0],
        "tool_radius": 0.0,
        "gcode": "G54 G1 X1\n"
      }'

echo "== 6) 登记禁区新版本（几何平移到远处），旧版本保持不变 =="
curl -s -X POST "$BASE/zones/1/versions" \
  -H 'Content-Type: application/json' \
  -d '{
        "polygon": [[100,100],[110,100],[110,110],[100,110]],
        "z_min": -5,
        "z_max": 5
      }'; echo

echo "== 7) 同一刀路按新版本审查（无碰撞） =="
curl -s -X POST "$BASE/reviews" \
  -H 'Content-Type: application/json' \
  -d '{
        "zones": [{"zone_no": 1, "version": 2}],
        "start": [-5, 5, 0],
        "tool_radius": 0.0,
        "gcode": "G1 X15 Y5 Z0\n"
      }'; echo

echo "== 8) 旧报告内容不随后续禁区更新改变 =="
curl -s "$BASE/reports/1"; echo

echo "== 9) 可选：启用全程最小水平净间隙（距边 1mm，刀半径 0.5 => 净间隙 0.5mm） =="
curl -s -X POST "$BASE/reviews" \
  -H 'Content-Type: application/json' \
  -d '{
        "zones": [{"zone_no": 1, "version": 1}],
        "start": [-5, 11, 0],
        "tool_radius": 0.5,
        "gcode": "G1 X15 Y11 Z0",
        "include_min_clearance": true
      }'; echo

echo "== 10) 边界接触 => 最小净间隙为 0 =="
curl -s -X POST "$BASE/reviews" \
  -H 'Content-Type: application/json' \
  -d '{
        "zones": [{"zone_no": 1, "version": 1}],
        "start": [-5, 10.5, 0],
        "tool_radius": 0.5,
        "gcode": "G1 X15 Y10.5 Z0",
        "include_min_clearance": true
      }'; echo

echo "== 11) 圆弧整圆 + 启用净间隙（入口在弧上） =="
curl -s -X POST "$BASE/reviews" \
  -H 'Content-Type: application/json' \
  -d '{
        "zones": [{"zone_no": 1, "version": 1}],
        "start": [-50, -50, 0],
        "tool_radius": 0.0,
        "gcode": "G90 G21 G17\nG1 X10 Y0 Z0\nG3 X10 Y0 I-10 J0",
        "include_min_clearance": true
      }'; echo

echo "== 12) 未启用时响应与 GET 报告均不含 min_clearance 字段（向后兼容） =="
curl -s -X POST "$BASE/reviews" \
  -H 'Content-Type: application/json' \
  -d '{
        "zones": [{"zone_no": 1, "version": 1}],
        "start": [-5, 50, 0],
        "tool_radius": 0.0,
        "gcode": "G1 X15 Y50 Z0"
      }' | python3 -c 'import json,sys; b=json.load(sys.stdin); print("has min_clearance key:", "min_clearance" in b)'
