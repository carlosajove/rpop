#!/usr/bin/env python
"""Assemble the set-designer report (out/report.html) from the analysis JSONs and figures."""
import base64, json, os
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")

def img(name):
    with open(os.path.join(OUT, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

reach = json.load(open(os.path.join(OUT, "reach_r1_scene_bent.json")))
reach_w = json.load(open(os.path.join(OUT, "reach_r1_scene_bent_waist.json")))
two = json.load(open(os.path.join(OUT, "two_hand_handle.json")))
rk = {}
for fn in ("rake_scenes.json", "rake_scenes_tilt.json"):  # plain-grip run, then the (corrected) angled-grip run
    fp_ = os.path.join(OUT, fn)
    if os.path.exists(fp_):
        for k, v in json.load(open(fp_)).items():
            if fn == "rake_scenes.json" and "tilt" in k:
                continue  # superseded by rake_scenes_tilt.json
            rk[k] = v
rk = dict(sorted(rk.items(), key=lambda kv: ("tilt" in kv[0], kv[0])))
R = reach["hands"]["right"]; G = R["grasp"]["collision_free"]; GW = reach_w["hands"]["right"]["grasp"]["collision_free"]

def best_tilt(L, tilt):
    v = R["rake"][f"{L:.2f}"]["per_tilt"]; best = None
    for key, r in v.items():
        if key.startswith(f"tilt{tilt:+d}_") and r["n_fwd"] and (best is None or r["footprint"]["area_m2"] > best["footprint"]["area_m2"]):
            best = r
    return best

rows = []
for L in (0.6, 0.8, 1.0, 1.2):
    cells = []
    for t in (0, 45, 60, 90):
        b = best_tilt(L, t)
        if b is None:
            cells.append('<td class="no">—</td>')
        else:
            fp = b["footprint"]
            cls = "ok" if fp["area_m2"] >= 0.5 else ("meh" if fp["area_m2"] >= 0.15 else "no")
            cells.append(f'<td class="{cls}"><b>{fp["area_m2"]:.2f} m²</b><br><span class="sub">to {fp["x_max"]:.2f} m ahead · {b["incl_range"][0]:.0f}–{b["incl_range"][1]:.0f}°</span></td>')
    rows.append(f'<tr><th>{L*100:.0f} cm</th>{"".join(cells)}</tr>')
rake_table = "\n".join(rows)

# collision-aware check with the real rake body (tilt 0 / 60 / 90), if available
coll_rows = []
for sc, o in rk.items():
    name = os.path.basename(sc).replace("r1_rake_right_", "").replace(".xml", "")
    L = name.split("_")[0]; tilt = name.split("tilt")[1] if "tilt" in name else "0"
    fp = o.get("footprint")
    coll_rows.append(f'<tr><th>{float(L)*100:.0f} cm · grip {tilt}°</th><td>{o["frac_fwd_floor_ignoring_collisions"]*100:.2f} %</td><td>{o["frac_fwd_floor"]*100:.2f} %</td>'
                     + (f'<td>{fp["area_m2"]:.2f} m² · to {fp["x_max"]:.2f} m</td>' if fp else '<td class="no">not reachable</td>') + '</tr>')
coll_table = "\n".join(coll_rows) if coll_rows else '<tr><td colspan="4">collision-aware run pending</td></tr>'

th = {}
for s in ("0.25", "0.35", "0.45"):
    sm = two["summary"][f"s{s}"]
    th[s] = (sm["frac_rigid_both"] * 100, sm["frac_rigid_loose"] * 100)
tpl = open(os.path.join(HERE, "report_template.html")).read()
vals = dict(
    IMG_WS=img("reach_workspace.png"), IMG_RAKE=img("rake_footprints.png"), IMG_SCENE=img("render_report_hero.png" if os.path.exists(os.path.join(OUT, "render_report_hero.png")) else "render_scene_props_iso.png"),
    RAKE_TABLE=rake_table, COLL_TABLE=coll_table,
    VOL=f"{G['volume_m3']:.2f}", VOL5=f"{G['volume_m3_res5cm']:.2f}", VOLW=f"{GW['volume_m3']:.2f}",
    ZMIN=f"{G['z_min']:.2f}", ZMAX=f"{G['z_max']:.2f}", XMAX=f"{G['x_max']:.2f}", XMAXW=f"{GW['x_max']:.2f}",
    R05=f"{G['reach_at_height']['0.5']['x_max']:.2f}", R07=f"{G['reach_at_height']['0.7']['x_max']:.2f}", R09=f"{G['reach_at_height']['0.9']['x_max']:.2f}", R11=f"{G['reach_at_height']['1.1']['x_max']:.2f}", R13=f"{G['reach_at_height']['1.3']['x_max']:.2f}",
    OVL=f"{reach['two_hand_overlap_grasp']['volume_m3']:.2f}",
    TH25B=f"{th['0.25'][0]:.0f}", TH25L=f"{th['0.25'][1]:.0f}", TH35B=f"{th['0.35'][0]:.0f}", TH35L=f"{th['0.35'][1]:.0f}", TH45B=f"{th['0.45'][0]:.0f}", TH45L=f"{th['0.45'][1]:.0f}",
)
for k, v in vals.items():
    tpl = tpl.replace("{{" + k + "}}", v)
open(os.path.join(OUT, "report.html"), "w").write(tpl)
print("wrote out/report.html", len(tpl) // 1024, "KB")
