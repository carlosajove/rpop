#!/usr/bin/env python
"""Reachable workspace of the R1 hands (standing, legs fixed) and rake-to-floor check.

Method: uniform random sampling of the arm joint space (5 joints per arm, optionally + waist yaw), forward
kinematics with MuJoCo, self/ground collision check, then voxelisation (default 2 cm) of the reachable points.

Points reported per hand
  ee     : xr_teleoperate IK target (0.20 m along the wrist roll axis; ~mid palm)
  grasp  : centre of a handle held in a power grasp (2.5 cm off the palm, 5.5 cm up from the hand base)

Rake check: a rake handle rigidly held in the grasp, handle axis across the palm (perpendicular to the fingers),
optionally tilted in the palm plane by --tilt degrees (diagonal grip).  For each handle length the rake head
position is grasp + L * handle_dir; the head "reaches the floor" when it is within `floor_tol` of z = 0 and the
handle is inclined at least `min_incl` degrees above the floor (a flat handle cannot pull material).

Usage
  python reach_analysis.py --scene out/r1_scene_bent.xml --n 300000 --waist
"""
import argparse, json, os, time
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll"]


def arm_joints(side):
    return [f"{side}_{j}_joint" for j in ARM]


def chain_bodies(m, root_body):
    """All body ids in the subtree rooted at root_body (inclusive)."""
    ids = [root_body]
    for b in range(m.nbody):
        p = b
        while p != 0:
            if p == root_body:
                ids.append(b); break
            p = m.body_parentid[p]
    return set(ids)


class Sampler:
    def __init__(self, scene, use_waist=False, seed=0):
        self.m = mujoco.MjModel.from_xml_path(scene)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self.q0 = self.d.qpos.copy()
        self.rng = np.random.default_rng(seed)
        self.use_waist = use_waist
        m = self.m
        # geoms per chain, to classify contacts
        self.chain = {}
        for side in ("left", "right"):
            bodies = chain_bodies(m, m.body(f"{side}_shoulder_pitch_link").id)
            self.chain[side] = {g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies}
        self.floor = m.geom("floor").id
        self.feet = {g for g in range(m.ngeom) if "ankle" in m.body(m.geom_bodyid[g]).name}
        # the rake head/tines are supposed to touch the floor: never count that as a collision
        self.floor_ok = self.feet | {g for g in range(m.ngeom) if "rake" in m.geom(g).name}
        # baseline contacts at the keyframe (mesh overlaps within the body) are ignored everywhere
        mujoco.mj_forward(m, self.d)
        self.baseline = {(min(c.geom1, c.geom2), max(c.geom1, c.geom2)) for c in self.d.contact[: self.d.ncon]}

    def joint_ids(self, side):
        names = arm_joints(side) + (["waist_yaw_joint"] if self.use_waist else [])
        return [self.m.joint(n).qposadr[0] for n in names], [self.m.joint(n).range for n in names]

    def in_collision(self, side_moving):
        """True if a geom of the moving chain (or the rake) touches something outside that chain, or anything
        touches the floor other than the feet."""
        d, m = self.d, self.m
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            key = (min(g1, g2), max(g1, g2))
            if key in self.baseline:
                continue
            if self.floor in key:
                other = g2 if g1 == self.floor else g1
                if other in self.floor_ok:
                    continue
                return True
            in1 = g1 in self.chain[side_moving]; in2 = g2 in self.chain[side_moving]
            if in1 and in2:
                continue  # intra-arm mesh overlap, ignore
            if in1 or in2:
                return True
            # contact between two non-moving parts (e.g. other arm vs torso) -> baseline noise
        return False

    def sample(self, side, n, sites, extra=None):
        """Sample n joint configs; returns dict with q, positions and rotation matrices of sites, collision flags."""
        m, d = self.m, self.d
        qadr, ranges = self.joint_ids(side)
        lo = np.array([r[0] for r in ranges]); hi = np.array([r[1] for r in ranges])
        qs = self.rng.uniform(lo, hi, size=(n, len(qadr)))
        sids = [m.site(s).id for s in sites]
        pos = np.zeros((n, len(sids), 3)); rot = np.zeros((n, len(sids), 3, 3)); coll = np.zeros(n, bool)
        d.qpos[:] = self.q0
        t0 = time.time()
        for i in range(n):
            d.qpos[qadr] = qs[i]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_collision(m, d)
            coll[i] = self.in_collision(side)
            pos[i] = d.site_xpos[sids]; rot[i] = d.site_xmat[sids].reshape(-1, 3, 3)
        d.qpos[:] = self.q0; mujoco.mj_forward(m, d)
        return dict(q=qs, pos=pos, rot=rot, coll=coll, dt=time.time() - t0, qnames=arm_joints(side) + (["waist_yaw_joint"] if self.use_waist else []))


def voxelize(points, res):
    idx = np.floor(points / res).astype(np.int64)
    return np.unique(idx, axis=0)


def voxel_stats(points, res):
    v = voxelize(points, res)
    out = dict(n_voxels=int(len(v)), volume_m3=float(len(v) * res ** 3))
    for r2 in (0.03, 0.05):
        out[f"volume_m3_res{int(r2*100)}cm"] = float(len(voxelize(points, r2)) * r2 ** 3)
    return out


def extents(p):
    return dict(x_min=float(p[:, 0].min()), x_max=float(p[:, 0].max()), y_min=float(p[:, 1].min()), y_max=float(p[:, 1].max()),
                z_min=float(p[:, 2].min()), z_max=float(p[:, 2].max()))


def reach_at_heights(p, res, heights=(0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3)):
    out = {}
    for h in heights:
        sel = p[np.abs(p[:, 2] - h) < res]
        if len(sel) == 0:
            out[f"{h:.1f}"] = None; continue
        out[f"{h:.1f}"] = dict(x_max=float(sel[:, 0].max()), x_min=float(sel[:, 0].min()), y_min=float(sel[:, 1].min()), y_max=float(sel[:, 1].max()),
                               area_m2=float(len(voxelize(sel[:, :2], res)) * res ** 2))
    return out


def rake_check(grasp_pos, grasp_rot, lengths, tilts_deg, floor_tol=0.05, min_incl=25.0, res=0.02):
    """For each length and tilt: rake head = grasp + L * dir, dir = -(cos t * y_hand + sin t * z_hand)."""
    y = grasp_rot[:, :, 1]; z = grasp_rot[:, :, 2]
    results = {}
    for L in lengths:
        best = None
        per_tilt = {}
        for t in tilts_deg:
            tr = np.deg2rad(t)
            dirs = -(np.cos(tr) * y + np.sin(tr) * z)
            # the handle may point either way along the axis (the hand can hold the handle from either side)
            for sgn in (1, -1):
                dd = sgn * dirs
                head = grasp_pos + L * dd
                incl = np.degrees(np.arcsin(np.clip(-dd[:, 2], -1, 1)))  # positive when the head is below the hand
                ok = (np.abs(head[:, 2]) < floor_tol) & (incl > min_incl)
                r = dict(tilt=t, sign=sgn, n_ok=int(ok.sum()), frac_ok=float(ok.mean()))
                # useful zone only: in front of the robot (head >= 0.30 m ahead of the pelvis, i.e. beyond the toes at 0.13 m) and a working
                # inclination band 25..60 deg (steeper than 60 deg = poking, not raking)
                fwd = ok & (head[:, 0] >= 0.30) & (incl <= 60.0)
                r["n_fwd"] = int(fwd.sum()); r["frac_fwd"] = float(fwd.mean())
                if fwd.any():
                    hp = head[fwd]
                    r.update(footprint=dict(x_min=float(hp[:, 0].min()), x_max=float(hp[:, 0].max()), y_min=float(hp[:, 1].min()), y_max=float(hp[:, 1].max()),
                                            area_m2=float(len(voxelize(hp[:, :2], res)) * res ** 2)),
                             incl_range=[float(incl[fwd].min()), float(incl[fwd].max())],
                             stroke_len_x=float(hp[:, 0].max() - hp[:, 0].min()),
                             hand_height_range=[float(grasp_pos[fwd, 2].min()), float(grasp_pos[fwd, 2].max())])
                per_tilt[f"tilt{t:+d}_sign{sgn:+d}"] = r
                if r["n_fwd"] and (best is None or r["footprint"]["area_m2"] > best["footprint"]["area_m2"]):
                    best = r
        results[f"{L:.2f}"] = dict(best=best, per_tilt=per_tilt)
    return results


def rake_scene_check(scene, n, use_waist, floor_tol=0.05, res=0.02, seed=1):
    """Collision-aware rake check on a scene built with --rake-length (rake welded in the RIGHT hand, tilt 0)."""
    S = Sampler(scene, use_waist=use_waist, seed=seed)
    r = S.sample("right", n, ["R_grasp", "R_rake_end", "R_rake_tip"])
    free = ~r["coll"]
    head = r["pos"][:, 1]; grasp = r["pos"][:, 0]
    dd = head - grasp; L = np.linalg.norm(dd, axis=1); dd /= L[:, None]
    incl = np.degrees(np.arcsin(np.clip(-dd[:, 2], -1, 1)))
    ok = free & (np.abs(head[:, 2]) < floor_tol) & (incl > 25.0) & (incl <= 60.0) & (head[:, 0] >= 0.30)
    ok_nocoll = (np.abs(head[:, 2]) < floor_tol) & (incl > 25.0) & (incl <= 60.0) & (head[:, 0] >= 0.30)
    out = dict(scene=scene, n=n, length=float(L.mean()), collision_free_fraction=float(free.mean()), frac_fwd_floor=float(ok.mean()),
               frac_fwd_floor_ignoring_collisions=float(ok_nocoll.mean()))
    if ok.any():
        hp = head[ok]
        out.update(footprint=dict(x_min=float(hp[:, 0].min()), x_max=float(hp[:, 0].max()), y_min=float(hp[:, 1].min()), y_max=float(hp[:, 1].max()),
                                  area_m2=float(len(voxelize(hp[:, :2], res)) * res ** 2)),
                   incl_range=[float(incl[ok].min()), float(incl[ok].max())], hand_height_range=[float(grasp[ok, 2].min()), float(grasp[ok, 2].max())])
        # example configuration with the head far forward (for the viewer / a keyframe)
        i = np.flatnonzero(ok)[np.argmax(hp[:, 0])]
        out["example_q"] = dict(zip(r["qnames"], r["q"][i].round(4).tolist())); out["example_head"] = head[i].round(3).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rake-scenes", nargs="*", default=None, help="scenes built with --rake-length; run the collision-aware rake check only")
    ap.add_argument("--scene", default=os.path.join(HERE, "out", "r1_scene_bent.xml"))
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--res", type=float, default=0.02)
    ap.add_argument("--waist", action="store_true", help="also sample waist yaw")
    ap.add_argument("--lengths", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2])
    ap.add_argument("--tilts", type=int, nargs="+", default=[0, 30, 45, 60, 75, 90])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.rake_scenes:
        res_all = {}
        for sc in args.rake_scenes:
            o = rake_scene_check(sc, args.n, args.waist, res=args.res); res_all[sc] = o
            fp = o.get("footprint")
            print(f"[rake scene] {os.path.basename(sc)} L={o['length']:.2f} m: collision-free {o['collision_free_fraction']*100:.1f}%, head-on-floor forward {o['frac_fwd_floor']*100:.2f}% "
                  f"(ignoring collisions {o['frac_fwd_floor_ignoring_collisions']*100:.2f}%)" + (f", footprint x[{fp['x_min']:+.2f},{fp['x_max']:+.2f}] y[{fp['y_min']:+.2f},{fp['y_max']:+.2f}] area {fp['area_m2']:.2f} m^2, incl {o['incl_range'][0]:.0f}-{o['incl_range'][1]:.0f}, hand z {o['hand_height_range'][0]:.2f}-{o['hand_height_range'][1]:.2f}" if fp else ", NOT reachable"))
        out_json = args.out or os.path.join(HERE, "out", "rake_scenes" + ("_waist" if args.waist else "") + ".json")
        with open(out_json, "w") as f:
            json.dump(res_all, f, indent=1)
        print("wrote", out_json); return

    S = Sampler(args.scene, use_waist=args.waist)
    tag = os.path.splitext(os.path.basename(args.scene))[0] + ("_waist" if args.waist else "")
    out_json = args.out or os.path.join(HERE, "out", f"reach_{tag}.json")
    report = dict(scene=args.scene, n_samples=args.n, voxel_m=args.res, waist_sampled=args.waist, hands={})
    clouds = {}
    for side, pre in (("left", "L_"), ("right", "R_")):
        r = S.sample(side, args.n, [f"{pre}ee", f"{pre}grasp", f"{pre}wrist"])
        free = ~r["coll"]
        print(f"[{side}] {args.n} samples in {r['dt']:.1f}s, collision-free {free.mean()*100:.1f}%")
        h = dict(collision_free_fraction=float(free.mean()))
        for k, si in (("ee", 0), ("grasp", 1)):
            p = r["pos"][:, si]
            h[k] = dict(all=dict(**voxel_stats(p, args.res), **extents(p)),
                        collision_free=dict(**voxel_stats(p[free], args.res), **extents(p[free]), reach_at_height=reach_at_heights(p[free], args.res)))
        h["rake"] = rake_check(r["pos"][free, 1], r["rot"][free, 1], args.lengths, args.tilts, res=args.res)
        report["hands"][side] = h
        clouds[side] = dict(pos=r["pos"][free], rot=r["rot"][free], q=r["q"][free])
    # two-handed: voxels reachable (collision-free, independently) by BOTH hands
    for k, si in (("ee", 0), ("grasp", 1)):
        vl = voxelize(clouds["left"]["pos"][:, si], args.res); vr = voxelize(clouds["right"]["pos"][:, si], args.res)
        sl = {tuple(v) for v in vl}; sr = {tuple(v) for v in vr}
        both = np.array(sorted(sl & sr)) if sl & sr else np.zeros((0, 3), int)
        union = len(sl | sr)
        rep = dict(n_voxels=int(len(both)), volume_m3=float(len(both) * args.res ** 3), union_volume_m3=float(union * args.res ** 3))
        if len(both):
            pb = (both + 0.5) * args.res
            rep.update(extents(pb), reach_at_height=reach_at_heights(pb, args.res))
        report[f"two_hand_overlap_{k}"] = rep
    np.savez_compressed(os.path.join(HERE, "out", f"cloud_{tag}.npz"), **{f"{s}_{k}": v for s, dd in clouds.items() for k, v in dd.items()})
    with open(out_json, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", out_json)

    # summary
    for side in ("left", "right"):
        h = report["hands"][side]
        g = h["grasp"]["collision_free"]
        print(f"\n== {side} hand (grasp point, collision-free) ==")
        print(f"  volume {g['volume_m3']:.3f} m^3   x[{g['x_min']:+.2f},{g['x_max']:+.2f}] y[{g['y_min']:+.2f},{g['y_max']:+.2f}] z[{g['z_min']:+.2f},{g['z_max']:+.2f}]")
        for hh, v in g["reach_at_height"].items():
            if v: print(f"  z={hh}: forward reach x_max={v['x_max']:+.2f}  lateral y[{v['y_min']:+.2f},{v['y_max']:+.2f}]  area {v['area_m2']:.2f} m^2")
        print("  rake head on floor (best grip):")
        for L, v in h["rake"].items():
            b = v["best"]
            if b is None: print(f"    L={L} m: NOT reachable"); continue
            fp = b["footprint"]
            print(f"    L={L} m: {b['frac_fwd']*100:5.1f}% of configs, tilt {b['tilt']:+d} deg, forward footprint x[{fp['x_min']:+.2f},{fp['x_max']:+.2f}] y[{fp['y_min']:+.2f},{fp['y_max']:+.2f}] area {fp['area_m2']:.2f} m^2, incl {b['incl_range'][0]:.0f}-{b['incl_range'][1]:.0f} deg, hand z {b['hand_height_range'][0]:.2f}-{b['hand_height_range'][1]:.2f}")
    t = report["two_hand_overlap_grasp"]
    print(f"\n== two-hand overlap (grasp point) == volume {t['volume_m3']:.3f} m^3 (union {t['union_volume_m3']:.3f})", {k: round(v, 2) for k, v in t.items() if k.startswith(('x_', 'y_', 'z_'))})


if __name__ == "__main__":
    main()
