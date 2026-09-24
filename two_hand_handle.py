#!/usr/bin/env python
"""Two-hand closed-chain check: can both R1 hands sit on one straight handle within joint limits?

For a grid of handle poses (mid-point p, unit direction u, hand separation s) we solve, per hand, a bounded
least-squares IK over that arm's 5 joints (waist yaw fixed per candidate) with two constraints
  * grasp point on the handle:  grasp_h = p -/+ (s/2) u          (3 residuals)
  * handle across the palm:     y_hand x u = 0                    (2 independent residuals; sign of u free)
Rotation about the handle axis is unconstrained (cylindrical grip).  Starting points come from the sampled
workspace cloud (nearest matches), then a local refinement.  The combined configuration is then collision-checked.

Modes reported per handle pose
  rigid_both : both hands satisfy position (<= tol_p) and axis alignment (<= tol_a) -> a real two-hand rigid grip
  rigid_loose: primary (right) hand rigid, second hand only within tol_loose of its point on the handle, orientation
               free -> "one hand plus a loose second"

Usage: python two_hand_handle.py --scene out/r1_scene_bent.xml --cloud out/cloud_r1_scene_bent.npz
"""
import argparse, json, os, time, itertools
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from reach_analysis import Sampler, arm_joints

HERE = os.path.dirname(os.path.abspath(__file__))


class ArmIK:
    def __init__(self, S, side):
        self.S, self.side = S, side
        self.m, self.d = S.m, S.d
        names = arm_joints(side)
        self.qadr = np.array([self.m.joint(n).qposadr[0] for n in names])
        rng = np.array([self.m.joint(n).range for n in names])
        self.lo, self.hi = rng[:, 0], rng[:, 1]
        self.sid = self.m.site(("L_" if side == "left" else "R_") + "grasp").id

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3)[:, 1].copy()

    def residual(self, q, target, u, w_rot):
        p, y = self.fk(q)
        return np.concatenate([p - target, w_rot * np.cross(y, u)])

    def solve(self, target, u, q_inits, w_rot=0.05, tol_p=0.01, tol_a_deg=10.0):
        best = None
        for q0 in q_inits:
            r = least_squares(self.residual, np.clip(q0, self.lo + 1e-6, self.hi - 1e-6), bounds=(self.lo, self.hi), args=(target, u, w_rot),
                              xtol=1e-6, ftol=1e-8, max_nfev=60)
            p, y = self.fk(r.x)
            ep = np.linalg.norm(p - target); ea = np.degrees(np.arcsin(np.clip(np.linalg.norm(np.cross(y, u)), 0, 1)))
            if best is None or ep + 0.01 * ea < best[1] + 0.01 * best[2]:
                best = (r.x.copy(), ep, ea)
            if ep <= tol_p and ea <= tol_a_deg:
                break
        return best


def combined_collision(S, ql, qr, ikl, ikr, waist_adr=None, waist=0.0):
    d, m = S.d, S.m
    d.qpos[:] = S.q0
    if waist_adr is not None:
        d.qpos[waist_adr] = waist
    d.qpos[ikl.qadr] = ql; d.qpos[ikr.qadr] = qr
    mujoco.mj_kinematics(m, d); mujoco.mj_collision(m, d)
    chains = S.chain["left"] | S.chain["right"]
    for i in range(d.ncon):
        c = d.contact[i]; g1, g2 = c.geom1, c.geom2
        key = (min(g1, g2), max(g1, g2))
        if key in S.baseline:
            continue
        if S.floor in key:
            other = g2 if g1 == S.floor else g1
            if other in S.feet:
                continue
            return True
        same_chain = (g1 in S.chain["left"] and g2 in S.chain["left"]) or (g1 in S.chain["right"] and g2 in S.chain["right"])
        if same_chain:
            continue
        if g1 in chains or g2 in chains:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=os.path.join(HERE, "out", "r1_scene_bent.xml"))
    ap.add_argument("--cloud", default=os.path.join(HERE, "out", "cloud_r1_scene_bent.npz"))
    ap.add_argument("--seps", type=float, nargs="+", default=[0.25, 0.35, 0.45])
    ap.add_argument("--xs", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--ys", type=float, nargs="+", default=[-0.15, 0.0, 0.15])
    ap.add_argument("--zs", type=float, nargs="+", default=[0.5, 0.7, 0.9, 1.1, 1.3])
    ap.add_argument("--pitches", type=float, nargs="+", default=[-60, -45, -30, 0, 30, 45, 60], help="handle pitch (deg); negative = far (right-hand) end lower")
    ap.add_argument("--yaws", type=float, nargs="+", default=[-60, -30, 0, 30, 60, 90], help="handle yaw (deg); 0 = along x (one hand behind the other), 90 = across the body")
    ap.add_argument("--waist", type=float, nargs="+", default=[0.0], help="waist yaw candidates (rad)")
    ap.add_argument("--tol-p", type=float, default=0.01)
    ap.add_argument("--tol-a", type=float, default=10.0)
    ap.add_argument("--tol-loose", type=float, default=0.03)
    ap.add_argument("--k", type=int, default=3, help="nearest cloud samples used as IK starts")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    S = Sampler(args.scene, use_waist=False)
    ik = {"left": ArmIK(S, "left"), "right": ArmIK(S, "right")}
    waist_adr = S.m.joint("waist_yaw_joint").qposadr[0]
    cloud = np.load(args.cloud)
    trees, cq, cy = {}, {}, {}
    for side in ("left", "right"):
        pos = cloud[f"{side}_pos"][:, 1]; rot = cloud[f"{side}_rot"][:, 1]
        cq[side] = cloud[f"{side}_q"][:, :5]; cy[side] = rot[:, :, 1]
        trees[side] = cKDTree(pos)
    print(f"cloud: {len(cq['left'])} left / {len(cq['right'])} right samples")

    results = []
    t0 = time.time()
    grid = list(itertools.product(args.seps, args.xs, args.ys, args.zs, args.pitches, args.yaws))
    print(f"{len(grid)} handle poses x {len(args.waist)} waist candidates")
    for gi, (s, x, y, z, pitch, yaw) in enumerate(grid):
        th, ph = np.deg2rad(pitch), np.deg2rad(yaw)
        u = np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)])
        p = np.array([x, y, z])
        tgt = {"left": p - 0.5 * s * u, "right": p + 0.5 * s * u}
        rec = dict(s=s, x=x, y=y, z=z, pitch=pitch, yaw=yaw, rigid_both=False, rigid_loose=False, waist=None)
        for w in args.waist:
            S.d.qpos[:] = S.q0; S.d.qpos[waist_adr] = w
            sol, loose_ok = {}, {}
            for side in ("left", "right"):
                # candidate starts: nearest cloud points, preferring those whose y axis is roughly aligned with u
                dist, idx = trees[side].query(tgt[side], k=40)
                idx = np.atleast_1d(idx); dist = np.atleast_1d(dist)
                align = np.abs(cy[side][idx] @ u)
                order = np.argsort(dist / 0.03 - align)  # trade position error vs alignment
                starts = cq[side][idx[order[: args.k]]]
                loose_ok[side] = bool(dist.min() <= args.tol_loose)  # (waist 0 cloud; ok for loose test)
                sol[side] = ik[side].solve(tgt[side], u, starts, tol_p=args.tol_p, tol_a_deg=args.tol_a)
            rig = {sd: (sol[sd][1] <= args.tol_p and sol[sd][2] <= args.tol_a) for sd in sol}
            if rig["left"] and rig["right"]:
                if not combined_collision(S, sol["left"][0], sol["right"][0], ik["left"], ik["right"], waist_adr, w):
                    rec.update(rigid_both=True, waist=w, q_left=sol["left"][0].round(4).tolist(), q_right=sol["right"][0].round(4).tolist())
            if rig["right"] and (loose_ok["left"] or rig["left"]):
                rec["rigid_loose"] = True
            if rig["left"] and (loose_ok["right"] or rig["right"]):
                rec["rigid_loose"] = True
            rec["err_left"] = [round(sol["left"][1], 4), round(sol["left"][2], 1)]; rec["err_right"] = [round(sol["right"][1], 4), round(sol["right"][2], 1)]
            if rec["rigid_both"]:
                break
        results.append(rec)
        if gi % 500 == 0:
            print(f"  {gi}/{len(grid)}  {time.time()-t0:.0f}s  rigid_both so far {sum(r['rigid_both'] for r in results)}")
    S.d.qpos[:] = S.q0; mujoco.mj_forward(S.m, S.d)

    # summary
    summary = {}
    for s in args.seps:
        rs = [r for r in results if r["s"] == s]
        both = [r for r in rs if r["rigid_both"]]; loose = [r for r in rs if r["rigid_loose"]]
        summ = dict(n_poses=len(rs), rigid_both=len(both), rigid_loose=len(loose), frac_rigid_both=len(both) / len(rs), frac_rigid_loose=len(loose) / len(rs))
        if both:
            summ["rigid_both_region"] = dict(x=sorted({r["x"] for r in both}), y=sorted({r["y"] for r in both}), z=sorted({r["z"] for r in both}),
                                             pitch=sorted({r["pitch"] for r in both}), yaw=sorted({r["yaw"] for r in both}))
            # per (pitch,yaw) count
            summ["by_orientation"] = {f"pitch{int(pp):+d}_yaw{int(yy):+d}": sum(1 for r in both if r["pitch"] == pp and r["yaw"] == yy) for pp in args.pitches for yy in args.yaws}
            summ["by_height"] = {f"z{zz:.1f}": sum(1 for r in both if r["z"] == zz) for zz in args.zs}
        summary[f"s{s:.2f}"] = summ
        print(f"\n== hand separation {s:.2f} m ==  poses {len(rs)}  rigid both hands: {len(both)} ({100*len(both)/len(rs):.1f}%)  rigid+loose: {len(loose)} ({100*len(loose)/len(rs):.1f}%)")
        if both:
            print("   feasible heights z:", summ["rigid_both_region"]["z"], " x:", summ["rigid_both_region"]["x"], " pitch:", summ["rigid_both_region"]["pitch"], " yaw:", summ["rigid_both_region"]["yaw"])
            top = sorted(summ["by_orientation"].items(), key=lambda kv: -kv[1])[:6]
            print("   best orientations:", top)
    out = args.out or os.path.join(HERE, "out", "two_hand_handle.json")
    with open(out, "w") as f:
        json.dump(dict(args=vars(args), summary=summary, results=results), f, indent=1)
    print("wrote", out, f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
