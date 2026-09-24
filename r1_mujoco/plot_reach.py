#!/usr/bin/env python
"""Figures for the set designer: hand workspace (side/front/top) and rake-head floor footprints.
Usage: python plot_reach.py [--cloud out/cloud_r1_scene_bent.npz] [--lengths 0.6 0.8 1.0 1.2]"""
import argparse, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reach_analysis import voxelize

HERE = os.path.dirname(os.path.abspath(__file__))


def density_img(p, a, b, res, lim):
    """2-D occupancy (projected voxels) of point set p on axes a,b."""
    v = voxelize(p[:, [a, b]], res)
    img = np.zeros((int((lim[3] - lim[2]) / res) + 1, int((lim[1] - lim[0]) / res) + 1), bool)
    ix = v[:, 0] - int(np.floor(lim[0] / res)); iy = v[:, 1] - int(np.floor(lim[2] / res))
    ok = (ix >= 0) & (ix < img.shape[1]) & (iy >= 0) & (iy < img.shape[0])
    img[iy[ok], ix[ok]] = True
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", default=os.path.join(HERE, "out", "cloud_r1_scene_bent.npz"))
    ap.add_argument("--lengths", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2])
    ap.add_argument("--res", type=float, default=0.02)
    args = ap.parse_args()
    c = np.load(args.cloud)
    L = c["left_pos"][:, 1]; R = c["right_pos"][:, 1]; Rrot = c["right_rot"][:, 1]; Lrot = c["left_rot"][:, 1]
    res = args.res
    robot = dict(pelvis=0.728, shoulder=0.974, head=1.217, foot_toe=0.13, foot_heel=-0.05)

    # ---- figure 1: workspace projections
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
    lim_xz = (-0.7, 0.8, 0.0, 1.7); lim_yz = (-0.9, 0.9, 0.0, 1.7); lim_xy = (-0.7, 0.8, -0.9, 0.9)
    both = {tuple(v) for v in voxelize(L, res)} & {tuple(v) for v in voxelize(R, res)}
    both = (np.array(sorted(both)) + 0.5) * res if both else np.zeros((0, 3))
    for ax, (a, b, lim, xl, yl, title) in zip(axs, [(0, 2, lim_xz, "x forward [m]", "z [m]", "side view (right hand)"),
                                                    (1, 2, lim_yz, "y left [m]", "z [m]", "front view (both hands)"),
                                                    (0, 1, lim_xy, "x forward [m]", "y left [m]", "top view (both hands)")]):
        ext = [lim[0], lim[1], lim[2], lim[3]]
        if title.startswith("side"):
            ax.imshow(density_img(R, a, b, res, lim), origin="lower", extent=ext, cmap="Blues", alpha=0.9, aspect="equal")
        else:
            ax.imshow(density_img(R, a, b, res, lim), origin="lower", extent=ext, cmap="Blues", alpha=0.6, aspect="equal")
            ax.imshow(np.ma.masked_where(~density_img(L, a, b, res, lim), density_img(L, a, b, res, lim)), origin="lower", extent=ext, cmap="Reds", alpha=0.5, aspect="equal")
            if len(both):
                ax.imshow(np.ma.masked_where(~density_img(both, a, b, res, lim), density_img(both, a, b, res, lim)), origin="lower", extent=ext, cmap="Greens", alpha=0.9, aspect="equal")
        if b == 2:
            ax.axhline(0, color="k", lw=2); ax.axhline(robot["pelvis"], color="gray", ls="--", lw=0.8); ax.axhline(robot["shoulder"], color="gray", ls="--", lw=0.8); ax.axhline(robot["head"], color="gray", ls=":", lw=0.8)
            ax.text(lim[0] + 0.02, robot["shoulder"] + 0.02, "shoulder 0.97", fontsize=8, color="gray"); ax.text(lim[0] + 0.02, robot["pelvis"] + 0.02, "pelvis 0.73", fontsize=8, color="gray")
            ax.axhline(0.46, color="orange", ls="--", lw=1); ax.text(lim[1] - 0.45, 0.47, "lowest hand 0.46 m", fontsize=8, color="orange")
        if a == 0 and b == 2:
            ax.plot([robot["foot_heel"], robot["foot_toe"]], [0.01, 0.01], color="k", lw=6)
        if a == 0 and b == 1:
            ax.add_patch(plt.Rectangle((robot["foot_heel"], -0.15), robot["foot_toe"] - robot["foot_heel"], 0.30, fill=False, color="k", lw=1.5)); ax.text(-0.05, -0.2, "feet", fontsize=8)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title); ax.grid(alpha=0.3)
    fig.suptitle("R1 hand (grasp point) reachable workspace, standing, legs fixed, waist not used — blue right, red left, green both", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "out", "reach_workspace.png"), dpi=130)

    # ---- figure 2: rake head on floor, forward footprint per length (right hand), best tilt
    fig, axs = plt.subplots(1, len(args.lengths), figsize=(4.2 * len(args.lengths), 4.6), sharey=True)
    y = Rrot[:, :, 1]; z = Rrot[:, :, 2]
    for ax, Lr in zip(np.atleast_1d(axs), args.lengths):
        best, best_area = None, -1
        for t in (0, 30, 45, 60, 75, 90):
            tr = np.deg2rad(t)
            for sgn in (1, -1):
                dd = -sgn * (np.cos(tr) * y + np.sin(tr) * z)
                head = R + Lr * dd
                incl = np.degrees(np.arcsin(np.clip(-dd[:, 2], -1, 1)))
                ok = (np.abs(head[:, 2]) < 0.05) & (incl > 25) & (incl <= 60) & (head[:, 0] >= 0.10)
                area = len(voxelize(head[ok][:, :2], res)) * res ** 2 if ok.any() else 0
                if area > best_area:
                    best_area, best = area, (t, sgn, head[ok], incl[ok])
        t, sgn, hp, incl = best
        if len(hp):
            sc = ax.scatter(hp[:, 0], hp[:, 1], c=incl, s=2, cmap="viridis", vmin=25, vmax=60)
        ax.add_patch(plt.Rectangle((robot["foot_heel"], -0.15), robot["foot_toe"] - robot["foot_heel"], 0.30, fill=True, color="k", alpha=0.6)); ax.text(-0.05, -0.25, "feet", fontsize=8)
        ax.set_xlim(-0.3, 1.6); ax.set_ylim(-1.6, 1.0); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(f"rake L={Lr:.1f} m: floor footprint {best_area:.2f} m²\n(right hand, grip tilt {t:+d}°, x reach {hp[:,0].max() if len(hp) else 0:.2f} m)", fontsize=9)
        ax.set_xlabel("x forward [m]")
    np.atleast_1d(axs)[0].set_ylabel("y left [m]")
    if len(hp):
        cb = fig.colorbar(sc, ax=list(np.atleast_1d(axs)), shrink=0.8, pad=0.01); cb.set_label("handle inclination above floor [deg]")
    fig.suptitle("Where the rake head can touch the floor (|z|<5 cm, handle 25-60° to the floor), one hand, collision-free arm poses", fontsize=11)
    fig.savefig(os.path.join(HERE, "out", "rake_footprints.png"), dpi=130, bbox_inches="tight")
    print("wrote out/reach_workspace.png, out/rake_footprints.png")


if __name__ == "__main__":
    main()
