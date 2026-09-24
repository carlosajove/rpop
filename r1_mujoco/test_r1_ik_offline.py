#!/usr/bin/env python
"""Offline sanity check of xr_teleoperate's R1 5-DoF arm IK on a rake stroke, without a headset.

Feeds a scripted trajectory of wrist targets (in the r1_a5 base frame, i.e. the frame xr_teleoperate's
R1_A5_ArmIK expects) through R1_A5_ArmIK.solve_ik, then evaluates the solutions in the MuJoCo standing model:
position/orientation error of the 0.20 m end-effector point, joint-limit saturation, collisions, and renders
key frames.  Optionally streams the solution to the running r1_mujoco_sim bridge over DDS (--dds) the same way
R1_A5_ArmController does, so you can watch it in the MuJoCo viewer.

Run from the tv env:
    cd ~/projects/unitree-r1/r1_mujoco && python test_r1_ik_offline.py [--dds] [--rake-length 0.8]
"""
import argparse, os, sys, time, json
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
XR_TELEOP = os.path.abspath(os.path.join(HERE, "..", "xr_teleoperate"))
sys.path.insert(0, XR_TELEOP)
sys.path.insert(0, os.path.join(XR_TELEOP, "teleop"))

ARM_L = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint"]
ARM_R = [j.replace("left", "right") for j in ARM_L]
SLOTS_L, SLOTS_R = [15, 16, 17, 18, 19], [22, 23, 24, 25, 26]


def rot_y(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def pose(p, R):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p; return T


def rake_stroke(n=60, side_y=-0.20, two_hand=True, sep=0.35):
    """Right hand (lower hand) sweeps a rake head along the floor from ~1.0 m ahead back to ~0.5 m; left hand
    holds the handle `sep` metres up the shaft.  Poses expressed in the r1_a5 base frame (waist yaw origin):
    x forward, y left, z up; base origin is 0.777 m above the floor in the standing model."""
    z_base = 0.777
    traj = []
    for t in np.linspace(0, 1, n):
        # lower hand moves from (0.50, y, 0.55 m above floor) to (0.25, y, 0.70) while the shaft pitches from 35 to 55 deg
        p_r = np.array([0.45 - 0.22 * t, side_y, 0.55 + 0.15 * t - z_base])
        incl = np.deg2rad(35 + 20 * t)                       # shaft inclination above the floor
        u = np.array([np.cos(incl), 0.0, -np.sin(incl)])     # shaft direction hand -> head (forward-down)
        # hand frame: forearm/fingers roughly along the shaft normal, palm inward. Build R from x = shaft-normal
        # pointing forward-down-ish; the IK weights rotation lightly (0.5 vs 50), so this mostly guides the wrist.
        x_axis = np.array([np.sin(incl), 0.0, np.cos(incl)])  # perpendicular to the shaft, pointing up-forward
        x_axis = rot_y(np.deg2rad(60)) @ x_axis * 0 + np.array([np.cos(incl - np.pi / 2 + np.deg2rad(60)), 0, -np.sin(incl - np.pi / 2 + np.deg2rad(60))])
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.array([0.0, 1.0, 0.0]); z_axis = np.cross(x_axis, y_axis); z_axis /= np.linalg.norm(z_axis); y_axis = np.cross(z_axis, x_axis)
        R = np.column_stack([x_axis, y_axis, z_axis])
        # upper (left) hand further up the shaft and on the LEFT of the body: the shoulder-roll limits (-13 deg inward)
        # do not let an arm cross the midline, so the shaft runs diagonally across the body like a broom
        p_l = p_r - sep * u + np.array([0, 0.35, 0])
        traj.append((pose(p_l, R), pose(p_r, R)))
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--rake-length", default=None)
    ap.add_argument("--dds", action="store_true", help="also stream to the bridge on rt/lowcmd (domain 1)")
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--render", action="store_true", help="render key frames to out/ik_frame_*.png (needs MUJOCO_GL=egl)")
    args = ap.parse_args()
    scene = args.scene or (os.path.join(HERE, "out", f"r1_rake_right_{args.rake_length}.xml") if args.rake_length else os.path.join(HERE, "out", "r1_scene_bent.xml"))

    cwd = os.getcwd(); os.chdir(os.path.join(XR_TELEOP, "teleop"))  # R1_A5_ArmIK uses relative asset paths
    from teleop.robot_control.robot_arm_ik import R1_A5_ArmIK
    ik = R1_A5_ArmIK(Unit_Test=False, Visualization=False)
    os.chdir(cwd)

    m = mujoco.MjModel.from_xml_path(scene); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    base = d.body("waist_yaw_link").xpos.copy()  # r1_a5 root frame in world (waist yaw = 0)
    print(f"r1_a5 base frame in world: {base.round(4)}")
    qadr_l = [m.joint(j).qposadr[0] for j in ARM_L]; qadr_r = [m.joint(j).qposadr[0] for j in ARM_R]
    lo = np.array([m.joint(j).range[0] for j in ARM_L + ARM_R]); hi = np.array([m.joint(j).range[1] for j in ARM_L + ARM_R])

    pub = None
    if args.dds:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        ChannelFactoryInitialize(1)
        pub = ChannelPublisher("rt/lowcmd", LowCmd_); pub.Init()
        cmd = unitree_hg_msg_dds__LowCmd_()
        for i in range(35):
            cmd.motor_cmd[i].mode = 1; cmd.motor_cmd[i].kp = 0.0; cmd.motor_cmd[i].kd = 0.0
        for i in SLOTS_L + SLOTS_R:
            cmd.motor_cmd[i].kp = 50.0; cmd.motor_cmd[i].kd = 2.0
        for i in (19, 26):
            cmd.motor_cmd[i].kp = 30.0
        for i in (12, 13, 29, 30):
            cmd.motor_cmd[i].kp = 50.0; cmd.motor_cmd[i].kd = 2.0; cmd.motor_cmd[i].q = 0.0

    traj = rake_stroke()
    q_prev = np.zeros(10); rows = []
    renderer = None
    if args.render:
        renderer = mujoco.Renderer(m, 720, 960)
    for k, (T_l, T_r) in enumerate(traj):
        t0 = time.time()
        sol_q, sol_tau = ik.solve_ik(T_l, T_r, q_prev, np.zeros(10))
        dt_ik = time.time() - t0
        q_prev = sol_q
        d.qpos[qadr_l] = sol_q[:5]; d.qpos[qadr_r] = sol_q[5:]
        mujoco.mj_forward(m, d)
        e_l = d.site("L_ee").xpos - (base + T_l[:3, 3]); e_r = d.site("R_ee").xpos - (base + T_r[:3, 3])
        sat = np.minimum(sol_q - lo, hi - sol_q) < 0.02
        ncon = sum(1 for i in range(d.ncon) if not (m.geom(d.contact[i].geom1).name == "floor" or m.geom(d.contact[i].geom2).name == "floor"))
        rows.append(dict(k=k, ik_ms=dt_ik * 1e3, err_l_cm=np.linalg.norm(e_l) * 100, err_r_cm=np.linalg.norm(e_r) * 100, saturated=[n for n, s in zip(ARM_L + ARM_R, sat) if s],
                         R_ee=d.site("R_ee").xpos.round(3).tolist(), target_R=(base + T_r[:3, 3]).round(3).tolist()))
        if pub is not None:
            for i, v in zip(SLOTS_L, sol_q[:5]): cmd.motor_cmd[i].q = float(v)
            for i, v in zip(SLOTS_R, sol_q[5:]): cmd.motor_cmd[i].q = float(v)
            for _ in range(max(1, int(round(250 / args.rate)))):
                pub.Write(cmd); time.sleep(1 / 250)
        if renderer is not None and k in (0, len(traj) // 2, len(traj) - 1):
            import PIL.Image
            opt = mujoco.MjvOption(); opt.sitegroup[4] = 1
            renderer.update_scene(d, camera="iso", scene_option=opt); PIL.Image.fromarray(renderer.render()).save(os.path.join(HERE, "out", f"ik_frame_{k:02d}.png"))
    if pub is not None:
        # hold the last pose for a second, then leave the bridge to hold on its own
        for _ in range(250): pub.Write(cmd); time.sleep(1 / 250)
    err_l = np.array([r["err_l_cm"] for r in rows]); err_r = np.array([r["err_r_cm"] for r in rows]); ikt = np.array([r["ik_ms"] for r in rows])
    print(f"IK time: mean {ikt.mean():.1f} ms, max {ikt.max():.1f} ms  (first call includes warm-up)")
    print(f"position error of the 0.20 m ee point: right mean {err_r.mean():.1f} cm max {err_r.max():.1f} cm | left mean {err_l.mean():.1f} cm max {err_l.max():.1f} cm")
    sat_all = sorted({n for r in rows for n in r["saturated"]})
    print("joints at limit during the stroke:", sat_all if sat_all else "none")
    for r in rows[:: max(1, len(rows) // 6)]:
        print(f"  k={r['k']:2d} target_R={r['target_R']} reached={r['R_ee']} err={r['err_r_cm']:.1f} cm sat={r['saturated']}")
    with open(os.path.join(HERE, "out", "ik_offline_report.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
