#!/usr/bin/env python
"""Build a MuJoCo scene of the Unitree R1 (EDU, 5-DoF arms) standing on a floor with the legs fixed.

Source model: unitree_ros/robots/r1_description/R1.urdf (full body).  The r1_a5 URDF is the same upper body
without legs, so using the full body with the leg joints frozen gives identical arm kinematics *plus* the real
leg/torso geometry for collision and floor-height checks.

What the script does
  * loads the URDF with mujoco.MjSpec (mesh path bug in the Unitree URDF patched in assets/R1_fixed.urdf)
  * freezes the 12 leg joints at the standing pose (joint removed, child body rotated by the standing angle)
  * places the pelvis so that the lowest sole point touches z = 0 and adds a floor
  * attaches the BrainCo Revo2 hand URDFs (from xr_teleoperate/assets/brainco_hand) on both wrist flanges
  * adds sites: {L,R}_wrist (wrist roll joint), {L,R}_flange (hand mount), {L,R}_ee (the xr_teleoperate IK target,
    0.20 m along the wrist roll axis), {L,R}_grasp (handle centre in a power grasp), head_cam
  * adds torque actuators for the 14 upper-body joints (waist roll+yaw, 2x5 arm, 2 head) and a 'standing' keyframe
  * optionally adds a rake (cylindrical handle + head) welded into one hand's grasp

Usage
  python build_r1_scene.py                              # -> out/r1_scene.xml   (bent-knee standing pose)
  python build_r1_scene.py --pose straight              # straight legs (max height)
  python build_r1_scene.py --rake-length 0.8 --rake-hand right -o out/r1_rake80.xml
"""
import argparse
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
R1_URDF = os.path.join(HERE, "assets", "R1_fixed.urdf")
HAND_DIR = os.path.join(HERE, "..", "xr_teleoperate", "assets", "brainco_hand")

# Standing poses (rad).  "bent" is the pose the r1_lab Isaac Lab policy/init uses (G1-style, slightly bent knee);
# "straight" is all zeros (URDF zero = straight legs).
LEG_POSES = {
    "straight": {},
    "bent": {"hip_pitch": -0.20, "knee": 0.42, "ankle_pitch": -0.23},
}
LEG_JOINTS = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]

UPPER_JOINTS = [
    "waist_roll_joint", "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "head_pitch_joint", "head_yaw_joint",
]
# Effort limits from the URDF (N m)
EFFORT = {"waist_roll_joint": 60, "waist_yaw_joint": 60, "shoulder_pitch": 60, "shoulder_roll": 60, "shoulder_yaw": 33, "elbow": 33, "wrist_roll": 33, "head": 33}

# Hand mount: the wrist_roll_link mesh ends at x = 0.1427 m (flange face ~0.140 m) along the wrist roll axis.
FLANGE_X = 0.140
# xr_teleoperate R1_A5_ArmIK end-effector frame: 0.20 m along the wrist roll axis
EE_X = 0.20
# Handle centre in a power grasp, expressed in the BrainCo "<side>_base_link" frame (z along the fingers,
# +x = palm side, fingers curl towards +x): 5.5 cm up the palm, 2.5 cm off the palm surface (handle r ~1.5 cm).
GRASP_IN_HAND = np.array([0.025, 0.0, 0.055])


def rpy_to_quat(r, p, y):
    """URDF fixed-axis rpy -> MuJoCo quaternion (w, x, y, z)."""
    qx = np.array([np.cos(r / 2), np.sin(r / 2), 0, 0])
    qy = np.array([np.cos(p / 2), 0, np.sin(p / 2), 0])
    qz = np.array([np.cos(y / 2), 0, 0, np.sin(y / 2)])
    q = np.zeros(4); mujoco.mju_mulQuat(q, qz, qy); q2 = np.zeros(4); mujoco.mju_mulQuat(q2, q, qx)
    return q2


def quat_mul(a, b):
    out = np.zeros(4); mujoco.mju_mulQuat(out, a, b); return out


def quat_inv(a):
    out = np.zeros(4); mujoco.mju_negQuat(out, a); return out


def axis_angle_quat(axis, angle):
    out = np.zeros(4); mujoco.mju_axisAngle2Quat(out, np.asarray(axis, float), angle); return out


def freeze_legs(spec, pose):
    """Remove the 12 leg joints, baking the standing angle into each child body's orientation."""
    for side in ("left", "right"):
        for jn in LEG_JOINTS:
            jname = f"{side}_{jn}_joint"
            j = spec.joint(jname)
            body = j.parent
            assert np.allclose(j.pos, 0), f"{jname} not at body origin"
            angle = pose.get(jn, 0.0)
            body.quat = quat_mul(body.quat, axis_angle_quat(j.axis, angle))
            spec.delete(j)


def lowest_point(model, data, body_filter=lambda name: True):
    """Lowest mesh vertex (world z) over geoms whose body passes body_filter."""
    zmin, where = np.inf, None
    for g in range(model.ngeom):
        bname = model.body(model.geom_bodyid[g]).name
        if not body_filter(bname):
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            vs = model.mesh_vert[model.mesh_vertadr[mid]: model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            w = (data.geom_xmat[g].reshape(3, 3) @ vs.T).T + data.geom_xpos[g]
            z = w[:, 2].min()
        else:
            z = data.geom_xpos[g][2] - model.geom_rbound[g]
        if z < zmin:
            zmin, where = z, bname
    return zmin, where


def hand_mount_quat(side):
    """Orientation of the BrainCo <side>_base_link in the R1 wrist_roll_link frame.

    Same convention as Unitree's g1_with_brainco_hand URDF: left rpy=(-90,0,-90) deg, right rpy=(90,0,90) deg,
    which maps hand z (fingers) -> arm +x, hand x (palm side) -> arm +y for the right hand / -y for the left, i.e.
    palms facing the body midline when the forearms point forward.
    """
    s = -1.0 if side == "left" else 1.0
    return rpy_to_quat(s * np.pi / 2, 0.0, s * np.pi / 2)


def attach_hand(spec, side, wrist_body):
    prefix = "L_" if side == "left" else "R_"
    hand = mujoco.MjSpec.from_file(os.path.join(HAND_DIR, f"brainco_{side}.urdf"))
    hand.meshdir = HAND_DIR
    # the xr_teleoperate hand URDF has an extra root 'base_link' -> '<side>_base_link' fixed transform (rpy 1.57 3.14 0);
    # attach so that <side>_base_link lands exactly on the mount frame.
    base = hand.body("base_link")
    # the xr_teleoperate hand URDF carries RGB debug-axis visuals (cylinders + spheres) on base_link: drop them
    for g in list(base.geoms):
        if g.type != mujoco.mjtGeom.mjGEOM_MESH:
            hand.delete(g)
    inner = hand.body(f"{side}_base_link")
    q_mount = hand_mount_quat(side)
    q_inner = np.array(inner.quat, float)
    p_inner = np.array(inner.pos, float)
    # mount = site * inner  ->  site = mount * inner^-1
    q_site = quat_mul(q_mount, quat_inv(q_inner))
    p_off = np.zeros(3); mujoco.mju_rotVecQuat(p_off, p_inner, q_site)
    site = wrist_body.add_site(name=f"{prefix}hand_mount", pos=[FLANGE_X, 0, 0] - p_off, quat=q_site, size=[0.004, 0, 0], group=4)
    assert np.allclose(base.pos, 0) and np.allclose(base.quat, [1, 0, 0, 0])
    spec.attach(hand, prefix=prefix, site=site)
    return prefix


def add_sites(spec, side):
    prefix = "L_" if side == "left" else "R_"
    wrist = spec.body(f"{side}_wrist_roll_link")
    wrist.add_site(name=f"{prefix}wrist", pos=[0, 0, 0], size=[0.005, 0, 0], group=4)
    q = hand_mount_quat(side)
    wrist.add_site(name=f"{prefix}flange", pos=[FLANGE_X, 0, 0], quat=q, size=[0.005, 0, 0], group=4)
    wrist.add_site(name=f"{prefix}ee", pos=[EE_X, 0, 0], quat=q, size=[0.006, 0, 0], group=4, rgba=[1, 0, 0, 1])
    # grasp centre: hand-frame offset rotated into the wrist frame
    off = np.zeros(3); mujoco.mju_rotVecQuat(off, GRASP_IN_HAND, q)
    wrist.add_site(name=f"{prefix}grasp", pos=[FLANGE_X, 0, 0] + off, quat=q, size=[0.008, 0, 0], group=4, rgba=[0, 1, 0, 1])
    return wrist


def add_rake(spec, side, length, handle_radius=0.015, head_width=0.40, tilt_deg=0.0):
    """Rake welded into the grasp of one hand.

    Handle axis = hand y axis (across the palm, perpendicular to the fingers).  The grasp centre is the 'top' end
    (hand end) of the handle; the rake head sits `length` metres further along -y_hand, and the tines point along
    the palm normal (-x_hand, away from the palm) so the rake is pulled with the palm facing the user.
    """
    prefix = "L_" if side == "left" else "R_"
    wrist = spec.body(f"{side}_wrist_roll_link")
    q = hand_mount_quat(side)
    off = np.zeros(3); mujoco.mju_rotVecQuat(off, GRASP_IN_HAND, q)
    # tilt_deg rotates the rake about the palm normal (hand x): 0 = handle across the palm (perpendicular to the
    # fingers, plain power grasp), 90 = handle along the fingers/forearm (an angled or forearm-mounted grip)
    # (negative rotation so that at 90 deg the head end points past the fingertips, not back towards the elbow)
    q_rake = quat_mul(q, axis_angle_quat([1, 0, 0], -np.deg2rad(tilt_deg)))
    rake = wrist.add_body(name=f"{prefix}rake", pos=[FLANGE_X, 0, 0] + off, quat=q_rake)
    # handle runs from the grasp centre (y=0) to y=-length in the hand frame; leave 10 cm sticking out above the hand
    rake.add_geom(name=f"{prefix}rake_handle", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0.10, 0, 0, -length, 0],
                  size=[handle_radius, 0, 0], rgba=[0.55, 0.35, 0.15, 1], mass=0.6, contype=1, conaffinity=1)
    # head: bar across, then tines
    rake.add_geom(name=f"{prefix}rake_head", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -length, 0],
                  size=[0.012, 0.012, head_width / 2], rgba=[0.3, 0.3, 0.3, 1], mass=0.5, contype=1, conaffinity=1)
    for i, z in enumerate(np.linspace(-head_width / 2, head_width / 2, 9)):
        rake.add_geom(name=f"{prefix}rake_tine{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      fromto=[0, -length, z, -0.08, -length - 0.03, z], size=[0.004, 0, 0], rgba=[0.3, 0.3, 0.3, 1], mass=0.02,
                      contype=1, conaffinity=1)
    rake.add_site(name=f"{prefix}rake_tip", pos=[-0.08, -length - 0.03, 0], size=[0.01, 0, 0], group=4, rgba=[1, 0.5, 0, 1])
    # the rake is welded into the hand: never collide it with that hand's own links
    for b in spec.bodies:
        if b.name.startswith(prefix) and b.name != f"{prefix}rake":
            spec.add_exclude(bodyname1=f"{prefix}rake", bodyname2=b.name)
    spec.add_exclude(bodyname1=f"{prefix}rake", bodyname2=f"{side}_wrist_roll_link")
    rake.add_site(name=f"{prefix}rake_end", pos=[0, -length, 0], size=[0.01, 0, 0], group=4, rgba=[1, 1, 0, 1])


def build(pose_name="bent", rake_length=0.0, rake_hand="right", hands=True, rake_tilt=0.0):
    spec = mujoco.MjSpec.from_file(R1_URDF)
    spec.modelname = f"r1_standing_{pose_name}"
    spec.compiler.discardvisual = False
    spec.compiler.fusestatic = False  # keep pelvis/leg bodies (and their mass) after the leg joints are removed
    freeze_legs(spec, LEG_POSES[pose_name])

    pelvis = spec.body("pelvis_link")
    # compile once to measure the sole height, then lift the pelvis
    m0 = spec.compile(); d0 = mujoco.MjData(m0); mujoco.mj_forward(m0, d0)
    zmin, where = lowest_point(m0, d0, lambda n: "ankle" in n or "knee" in n)
    pelvis.pos = [0, 0, -zmin]

    spec.visual.global_.offwidth = 1920; spec.visual.global_.offheight = 1080
    # floor + light + free camera
    w = spec.worldbody
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.15, 0.2], width=512, height=512)
    spec.add_material(name="grid", texrepeat=[4, 4], texuniform=True, reflectance=0.1).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1], material="grid", contype=1, conaffinity=1)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.55, 0.7, 0.9], rgb2=[0.15, 0.2, 0.3], width=512, height=3072)
    spec.visual.headlight.ambient[:] = [0.45, 0.45, 0.45]; spec.visual.headlight.diffuse[:] = [0.6, 0.6, 0.6]
    w.add_light(pos=[1, 1, 3], dir=[-0.3, -0.3, -1], castshadow=True)
    w.add_light(pos=[-1, -1, 3], dir=[0.3, 0.3, -1], castshadow=False)
    # props in front of the robot (visual references for VR practice; all static)
    w.add_geom(name="back_wall", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[3.0, 0, 1.25], size=[0.05, 3.0, 1.25], rgba=[0.85, 0.8, 0.7, 1], contype=0, conaffinity=0)
    w.add_geom(name="side_wall", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.5, 3.0, 1.25], size=[3.0, 0.05, 1.25], rgba=[0.75, 0.8, 0.85, 1], contype=0, conaffinity=0)
    w.add_geom(name="bench", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[1.6, -0.9, 0.40], size=[0.4, 0.3, 0.40], rgba=[0.6, 0.45, 0.3, 1], contype=1, conaffinity=1)
    w.add_geom(name="crate", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[1.8, 0.8, 0.2], size=[0.2, 0.2, 0.2], rgba=[0.8, 0.3, 0.2, 1], contype=1, conaffinity=1)
    w.add_geom(name="post", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[2.4, 0.0, 0.6], size=[0.04, 0.6, 0], rgba=[0.2, 0.6, 0.3, 1], contype=1, conaffinity=1)
    # a strip of "debris" on the floor in the raking zone (visual markers at 0.5 .. 1.2 m ahead)
    for i, x in enumerate(np.arange(0.5, 1.25, 0.1)):
        w.add_geom(name=f"mark{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[x, -0.2, 0.002], size=[0.03, 0.002, 0], rgba=[1.0, 0.85, 0.1, 1], contype=0, conaffinity=0)
    w.add_camera(name="front", pos=[2.5, 0, 1.0], xyaxes=[0, 1, 0, 0, 0, 1])
    w.add_camera(name="side", pos=[0, -2.5, 1.0], xyaxes=[1, 0, 0, 0, 0, 1])
    w.add_camera(name="iso", pos=[2.0, -1.8, 1.6], xyaxes=[0.67, 0.74, 0, -0.3, 0.27, 0.92])

    # sites + hands
    for side in ("left", "right"):
        wrist = add_sites(spec, side)
        if hands:
            attach_hand(spec, side, wrist)
    head = spec.body("head_yaw_link")
    # head camera: near the top-front of the head, looking forward (MuJoCo cameras look along -z, y up)
    head.add_site(name="head_cam", pos=[0.08, 0, 0.05], size=[0.005, 0, 0], group=4)
    head.add_camera(name="head_camera", pos=[0.08, 0, 0.05], xyaxes=[0, -1, 0, 0, 0, 1], fovy=80)  # ~80 deg matches televuer ImageBackground height 1.7 at 1 m (XR_TELEOP_PANEL_HEIGHT)

    if rake_length > 0:
        add_rake(spec, rake_hand, rake_length, tilt_deg=rake_tilt)

    # actuators (pure torque; the DDS bridge computes its own PD from the LowCmd kp/kd)
    for jn in UPPER_JOINTS:
        key = jn.replace("left_", "").replace("right_", "").replace("_joint", "")
        lim = EFFORT.get(jn, EFFORT.get(key, EFFORT["head"] if "head" in jn else 33))
        a = spec.add_actuator(name=jn, target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT, gear=[1, 0, 0, 0, 0, 0])
        a.ctrlrange = [-lim, lim]; a.ctrllimited = True
    # a little joint damping/armature so the model is stable when driven
    for j in spec.joints:
        if j.name in UPPER_JOINTS:
            j.damping = np.full(3, 0.5); j.armature = 0.01
        else:  # hand fingers (tiny inertias): generous armature/damping keeps the fingers numerically tame
            j.damping = np.full(3, 0.2); j.armature = 0.005
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.timestep = 0.002

    model = spec.compile()
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    # convex-hull overlaps between neighbouring links in the neutral pose are modelling artefacts (e.g. pelvis vs
    # waist roll link, hand base vs thumb): exclude those body pairs so they do not inject spurious contact forces
    excluded = set()
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        n1, n2 = model.body(b1).name, model.body(b2).name
        if "floor" in (model.geom(c.geom1).name, model.geom(c.geom2).name):
            continue
        key = tuple(sorted((n1, n2)))
        if key not in excluded:
            excluded.add(key); spec.add_exclude(bodyname1=key[0], bodyname2=key[1])
    print("excluded neutral-pose contact pairs:", sorted(excluded))
    # keyframe: upper-body neutral arms-down pose (shoulder pitch 0.3, elbow 0.9 like the Isaac init), hands open
    q = np.zeros(model.nq)
    for name, val in {"left_shoulder_pitch_joint": 0.3, "right_shoulder_pitch_joint": 0.3, "left_shoulder_roll_joint": 0.15,
                      "right_shoulder_roll_joint": -0.15, "left_elbow_joint": 0.9, "right_elbow_joint": 0.9}.items():
        q[model.joint(name).qposadr[0]] = val
    key = spec.add_key(name="standing", qpos=q)
    model = spec.compile()
    return spec, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", choices=list(LEG_POSES), default="bent")
    ap.add_argument("--rake-length", type=float, default=0.0)
    ap.add_argument("--rake-hand", choices=["left", "right"], default="right")
    ap.add_argument("--rake-tilt", type=float, default=0.0, help="grip angle (deg): 0 handle across the palm, 90 along the forearm")
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    spec, model = build(args.pose, args.rake_length, args.rake_hand, hands=not args.no_hands, rake_tilt=args.rake_tilt)
    out = args.out or os.path.join(HERE, "out", f"r1_scene_{args.pose}.xml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # write MJCF with absolute mesh file paths so it loads from anywhere (R1 meshes and hand meshes live in
    # different directories, so a single meshdir cannot serve both)
    r1_dir = os.path.abspath(os.path.join(HERE, "..", "unitree_ros", "robots", "r1_description"))
    for mesh in spec.meshes:
        root = HAND_DIR if mesh.name.startswith(("L_", "R_")) else r1_dir
        mesh.file = os.path.abspath(os.path.join(root, mesh.file))
    spec.meshdir = ""
    with open(out, "w") as f:
        f.write(spec.to_xml())
    data = mujoco.MjData(model); mujoco.mj_resetDataKeyframe(model, data, 0); mujoco.mj_forward(model, data)
    print(f"wrote {out}")
    print(f"nq={model.nq} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom} mass={model.body_subtreemass[1]:.2f} kg")
    print("joints:", [model.joint(i).name for i in range(model.njnt)])
    zt, _ = lowest_point(model, data, lambda n: True)
    print(f"pelvis z = {data.body('pelvis_link').xpos[2]:.4f} m, lowest point z = {zt:.4f} m")
    top = max(((data.geom_xmat[g].reshape(3, 3) @ model.mesh_vert[model.mesh_vertadr[model.geom_dataid[g]]:model.mesh_vertadr[model.geom_dataid[g]] + model.mesh_vertnum[model.geom_dataid[g]]].T).T + data.geom_xpos[g])[:, 2].max()
              for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
    print(f"top of head z = {top:.4f} m")
    for s in ["L_wrist", "L_flange", "L_ee", "L_grasp", "R_wrist", "R_ee", "R_grasp", "head_cam"]:
        print(f"  site {s:9s} {data.site(s).xpos.round(4)}")
    for b in ["left_shoulder_pitch_link", "waist_yaw_link", "head_yaw_link"]:
        print(f"  body {b:26s} {data.body(b).xpos.round(4)}")


if __name__ == "__main__":
    main()
