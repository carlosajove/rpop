#!/usr/bin/env python
"""MuJoCo stand-in for the real R1 for Unitree xr_teleoperate (run with `--sim`).

Speaks the same DDS interface that xr_teleoperate uses in simulation mode (domain 1):
  subscribe rt/lowcmd            (unitree_hg LowCmd_, 35 motor slots, R1_A5_JointIndex layout)
  publish   rt/lowstate          (unitree_hg LowState_)                         at --state-hz
  subscribe rt/brainco/{left,right}/cmd    (unitree_go MotorCmds_, 6 normalised 0..1 finger commands)
  publish   rt/brainco/{left,right}/state  (unitree_go MotorStates_)
  subscribe rt/reset_pose/cmd    (std_msgs String_) -> reset the scene
and serves the head camera the way teleimager does (ZMQ config responder on :60000, JPEG stream on :55555),
so xr_teleoperate needs no code changes:

  conda activate tv; cd ~/projects/unitree-r1/r1_mujoco
  python r1_mujoco_sim.py [--rake-length 0.8]                # terminal 1
  cd ~/projects/unitree-r1/xr_teleoperate/teleop
  python teleop_hand_and_arm.py --arm R1_A5 --ee brainco --sim --img-server-ip 127.0.0.1   # terminal 2

Joint torques are PD from the LowCmd (kp, kd, q, dq, tau) exactly as the real motor controller would apply them.
"""
import argparse, os, sys, threading, time, json
import numpy as np

import mujoco

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_, unitree_go_msg_dds__MotorState_

HERE = os.path.dirname(os.path.abspath(__file__))

# R1_A5_JointIndex (xr_teleoperate/teleop/robot_control/robot_arm.py) -> MuJoCo joint name
SLOT_TO_JOINT = {
    0: "left_hip_pitch_joint", 1: "left_hip_roll_joint", 2: "left_hip_yaw_joint", 3: "left_knee_joint", 4: "left_ankle_pitch_joint", 5: "left_ankle_roll_joint",
    6: "right_hip_pitch_joint", 7: "right_hip_roll_joint", 8: "right_hip_yaw_joint", 9: "right_knee_joint", 10: "right_ankle_pitch_joint", 11: "right_ankle_roll_joint",
    12: "waist_roll_joint", 13: "waist_yaw_joint",
    15: "left_shoulder_pitch_joint", 16: "left_shoulder_roll_joint", 17: "left_shoulder_yaw_joint", 18: "left_elbow_joint", 19: "left_wrist_roll_joint",
    22: "right_shoulder_pitch_joint", 23: "right_shoulder_roll_joint", 24: "right_shoulder_yaw_joint", 25: "right_elbow_joint", 26: "right_wrist_roll_joint",
    29: "head_pitch_joint", 30: "head_yaw_joint",
}
NUM_MOTORS = 35
# legs are frozen in the model: report the standing pose the scene was built with
LEG_STANDING = {"hip_pitch": -0.20, "knee": 0.42, "ankle_pitch": -0.23}

# BrainCo Revo2: 6 motor slots, normalised 0 (open) .. 1 (closed).  Slot -> (proximal joint, range max, coupled distal joint, distal range max)
BRAINCO_SLOTS = [
    ("thumb_metacarpal", 1.5184, None, None),
    ("thumb_proximal", 1.0472, "thumb_distal", 1.0472),
    ("index_proximal", 1.4661, "index_distal", 1.693),
    ("middle_proximal", 1.4661, "middle_distal", 1.693),
    ("ring_proximal", 1.4661, "ring_distal", 1.693),
    ("pinky_proximal", 1.4661, "pinky_distal", 1.693),
]


class Bridge:
    def __init__(self, args):
        self.args = args
        scene = args.scene or (os.path.join(HERE, "out", f"r1_rake_right_{args.rake_length}.xml") if args.rake_length else os.path.join(HERE, "out", "r1_scene_bent.xml"))
        if not os.path.exists(scene):
            sys.exit(f"scene not found: {scene} (build it with build_r1_scene.py)")
        self.m = mujoco.MjModel.from_xml_path(scene)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = 1.0 / args.physics_hz
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        print(f"[sim] scene {scene}: nq={self.m.nq} nu={self.m.nu} dt={self.m.opt.timestep}")

        # slot -> (qpos adr, dof adr, actuator id, effort limit)
        self.slots = {}
        for slot, jn in SLOT_TO_JOINT.items():
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            aid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
            if jid < 0 or aid < 0:
                continue  # frozen leg joint
            self.slots[slot] = (self.m.jnt_qposadr[jid], self.m.jnt_dofadr[jid], aid, float(self.m.actuator_ctrlrange[aid, 1]))
        print(f"[sim] driven slots: {sorted(self.slots)}")
        # finger joints
        self.fingers = {}
        for side, pre in (("left", "L_"), ("right", "R_")):
            lst = []
            for prox, pmax, dist, dmax in BRAINCO_SLOTS:
                jp = self.m.joint(f"{pre}{side}_{prox}_joint")
                jd = self.m.joint(f"{pre}{side}_{dist}_joint") if dist else None
                lst.append((jp.qposadr[0], jp.dofadr[0], pmax, jd.qposadr[0] if jd else None, jd.dofadr[0] if jd else None, dmax))
            self.fingers[side] = lst

        # command state
        self.lock = threading.Lock()
        self.lowcmd = None
        self.lowcmd_time = 0.0
        self.hand_cmd = {"left": np.zeros(6), "right": np.zeros(6)}
        self.reset_requested = False

        # DDS
        ChannelFactoryInitialize(args.domain, args.interface)
        self.sub_lowcmd = ChannelSubscriber("rt/lowcmd", hg_LowCmd); self.sub_lowcmd.Init(self._on_lowcmd, 1)
        self.sub_lowcmd_motion = ChannelSubscriber("rt/arm_sdk", hg_LowCmd); self.sub_lowcmd_motion.Init(self._on_lowcmd, 1)
        self.pub_lowstate = ChannelPublisher("rt/lowstate", hg_LowState); self.pub_lowstate.Init()
        self.sub_hand = {}
        self.pub_hand = {}
        for side in ("left", "right"):
            s = ChannelSubscriber(f"rt/brainco/{side}/cmd", MotorCmds_); s.Init(lambda msg, side=side: self._on_hand(side, msg), 1); self.sub_hand[side] = s
            p = ChannelPublisher(f"rt/brainco/{side}/state", MotorStates_); p.Init(); self.pub_hand[side] = p
        self.sub_reset = ChannelSubscriber("rt/reset_pose/cmd", String_); self.sub_reset.Init(self._on_reset, 1)
        self.pub_simstate = ChannelPublisher("rt/sim_state_cmd", String_); self.pub_simstate.Init()

        self.lowstate = unitree_hg_msg_dds__LowState_()
        self.lowstate.mode_machine = 8  # arbitrary non-zero id, echoed back by the arm controller
        self.hand_state = {side: MotorStates_() for side in ("left", "right")}
        for side in ("left", "right"):
            self.hand_state[side].states = [unitree_go_msg_dds__MotorState_() for _ in range(6)]

        # camera server (teleimager protocol)
        self.cam_config = None
        if not args.no_camera:
            from teleimager.image_client import ZMQ_Responser, ZMQ_PublisherManager
            h, w = args.cam_height, args.cam_width
            self.cam_config = {
                "head_camera": dict(enable_zmq=True, zmq_port=55555, enable_webrtc=False, webrtc_port=60001, webrtc_codec="h264", type="opencv",
                                    image_shape=[h, 2 * w if args.binocular else w], binocular=bool(args.binocular), fps=args.cam_fps, video_id=0, serial_number=None, physical_path=None),
                "left_wrist_camera": dict(enable_zmq=False, zmq_port=55556, enable_webrtc=False, webrtc_port=60002, webrtc_codec="h264", type="opencv", image_shape=[480, 640], binocular=False, fps=30, video_id=None, serial_number=None, physical_path=None),
                "right_wrist_camera": dict(enable_zmq=False, zmq_port=55557, enable_webrtc=False, webrtc_port=60003, webrtc_codec="h264", type="opencv", image_shape=[480, 640], binocular=False, fps=30, video_id=None, serial_number=None, physical_path=None),
            }
            self.responser = ZMQ_Responser(self.cam_config, port=60000)
            self.zmq_pub = ZMQ_PublisherManager.get_instance()
            self.renderer = mujoco.Renderer(self.m, h, w)
            self.cam = mujoco.MjvCamera(); self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED; self.cam.fixedcamid = self.m.camera("head_camera").id
            import cv2; self.cv2 = cv2
            print(f"[sim] camera server: config on tcp://*:60000, head JPEG stream on tcp://*:55555 ({self.cam_config['head_camera']['image_shape']})")

    # ---------------- DDS callbacks ----------------
    def _on_lowcmd(self, msg: hg_LowCmd):
        with self.lock:
            self.lowcmd = msg; self.lowcmd_time = time.time()

    def _on_hand(self, side, msg: MotorCmds_):
        q = np.array([c.q for c in msg.cmds[:6]], float)
        if len(q) == 6:
            with self.lock:
                self.hand_cmd[side] = np.clip(q, 0, 1)

    def _on_reset(self, msg: String_):
        print(f"[sim] reset requested: {msg.data}")
        self.reset_requested = True

    # ---------------- control ----------------
    def apply_lowcmd(self):
        with self.lock:
            cmd = self.lowcmd; hand = {k: v.copy() for k, v in self.hand_cmd.items()}
        d = self.d
        d.ctrl[:] = 0.0
        if cmd is not None and time.time() - self.lowcmd_time < 0.5:
            for slot, (qa, da, aid, lim) in self.slots.items():
                mc = cmd.motor_cmd[slot]
                if mc.mode == 0 and mc.kp == 0 and mc.kd == 0:
                    continue
                tau = mc.kp * (mc.q - d.qpos[qa]) + mc.kd * (mc.dq - d.qvel[da]) + mc.tau
                d.ctrl[aid] = float(np.clip(tau, -lim, lim))
        else:
            # nobody is commanding: hold the current pose softly so the arms do not fall
            for slot, (qa, da, aid, lim) in self.slots.items():
                tau = 40.0 * (self.d_hold[qa] - d.qpos[qa]) - 2.0 * d.qvel[da]
                d.ctrl[aid] = float(np.clip(tau, -lim, lim))
        # fingers: kinematic (position) coupling, normalised 0..1 -> joint range
        for side, lst in self.fingers.items():
            for k, (qa, da, pmax, qda, dda, dmax) in enumerate(lst):
                target = hand[side][k] * pmax
                d.qpos[qa] += np.clip(target - d.qpos[qa], -0.05, 0.05); d.qvel[da] = 0.0
                if qda is not None:
                    d.qpos[qda] = d.qpos[qa] / pmax * dmax; d.qvel[dda] = 0.0

    def publish_state(self):
        d = self.d
        ls = self.lowstate
        ls.tick = int(d.time * 1000) & 0xFFFFFFFF
        for slot in range(NUM_MOTORS):
            ms = ls.motor_state[slot]
            if slot in self.slots:
                qa, da, _, _ = self.slots[slot]
                ms.q = float(d.qpos[qa]); ms.dq = float(d.qvel[da]); ms.tau_est = float(d.actuator_force[self.slots[slot][2]]); ms.mode = 1
            elif slot in SLOT_TO_JOINT:  # frozen leg
                jn = SLOT_TO_JOINT[slot]
                key = jn.split("_", 1)[1].replace("_joint", "")
                ms.q = float(LEG_STANDING.get(key, 0.0)); ms.dq = 0.0; ms.mode = 1
            else:
                ms.q = 0.0; ms.dq = 0.0
        # gravity direction in the pelvis frame (IMU) for completeness: upright
        ls.imu_state.quaternion[:] = [1.0, 0.0, 0.0, 0.0]
        self.pub_lowstate.Write(ls)
        for side, lst in self.fingers.items():
            st = self.hand_state[side]
            for k, (qa, da, pmax, *_rest) in enumerate(lst):
                st.states[k].q = float(np.clip(d.qpos[qa] / pmax, 0, 1)); st.states[k].dq = 0.0
            self.pub_hand[side].Write(st)

    def publish_camera(self):
        if self.cam_config is None:
            return
        cam_body_id = self.m.cam_bodyid[self.cam.fixedcamid]
        if self.args.binocular:
            # stereo: shift the camera along its own x axis by +/- half the IPD
            imgs = []
            base_pos = self.m.cam_pos[self.cam.fixedcamid].copy()
            xaxis = self.d.cam_xmat[self.cam.fixedcamid].reshape(3, 3)[:, 0]
            R = self.d.xmat[cam_body_id].reshape(3, 3)
            for sgn in (-1, 1):
                self.m.cam_pos[self.cam.fixedcamid] = base_pos + R.T @ (sgn * 0.5 * self.args.ipd * xaxis)
                mujoco.mj_camlight(self.m, self.d)
                self.renderer.update_scene(self.d, camera=self.cam); imgs.append(self.renderer.render())
            self.m.cam_pos[self.cam.fixedcamid] = base_pos; mujoco.mj_camlight(self.m, self.d)
            rgb = np.concatenate(imgs, axis=1)
        else:
            self.renderer.update_scene(self.d, camera=self.cam); rgb = self.renderer.render()
        ok, buf = self.cv2.imencode(".jpg", self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR), [int(self.cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            self.zmq_pub.publish(buf.tobytes(), 55555)

    def sim_state_json(self):
        st = {"robot": {jn: float(self.d.qpos[self.m.joint(jn).qposadr[0]]) for jn in SLOT_TO_JOINT.values() if mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, jn) >= 0}}
        for pre in ("L_", "R_"):
            for s in ("ee", "grasp"):
                st[f"{pre}{s}"] = self.d.site(f"{pre}{s}").xpos.round(4).tolist()
        for s in ("R_rake_end", "L_rake_end"):
            if mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, s) >= 0:
                st[s] = self.d.site(s).xpos.round(4).tolist()
        return json.dumps(st)

    def run(self):
        args = self.args
        self.d_hold = self.d.qpos.copy()
        viewer = None
        if not args.headless:
            from mujoco import viewer as mj_viewer
            viewer = mj_viewer.launch_passive(self.m, self.d, show_left_ui=False, show_right_ui=False)
            viewer.cam.distance = 2.5; viewer.cam.elevation = -15; viewer.cam.azimuth = 150; viewer.cam.lookat[:] = [0.2, 0, 0.8]
            viewer.opt.sitegroup[4] = True
        steps_per_state = max(1, int(round(args.physics_hz / args.state_hz)))
        steps_per_cam = max(1, int(round(args.physics_hz / args.cam_fps)))
        steps_per_viewer = max(1, int(round(args.physics_hz / 60)))
        n = 0; t_wall = time.perf_counter(); t_report = t_wall; last_cmd_flag = None
        print("[sim] running. Ctrl-C to stop.")
        try:
            while viewer is None or viewer.is_running():
                if self.reset_requested:
                    mujoco.mj_resetDataKeyframe(self.m, self.d, 0); self.d_hold = self.d.qpos.copy(); self.reset_requested = False
                self.apply_lowcmd()
                mujoco.mj_step(self.m, self.d)
                n += 1
                if n % steps_per_state == 0:
                    self.publish_state()
                if n % steps_per_cam == 0:
                    self.publish_camera()
                    self.pub_simstate.Write(String_(data=self.sim_state_json()))
                if viewer is not None and n % steps_per_viewer == 0:
                    viewer.sync()
                # real-time pacing
                t_wall += self.m.opt.timestep
                dt = t_wall - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
                elif dt < -0.5:
                    t_wall = time.perf_counter()  # fell behind; resync
                if time.perf_counter() - t_report > 2.0:
                    t_report = time.perf_counter()
                    flag = self.lowcmd is not None and time.time() - self.lowcmd_time < 0.5
                    if flag != last_cmd_flag:
                        print(f"[sim] lowcmd {'ACTIVE' if flag else 'absent (holding pose)'}"); last_cmd_flag = flag
                    if args.verbose:
                        print(f"[sim] t={self.d.time:7.2f}  L_ee={self.d.site('L_ee').xpos.round(3)} R_ee={self.d.site('R_ee').xpos.round(3)}")
        except KeyboardInterrupt:
            pass
        finally:
            if viewer is not None:
                viewer.close()
            if self.cam_config is not None:
                self.responser.stop(); self.zmq_pub.close(); self.renderer.close()
            print("[sim] stopped")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None, help="MJCF from build_r1_scene.py (default out/r1_scene_bent.xml or the rake scene)")
    ap.add_argument("--rake-length", default=None, help="use out/r1_rake_right_<L>.xml, e.g. 0.8")
    ap.add_argument("--domain", type=int, default=1, help="DDS domain (xr_teleoperate --sim uses 1)")
    ap.add_argument("--interface", default=None, help="network interface for DDS (default: auto)")
    ap.add_argument("--physics-hz", type=float, default=500.0)
    ap.add_argument("--state-hz", type=float, default=250.0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=480)
    ap.add_argument("--cam-fps", type=float, default=30.0)
    ap.add_argument("--binocular", type=int, default=1, help="1: stereo head image (2*width), 0: mono")
    ap.add_argument("--ipd", type=float, default=0.064)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    Bridge(args).run()


if __name__ == "__main__":
    main()
