"""
Variant of joint grasp data recorder that allows freezing one arm
while moving the other.

Controls:
- Press 'P' to toggle freeze/unfreeze of the LEFT arm.
- Press 'H' to toggle freeze/unfreeze of the RIGHT arm.
"""

import os
import time
import h5py
import argparse
import numpy as np
from tqdm import tqdm
import yaml

from constants_aloha2 import DT, START_ARM_POSE, TASK_CONFIGS
from constants_aloha2 import MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
from robot_utils import Recorder, ImageRecorder, get_arm_gripper_positions
from robot_utils import move_arms, torque_on, torque_off, move_grippers
from real_env import make_real_env, get_action

from interbotix_xs_modules.arm import InterbotixManipulatorXS

from record_episodes import get_auto_index, print_dt_diagnosis
import sys

from pynput import keyboard

sys.path.append('/home/xuhang/interbotix_ws/src/pddlstream_aloha/')

from examples.pybullet.aloha_real.scripts.ros_openworld_base import observation_to_file
from examples.pybullet.aloha_real.scripts.aloha_tamp_constants import PERCEPT_ARM_POSE, qpos_to_eetrans, RBT_ID
from examples.pybullet.aloha_real.openworld_aloha.open_world_utils import get_camera_mappings

import IPython

e = IPython.embed

import open3d as o3d
import cv2
import json

GRIPPER_THRETH = 0.03

# Global freeze flags for left and right arms.
left_frozen = False
right_frozen = False

# Global references to master robots to allow torque control in key handler.
master_bot_left = None
master_bot_right = None


def _on_key_press(key):
    """Keyboard callback to toggle per-arm freeze flags and lock/unlock master arms."""
    global left_frozen, right_frozen, master_bot_left, master_bot_right
    try:
        if key == keyboard.KeyCode.from_char('p') or key == keyboard.KeyCode.from_char('P'):
            left_frozen = not left_frozen
            state = "frozen" if left_frozen else "unfrozen"
            print(f"[Freeze] Left arm {state}.")
            # When freezing/unfreezing the left puppet arm, also fix/release the left master arm.
            if master_bot_left is not None:
                if left_frozen:
                    torque_on(master_bot_left)
                else:
                    torque_off(master_bot_left)
        elif key == keyboard.KeyCode.from_char('h') or key == keyboard.KeyCode.from_char('H'):
            right_frozen = not right_frozen
            state = "frozen" if right_frozen else "unfrozen"
            print(f"[Freeze] Right arm {state}.")
            # When freezing/unfreezing the right puppet arm, also fix/release the right master arm.
            if master_bot_right is not None:
                if right_frozen:
                    torque_on(master_bot_right)
                else:
                    torque_off(master_bot_right)
    except AttributeError:
        # Non-character key pressed; ignore.
        pass


def initialize_bots(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right, **kwargs):
    """Move all 4 robots to a pose where it is easy to start demonstration."""
    puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
    puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    master_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")

    puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    master_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")

    torque_on(puppet_bot_left)
    torque_on(master_bot_left)
    torque_on(puppet_bot_right)
    torque_on(master_bot_right)


def opening_ceremony(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right, **kwargs):
    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms(
        [master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right],
        [start_arm_qpos] * 4,
        move_time=1.5,
    )

    # press gripper to start data collection
    # disable torque for only gripper joint of master robot to allow user movement
    master_bot_left.dxl.robot_torque_enable("single", "gripper", False)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", False)
    print("Close both master grippers to start recording.")
    close_thresh = 0.01244
    pressed = False
    while not pressed:
        gripper_pos_left = get_arm_gripper_positions(master_bot_left)
        gripper_pos_right = get_arm_gripper_positions(master_bot_right)
        if (gripper_pos_left < close_thresh) and (gripper_pos_right < close_thresh):
            pressed = True
        time.sleep(DT / 10)
    torque_off(master_bot_left)
    torque_off(master_bot_right)
    print("Started!")


def sense_tabletop(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right, cam_dir_mapping=None, cam_config=None):
    # before demo, do perception
    move_arms([puppet_bot_left, puppet_bot_right], [PERCEPT_ARM_POSE] * 2, move_time=1.5)
    # move grippers to starting position
    move_grippers(
        [master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right],
        [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE] * 2,
        move_time=0.5,
    )

    for k, v in cam_dir_mapping.items():
        if not os.path.exists(v):
            os.makedirs(v)

    observation_to_file(cam_dir_mapping, cam_config=cam_config)

    color_imgs = {}
    depth_imgs = {}
    camera_infos = {}

    for k, v in cam_dir_mapping.items():
        color_file = os.path.join(v, "color_image.png")
        color_img = cv2.imread(color_file)
        color_imgs[k] = color_img

        depth_file = os.path.join(v, "depth_image.png")
        depth_img_mm = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        depth_img = depth_img_mm.astype(np.float32) / 1000.0
        depth_imgs[k] = depth_img

        camera_intrinsics_file = os.path.join(v, "color_info.json")
        with open(camera_intrinsics_file, "r") as f:
            camera_info_color = json.load(f)
        camera_infos[k] = camera_info_color

    return color_imgs, depth_imgs, camera_infos


def capture_one_episode(
    dt,
    max_timesteps,
    camera_names,
    dataset_dir,
    dataset_name,
    overwrite,
    cam_dir_mapping=None,
    cam_config=None,
    **kwargs,
):
    global left_frozen, right_frozen, master_bot_left, master_bot_right

    print(f"Dataset name: {dataset_name}")

    # source of data
    master_bot_left = InterbotixManipulatorXS(
        robot_model="wx250s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="master_left",
        init_node=True,
    )
    master_bot_right = InterbotixManipulatorXS(
        robot_model="wx250s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="master_right",
        init_node=False,
    )
    env = make_real_env(init_node=False, setup_robots=False, camera_names=camera_names)

    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f"Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.")
        exit()

    initialize_bots(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right, **kwargs)

    # do perception, record the point cloud and the mesh
    start_color_imgs, start_depth_imgs, camera_infos = sense_tabletop(
        master_bot_left,
        master_bot_right,
        env.puppet_bot_left,
        env.puppet_bot_right,
        cam_dir_mapping=cam_dir_mapping,
        cam_config=cam_config,
    )

    opening_ceremony(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right)

    # Reset freeze flags at the start of each episode.
    left_frozen = False
    right_frozen = False

    # Keyboard listener for per-arm freeze toggling.
    listener = keyboard.Listener(on_press=_on_key_press)
    listener.start()

    # Data collection
    ts = env.reset(fake=True)
    timesteps = [ts]
    actions = []
    actual_dt_history = []
    left_frozen_hist = []
    right_frozen_hist = []

    # Last applied action, used to hold pose for frozen arm segments.
    last_action = None

    # Indices for left/right arm+gripper in the 14D action vector.
    # Assumes 0:7 = left, 7:14 = right.
    left_slice = slice(0, 7)
    right_slice = slice(7, 14)

    for t in tqdm(range(max_timesteps)):
        t0 = time.time()
        action = get_action(master_bot_left, master_bot_right)
        t1 = time.time()

        if last_action is None:
            last_action = np.array(action, copy=True)

        # Apply per-arm freeze: keep frozen arm segment at last applied command.
        if left_frozen:
            action[left_slice] = last_action[left_slice]
        if right_frozen:
            action[right_slice] = last_action[right_slice]

        ts = env.step(action)
        t2 = time.time()

        timesteps.append(ts)
        actions.append(np.array(action, copy=True))
        actual_dt_history.append([t0, t1, t2])
        left_frozen_hist.append(bool(left_frozen))
        right_frozen_hist.append(bool(right_frozen))

        last_action = np.array(action, copy=True)

    # Stop keyboard listener after data collection.
    listener.stop()

    # Torque on both master bots
    torque_on(master_bot_left)
    torque_on(master_bot_right)

    # do perception again for final grasp detection
    end_color_imgs, end_depth_imgs, _ = sense_tabletop(
        master_bot_left,
        master_bot_right,
        env.puppet_bot_left,
        env.puppet_bot_right,
        cam_dir_mapping=cam_dir_mapping,
        cam_config=cam_config,
    )

    # Open puppet grippers
    move_grippers(
        [env.puppet_bot_left, env.puppet_bot_right],
        [PUPPET_GRIPPER_JOINT_OPEN] * 2,
        move_time=0.5,
    )

    freq_mean = print_dt_diagnosis(actual_dt_history)
    if freq_mean < 42:
        return False

    """
    For each timestep:
    observations
    - images
        - cam_high          (480, 640, 3) 'uint8'
        - cam_low           (480, 640, 3) 'uint8'
        - cam_left_wrist    (480, 640, 3) 'uint8'
        - cam_right_wrist   (480, 640, 3) 'uint8'
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'

    action                  (14,)         'float64'
    """

    data_dict = {
        "/observations/qpos": [],
        "/observations/qvel": [],
        "/observations/effort": [],
        "/action": [],
    }
    for cam_name in camera_names:
        data_dict[f"/observations/images/{cam_name}"] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict["/observations/qpos"].append(ts.observation["qpos"])
        data_dict["/observations/qvel"].append(ts.observation["qvel"])
        data_dict["/observations/effort"].append(ts.observation["effort"])
        data_dict["/action"].append(action)
        for cam_name in camera_names:
            data_dict[f"/observations/images/{cam_name}"].append(
                ts.observation["images"][cam_name]
            )

    # Freeze flags per timestep.
    data_dict["/left_frozen"] = np.array(left_frozen_hist, dtype=bool)
    data_dict["/right_frozen"] = np.array(right_frozen_hist, dtype=bool)

    for rs_cam in cam_dir_mapping.keys():
        data_dict[f"/color_img_{rs_cam}"] = [
            start_color_imgs[rs_cam],
            end_color_imgs[rs_cam],
        ]
        data_dict[f"/depth_img_{rs_cam}"] = [
            start_depth_imgs[rs_cam],
            end_depth_imgs[rs_cam],
        ]

    # HDF5
    t0 = time.time()
    with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs["sim"] = False
        obs = root.create_group("observations")
        image = obs.create_group("images")
        for cam_name in camera_names:
            if "depth" in cam_name:
                _ = image.create_dataset(
                    cam_name,
                    (max_timesteps, 480, 640),
                    dtype="float32",
                    chunks=(1, 480, 640),
                )
            else:
                _ = image.create_dataset(
                    cam_name,
                    (max_timesteps, 480, 640, 3),
                    dtype="uint8",
                    chunks=(1, 480, 640, 3),
                )

        _ = obs.create_dataset("qpos", (max_timesteps, 14))
        _ = obs.create_dataset("qvel", (max_timesteps, 14))
        _ = obs.create_dataset("effort", (max_timesteps, 14))
        _ = root.create_dataset("action", (max_timesteps, 14))

        # Per-timestep freeze flags for left/right arms.
        _ = root.create_dataset("left_frozen", (max_timesteps,), dtype="bool")
        _ = root.create_dataset("right_frozen", (max_timesteps,), dtype="bool")

        for rs_cam in cam_dir_mapping.keys():
            color_img = start_color_imgs[rs_cam]
            depth_img = start_depth_imgs[rs_cam]
            _ = root.create_dataset(
                f"color_img_{rs_cam}",
                (2, *color_img.shape),
                dtype=color_img.dtype,
            )
            _ = root.create_dataset(
                f"depth_img_{rs_cam}",
                (2, *depth_img.shape),
                dtype=depth_img.dtype,
            )

        for name, array in data_dict.items():
            root[name][...] = array

        for rs_cam in cam_dir_mapping.keys():
            _ = root.create_dataset(
                f"camera_info_{rs_cam}",
                data=json.dumps(camera_infos[rs_cam]),
            )

    print(f"Saving: {time.time() - t0:.1f} secs")

    return True


def main(args):
    task_config = TASK_CONFIGS[args["task_name"]]
    dataset_dir = task_config["dataset_dir"]
    max_timesteps = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    if args["episode_idx"] is not None:
        episode_idx = args["episode_idx"]
    else:
        episode_idx = get_auto_index(dataset_dir)
    overwrite = True

    dataset_name = f"episode_{episode_idx}"
    print(dataset_name + "\n")

    # Configure RealSense camera directories using shared open_world_utils helper
    # Load camera parameters directly from sgBase.yaml
    sgbase_path = "/home/xuhang/interbotix_ws/src/pddlstream_aloha/examples/pybullet/aloha_real/openworld_aloha/configs/sgBase.yaml"
    with open(sgbase_path, "r") as f:
        cam_para = yaml.load(f, Loader=yaml.FullLoader)

    cam_dir_mapping, _, _ = get_camera_mappings(cam_para)

    while True:
        is_healthy = capture_one_episode(
            DT,
            max_timesteps,
            camera_names,
            dataset_dir,
            dataset_name,
            overwrite,
            task_name=args["task_name"],
            episode_idx=args["episode_idx"],
            cam_dir_mapping=cam_dir_mapping,
            cam_config=cam_para,
        )
        if is_healthy:
            break


def get_auto_index(dataset_dir, dataset_name_prefix="", data_suffix="hdf5"):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx + 1):
        if not os.path.isfile(
            os.path.join(dataset_dir, f"{dataset_name_prefix}episode_{i}.{data_suffix}")
        ):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print(
        f"Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}"
    )
    return freq_mean


def debug():
    print("====== Debug mode ======")
    recorder = Recorder("right", is_debug=True)
    image_recorder = ImageRecorder(init_node=False, is_debug=True)
    while True:
        time.sleep(1)
        recorder.print_diagnostics()
        image_recorder.print_diagnostics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        action="store",
        type=str,
        help="Task name.",
        default="aloha_transfer_tape",
        required=False,
    )
    parser.add_argument(
        "--episode_idx",
        action="store",
        type=int,
        help="Episode index.",
        default=99,
        required=False,
    )
    main(vars(parser.parse_args()))
    # debug()

