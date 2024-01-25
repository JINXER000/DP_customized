if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)


import os
import h5py
from typing import Dict, List
import torch
import numpy as np
import copy
from tqdm import tqdm
import zarr
import os
import shutil
import copy
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
import matplotlib.pyplot as plt
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.env.aloha.constants import (
    JOINT_NAMES, DT, vx300s, LEFT_BASE_POSE, RIGHT_BASE_POSE,
    GRIPPER_EPSILON, EE_VEL_EPSILONE, EE_DIST_BOUND
)

import modern_robotics as mr
import cv2

register_codecs()


class AlohaImageDataset(BaseImageDataset):
    def __init__(
        self,
        dataset_dir: str,
        shape_meta: dict,
        num_episodes=50,
        camera_names=["top"],
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        n_obs_steps=None,
        max_train_episodes=None,
        use_cache=False,
        task="sim_transfer_cube_scripted"
    ):
        super().__init__()

        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_dir + "/" + task + ".zarr.zip"
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print("Cache does not exist. Creating!")
                        replay_buffer = _convert_to_replay(
                            num_episodes=num_episodes,
                            dataset_dir=dataset_dir,
                            camera_names=camera_names,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore(),
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("Loaded!")
        else:
            replay_buffer = self.load_data(
                num_episodes=num_episodes,
                dataset_dir=dataset_dir,
                camera_names=camera_names,
            )

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        )

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.camera_names = camera_names

    def load_data(
        self,
        num_episodes,
        dataset_dir,
        camera_names,
    ):
        replay_buffer = ReplayBuffer.create_empty_numpy()
        for i in tqdm(range(num_episodes)):  # num_episodes
            dataset_path = os.path.join(dataset_dir, f"episode_{i}.hdf5")
            with h5py.File(dataset_path, "r") as root:
                qpos = root["/observations/qpos"][()]
                qvel = root["/observations/qvel"][()]
                action = root["/action"][()]
                # new axis for different cameras
                all_cam_images = []
                for cam_name in camera_names:
                    all_cam_images.append(root[f"/observations/images/{cam_name}"][()])
                all_cam_images = np.stack(all_cam_images, axis=0)
            episode = {
                "qpos": qpos,
                "action": action,
                "images": all_cam_images.squeeze(),  # XXX: assume only one camera
            }
            replay_buffer.add_episode(episode)

        return replay_buffer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "qpos": self.replay_buffer["qpos"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["images"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = (
                np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            # T,C,H,W
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }
        return torch_data


def _convert_to_replay(
    store,
    shape_meta,
    dataset_dir,
    num_episodes,
    camera_names,
    n_workers=None,
    max_inflight_tasks=None,
):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    # count total steps
    episode_ends = list()
    prev_end = 0
    for i in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{i}.hdf5")
        with h5py.File(dataset_path, "r") as demo:
            episode_length = demo["/action"].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
    n_steps = episode_ends[-1]
    episode_starts = [0] + episode_ends[:-1]
    _ = meta_group.array(
        "episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )

    # save lowdim data
    for key in tqdm(lowdim_keys + ["action"], desc="Loading lowdim data"):
        data_key = "observations/" + key
        if key == "action":
            data_key = "action"
        this_data = list()
        for i in range(num_episodes):
            dataset_path = os.path.join(dataset_dir, f"episode_{i}.hdf5")
            with h5py.File(dataset_path, "r") as demo:
                this_data.append(demo[data_key][:].astype(np.float32))
        this_data = np.concatenate(this_data, axis=0)
        if key == "action":
            assert this_data.shape == (n_steps,) + tuple(shape_meta["action"]["shape"])
        else:
            assert this_data.shape == (n_steps,) + tuple(
                shape_meta["obs"][key]["shape"]
            )
        _ = data_group.array(
            name=key,
            data=this_data,
            shape=this_data.shape,
            chunks=this_data.shape,
            compressor=None,
            dtype=this_data.dtype,
        )

    def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False

    with tqdm(
        total=n_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0
    ) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = set()
            for key in rgb_keys:
                data_key = f"observations/{key}/{camera_names[0]}"  # XXX: assume only one camera
                shape = tuple(shape_meta["obs"][key]["shape"])
                c, h, w = shape
                this_compressor = Jpeg2k(level=50)
                img_arr = data_group.require_dataset(
                    name=key,
                    shape=(n_steps, h, w, c),
                    chunks=(1, h, w, c),
                    compressor=this_compressor,
                    dtype=np.uint8,
                )
                for i in range(num_episodes):
                    dataset_path = os.path.join(dataset_dir, f"episode_{i}.hdf5")
                    with h5py.File(dataset_path, "r") as demo:
                        hdf5_arr = demo[data_key][:]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(
                                    futures,
                                    return_when=concurrent.futures.FIRST_COMPLETED,
                                )
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError("Failed to encode image!")
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[i] + hdf5_idx
                            futures.add(
                                executor.submit(
                                    img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx
                                )
                            )
            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError("Failed to encode image!")
            pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def _smooth(data, window_size=5):
    if window_size % 2 == 0:
        print("window size must be odd, add 1 autonomously.")
        window_size += 1
    data = np.pad(data, (window_size // 2, window_size // 2), mode='edge')
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def _find_keypose_idx(
    gripper: np.ndarray,
    ee_vel: np.ndarray,
    window_size: int=5,
    gripper_epsilon=GRIPPER_EPSILON,
    vel_epsilon=EE_VEL_EPSILONE,
) -> List[int]:
    '''
    Locate keypose indices in a trajectory.

    Args:
        gripper_: array of normalized gripper openness wrt time, (T,)
            0 - totally closed, 1 - totally open
        ee_vel: array of end-effector velocity

    Returns:
        list of indices of keyposes for both arms.
    '''
    keypose_indices = list()
    T = len(gripper)
    # smooth to remove noise
    gripper = _smooth(gripper, window_size=window_size)
    ee_vel = _smooth(ee_vel, window_size=5)
    gripper_change_rate = np.diff(gripper) / DT
    curr_state = "stable"  # opening, closing, stable
    problem = False
    for i in range(T-1):
        if i == 0:
            keypose_indices.append(i)
        else:
            if curr_state == "stable":
                if gripper_change_rate[i] > gripper_epsilon:
                    curr_state = "opening"
                    keypose_indices.append(i)
                elif gripper_change_rate[i] < -gripper_epsilon:
                    curr_state = "closing"
                    keypose_indices.append(i)
            elif curr_state == "opening":
                if abs(gripper_change_rate[i]) < gripper_epsilon:
                    curr_state = "stable"
                elif gripper_change_rate[i] < -gripper_epsilon:
                    print(f"why the gripper is closing when it is opening at {i}? ")
                    problem = True
            elif curr_state == "closing":
                if abs(gripper_change_rate[i]) < gripper_epsilon:
                    curr_state = "stable"
                    keypose_indices.append(i)
                elif gripper_change_rate[i] > gripper_epsilon:
                    print(f"why the gripper is opening when it is closing at {i}?")
                    problem = True
            if keypose_indices[-1] != i:
                ## gripper state is not key, check velocity
                if ee_vel[i-1] > vel_epsilon and ee_vel[i] < vel_epsilon:
                    keypose_indices.append(i)

    keypose_indices.append(T-1)            
    return keypose_indices, problem


def _save_videos(video, dt, video_path=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')


def _plot_ee_and_gripper(
    dataset_dir: str,
    i: int,
):
    dataset_path = os.path.join(dataset_dir, f"episode_{i}.hdf5")
    with h5py.File(dataset_path, "r") as demo:
        ### load images and save videos
        this_image = dict()
        for cam_name in demo[f'/observations/images/'].keys():
            this_image[cam_name] = demo[f'/observations/images/{cam_name}'][:].astype(np.uint8)
        # _save_videos(this_image, DT, video_path=f'{dataset_dir}/episode_{i}.mp4')

        # extract qpos and gripper pos
        this_qpos_left = demo["observations/qpos"][:, :6].astype(np.float32)
        this_qpos_right = demo["observations/qpos"][:, 6+1:6+7].astype(np.float32)
        this_gripper_left = demo["observations/qpos"][:, 6].astype(np.float32)
        this_gripper_right = demo["observations/qpos"][:, 13].astype(np.float32)

        this_gripper_act_left = demo["action"][:, 6].astype(np.float32)
        this_gripper_act_right = demo["action"][:, 13].astype(np.float32)

        ### pos of x, y, z & vel, gripper, abs_vel, ee dist and ee dist rate
        num_t, num_dim = this_qpos_left.shape[0], 3 + 3 + 1 + 1 + 2
        h, w = 2, num_dim
        num_figs = num_dim

        ### cpmpute and plot ee
        this_ee_pos_left = np.zeros((num_t, 3))
        this_ee_pos_right = np.zeros((num_t, 3))
        for j in range(num_t):
            # pos
            left_pose_mat = mr.FKinSpace(vx300s.M, vx300s.Slist, this_qpos_left[j])
            right_pose_mat = mr.FKinSpace(vx300s.M, vx300s.Slist, this_qpos_right[j])
            this_ee_pos_left[j] = np.dot(LEFT_BASE_POSE, left_pose_mat)[:3, 3]
            this_ee_pos_right[j] = np.dot(RIGHT_BASE_POSE, right_pose_mat)[:3, 3]

        this_ee_dpos_left = np.diff(this_ee_pos_left, axis=0) / DT
        this_ee_dpos_right = np.diff(this_ee_pos_right, axis=0) / DT

        this_ee_vel_norm_left = np.linalg.norm(this_ee_dpos_left, axis=-1)
        this_ee_vel_norm_right = np.linalg.norm(this_ee_dpos_right, axis=-1)

        this_ee_dist = np.linalg.norm(this_ee_pos_left - this_ee_pos_right, axis=-1)
        this_ee_ddist = np.diff(this_ee_dist) / DT

        window_size = 5
        if i == 45:
            window_size = 31
        keypose_left, problem_left = _find_keypose_idx(this_gripper_left, this_ee_vel_norm_left, window_size=window_size)
        keypose_right, problem_right = _find_keypose_idx(this_gripper_right, this_ee_vel_norm_right, window_size=window_size)
        if problem_left:
            print(f'left problem in episode {i}')
        if problem_right:
            print(f'right problem in episode {i}')

        idx_ylabel_map = {
            0: r"$x$ [m]",
            1: r"$y$ [m]",
            2: r"$z$ [m]",
            3: r"$\dot{x}$ [m/s]",
            4: r"$\dot{y}$ [m/s]",
            5: r"$\dot{z}$ [m/s]",
            6: "gripper",
            7: r"$v_{\rm ee}$ [m/s]",
            8: r"$d_{\rm ee}$ [m]",
            9: r"$\dot{d}_{\rm ee} [m/s]$",
        }

        fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))
        t = np.arange(num_t) * DT
        for idx_dim in range(num_dim):
            ax = axs[idx_dim]
            if idx_dim < 3:
                ### x, y, z
                ax.plot(t, this_ee_pos_left[:, idx_dim], "r", label="left")
                ax.plot(t, this_ee_pos_right[:, idx_dim], "b", label="right")
                ax.legend()
            elif 3 <= idx_dim < 6:
                ### xdot, ydot, zdot
                ax.plot(t[:-1], this_ee_dpos_left[:, idx_dim-3], "r", label="left")
                ax.plot(t[:-1], this_ee_dpos_right[:, idx_dim-3], "b", label="right")
                ax.legend()
                # ax.set_ylim([-0.5, 0.5])
            elif idx_dim == 6:
                ### gripper
                ax.plot(t, this_gripper_left, "r", label="left")
                ax.plot(t, this_gripper_right, "b", label="right")
                left_first_diff = np.diff(_smooth(this_gripper_left)) / DT
                right_first_diff = np.diff(_smooth(this_gripper_right)) / DT
                ax.plot(t[:-1], left_first_diff, "r--")
                ax.plot(t[:-1], right_first_diff, "b--")
                ax.plot(t, this_gripper_act_left, "r:")
                ax.plot(t, this_gripper_act_right, "b:")
                ax.plot(t, np.ones_like(t) * GRIPPER_EPSILON, 'k--')
                ax.plot(t, -np.ones_like(t) * GRIPPER_EPSILON, 'k--')
                ax.scatter(t[keypose_left], this_gripper_left[keypose_left], marker='x', color='r')
                ax.scatter(t[keypose_right], this_gripper_right[keypose_right], marker='x', color='b')
                ax.legend()
            elif idx_dim == 7:
                ### ee vel
                ax.plot(t[:-1], this_ee_vel_norm_left, "r", label="left")
                ax.plot(t[:-1], this_ee_vel_norm_right, "b", label="right")
                ax.plot(t, np.ones_like(t) * EE_VEL_EPSILONE, 'k--')
                # set y limit
                ax.legend()
                ax.set_ylim([0, 0.1])
            elif idx_dim == 8:
                ### ee dist
                ax.plot(t, this_ee_dist, "r")
                ax.plot(t, np.ones_like(t) * EE_DIST_BOUND, 'k--')
            elif idx_dim == 9:
                ### ee dist rate
                ax.plot(t[:-1], this_ee_ddist, "b")
                ax.set_ylim([-1, 2])
            ax.set_xlabel("time [s]")
            ax.set_ylabel(idx_ylabel_map[idx_dim])

        plt.tight_layout()
        plt.savefig(f'{dataset_dir}/episode_{i}_ee.png', dpi=300)
        plt.close()


def iter_over_demos(
    dataset_dir: str,
    num_episodes: int = 50,
):
    '''plot vel and gripper curve for keypose finding
    '''
    dataset_dir = str(pathlib.Path(dataset_dir).expanduser())
    with tqdm(total=num_episodes, desc="Process", mininterval=1.0) as pbar:
        for i in range(num_episodes):
            _plot_ee_and_gripper(dataset_dir, i)
            pbar.update()


def main():
    task = "sim_insertion_scripted"
    dataset_dir = "~/bimanual/Diffusion-Policy/data/aloha/datasets/" + task
    shape_meta = {
        "obs": {
            "images": {
                "shape": (3, 480, 640),
                "type": "rgb",
            },
            "qpos": {
                "shape": (14,),
                "type": "low_dim",
            },
        },
        "action": {
            "shape": (14,),
            "type": "low_dim",
        },
    }
    dataset = AlohaImageDataset(
        str(pathlib.Path(dataset_dir).expanduser()),
        shape_meta,
        horizon=5,
        use_cache=False,
        task=task
    )

    # from matplotlib import pyplot as plt
    print(dataset.replay_buffer["images"].shape)
    normalizer = dataset.get_normalizer()
    nactions = normalizer['action'].normalize(dataset.replay_buffer['action'][:])
    diff = np.diff(nactions, axis=0)
    dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)


if __name__ == "__main__":
    iter_over_demos(
        dataset_dir="~/bimanual/Diffusion-Policy/data/aloha/datasets/sim_transfer_cube_scripted",
        num_episodes=5,
    )
