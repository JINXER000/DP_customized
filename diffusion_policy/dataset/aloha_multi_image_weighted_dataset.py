if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import time
import os
import pathlib
import h5py
from typing import Dict
import torch
import numpy as np
import copy
from tqdm import tqdm
from threadpoolctl import threadpool_limits

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer_with_mode import ReplayBufferWithMode
from diffusion_policy.common.sampler_with_mode import (
    SequenceSamplerWithMode,
    create_all_indices,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import get_image_range_normalizer


register_codecs()


class AlohaMultiImageWeightedDataset(BaseImageDataset):
    def __init__(
        self,
        dataset_dir: str,
        shape_meta: dict,
        num_episodes=20,
        round: int = 0,
        num_per_round: int = 10,
        camera_names=["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"],
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        n_obs_steps=None,
        max_train_trajs=None,
        weight_type="iwr",
    ):
        """
        Args: (HITL-specifc)
            @round: the i-th round of training0: initial, no HITL data; 1-3: HITL. Default: 0
            @num_per_round: number of added episodes per round. Default: 10
            @weight_type : ["iwr", "sirius", "ceiling"], different weighting schemes. Default: "iwr"
        """
        super().__init__()

        ## read .hdf5 files and save to replay buffer
        replay_buffer = self.load_data(
            dataset_dir=dataset_dir,
            num_episodes=num_episodes,
            round=round,
            round_prefix=weight_type + "_",
            num_per_round=num_per_round,
            camera_names=camera_names,
        )

        ## check obs type and save to corresponding list
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            #type = attr.get("type", "low_dim") # why "low_dim"?
            type = attr.get("type")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        # get train/val masks for indices and sampler
        all_indices = create_all_indices(
            replay_buffer.subtraj_ends,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
        )
        val_mask = get_val_mask(
            n_trajs=all_indices.shape[0],
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_trajs, seed=seed
        )

        sampler = SequenceSamplerWithMode(
            replay_buffer=replay_buffer,
            all_indices=all_indices,
            sequence_length=horizon,
            key_first_k=key_first_k,
            traj_mask=train_mask,
        )

        # compute sample weights based on mode ratio
        # mode: 0-demo, 1-auto, 2-preintervention, 3-human
        mode_weight_map = np.full((4,), np.nan, dtype=np.float32)  # weight = P* / P, P* should sum up to 1
        mode_ratio = np.array(sampler.mode_ratio, dtype=np.float32)
        if weight_type == "iwr":
            interv_ratio = mode_ratio[0] + mode_ratio[3]
            non_interv_ratio = mode_ratio[1] + mode_ratio[2]
            mode_weight_map[0] = mode_weight_map[3] = 0.5 / interv_ratio
            mode_weight_map[1] = mode_weight_map[2] = 0.5 / non_interv_ratio
        elif weight_type == "sirius":
            r0 = min(mode_ratio[0], 0.5)  # desired demo ratio
            desired_mode_ratio = np.array(
                [r0, max(0.0, 1 - r0 - 0.5), 0.0, 0.5],
                dtype=np.float32
            )
            mode_weight_map = desired_mode_ratio / mode_ratio
        elif weight_type == "ceiling":
            pass
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_first_k = key_first_k
        self.n_obs_steps = n_obs_steps
        self.all_indices = all_indices
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.camera_names = camera_names
        self.weight_type = weight_type
        self.mode_weight_map = mode_weight_map

    def load_data(
        self,
        dataset_dir,
        num_episodes,
        round,
        round_prefix,
        num_per_round,
        camera_names,
    ):
        def add_one_traj(replay_buffer: ReplayBufferWithMode, dataset_path):
            with h5py.File(dataset_path, "r") as root:
                qpos = root["/observations/qpos"][:]
                action = root["/action"][:]
                mode = root["/mode"][:] if "mode" in root else \
                    np.full((qpos.shape[0],), 0, dtype=np.int32)
                all_cam_images = dict()
                for cam_name in camera_names:
                    all_cam_images[cam_name] = root[f"/observations/images/{cam_name}"][:]

            episode = {
                "qpos": qpos, # [T, dim]
                "action": action, # [T, dim]
                "mode": mode, # [T,]
            }
            episode.update(all_cam_images)
            replay_buffer.add_episode(episode)

        replay_buffer = ReplayBufferWithMode.create_empty_numpy()

        for i in tqdm(range(num_episodes + round * num_per_round)):  # num_episodes
            if i < num_episodes:
                dataset_path = pathlib.Path(dataset_dir) / f"episode_{i}.hdf5"
            else:
                r = (i - num_episodes) // num_per_round + 1
                idx = (i - num_episodes) % num_per_round
                prefix = round_prefix if r > 1 else ""  # round 1 shares the same data
                dataset_path = pathlib.Path(dataset_dir) / f"{prefix}round_{r}/episode_{idx}.hdf5"
            add_one_traj(replay_buffer, dataset_path)

        return replay_buffer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSamplerWithMode(
            replay_buffer=self.replay_buffer,
            all_indices=self.all_indices,
            sequence_length=self.horizon,
            key_first_k=self.key_first_k,
            traj_mask=~self.train_mask
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
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
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

        # turn mode into weight
        weight = self.mode_weight_map[data["mode"]]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
            "weight": torch.as_tensor(weight, dtype=torch.float32),
        }
        return torch_data


def main():
    dataset_root = pathlib.Path("/ssd1/yudongjie/data/aloha/datasets/")
    task = "cup_random"
    dataset_dir = dataset_root / task
    shape_meta = {
        "obs": {
            "cam_high": {
                "shape": (3, 120, 160),
                "type": "rgb",
            },
            "cam_low": {
                "shape": (3, 120, 160),
                "type": "rgb",
            },
            "cam_left_wrist": {
                "shape": (3, 120, 160),
                "type": "rgb",
            },
            "cam_right_wrist": {
                "shape": (3, 120, 160),
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

    dataset = AlohaMultiImageWeightedDataset(
        str(dataset_dir.expanduser()),
        shape_meta,
        num_episodes=20,
        round=1,
        num_per_round=10,
        camera_names=["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"],
        horizon=30,
        pad_before=1,
        pad_after=24,
        val_ratio=0.1,
        weight_type="iwr",
    )
    normalizer = dataset.get_normalizer()
    batch = dataset[0]
 
    nobs = normalizer.normalize(batch["obs"])
    weight = batch["weight"]
    naction = normalizer["action"].normalize(batch["action"])

    # print sample shape
    for k, v in nobs.items():
        print(f"obs-{k} shape: {v.shape}")
    print(f"action shape: {naction.shape}")
    print(f"weight shape: {weight.shape}")
    print(f"weight: {weight}")

    val_dataset = dataset.get_validation_dataset()
    print(f"train set len: {len(dataset)}")
    print(f"  val set len: {len(val_dataset)}")

    print(dataset.sampler.mode_ratio)
    print(val_dataset.sampler.mode_ratio)

    # print("---- speed test begins ----")
    # train_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=64,
    #     num_workers=8,
    #     prefetch_factor=2,
    #     shuffle=True,
    #     pin_memory=True,
    #     persistent_workers=True,
    # )

    # np.set_printoptions(precision=3)
    # num_epochs = 1
    # num_steps = 10
    # for epoch in range(num_epochs):
    #     print(f"Epoch {epoch}:")
    #     train_time_per_batch = []
    #     start = time.time()
    #     for i, batch in enumerate(tqdm(train_loader)):
    #         time_get = time.time()
    #         train_time_per_batch.append(time_get - start)
    #         start = time_get
    #         if i + 1 == num_steps:
    #             break
    #     train = np.array(train_time_per_batch)
    #     print(f"Train mean: {train.mean():.3f}, std: {train.std():.3f}, max: {train.max():.3f}")
    #     print("train:", train[:10])

if __name__ == "__main__":
    main()
