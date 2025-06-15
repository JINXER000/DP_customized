import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from scripts.robomimic_dmg_wrapper import DMG_env_switchable

def create_env(env_meta, shape_meta, enable_render=True, use_onscreen_renderer = True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=use_onscreen_renderer, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    # env = DMG_env_switchable(
    #     env_name = env_meta['env_name'],
    #     cam_names = ["agentview",  "robot0_eye_in_hand", "robot1_eye_in_hand"]
    # )
    return env


class RobomimicImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        # Create single environment
        robomimic_env = create_env(
            env_meta=env_meta, 
            shape_meta=shape_meta
        )
        # Disable hard reset to reduce memory consumption
        robomimic_env.env.hard_reset = False
        
        self.env = MultiStepWrapper(
            RobomimicImageWrapper(
                env=robomimic_env,
                shape_meta=shape_meta,
                init_state=None,
                render_obs_key=render_obs_key
            ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps
        )

        # Store initialization parameters
        self.env_meta = env_meta
        self.dataset_path = dataset_path
        self.shape_meta = shape_meta
        self.n_train = n_train
        self.n_train_vis = n_train_vis
        self.train_start_idx = train_start_idx
        self.n_test = n_test
        self.n_test_vis = n_test_vis
        self.test_start_seed = test_start_seed
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.render_obs_key = render_obs_key
        self.fps = fps
        self.crf = crf
        self.past_action = past_action
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.output_dir = output_dir

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # Initialize lists to store results
        all_rewards = []
        all_seeds = []
        all_prefixs = []

        # # Run training episodes
        # with h5py.File(self.dataset_path, 'r') as f:
        #     for i in range(self.n_train):
        #         train_idx = self.train_start_idx + i
        #         enable_render = i < self.n_train_vis
        #         init_state = f[f'data/demo_{train_idx}/states'][0]

        #         # Setup environment
        #         env.env.init_state = init_state
        #         if enable_render:
        #             filename = pathlib.Path(self.output_dir).joinpath(
        #                 'media', wv.util.generate_id() + ".mp4")
        #             filename.parent.mkdir(parents=False, exist_ok=True)
        #             filename = str(filename)
        #             # Note: video recording is disabled in this version

        #         # Run episode
        #         obs = env.reset()
        #         past_action = None
        #         policy.reset()
        #         rewards = []

        #         env_name = self.env_meta['env_name']
        #         pbar = tqdm.tqdm(total=self.max_steps, 
        #             desc=f"Eval {env_name} Train {i+1}/{self.n_train}", 
        #             leave=False, 
        #             mininterval=self.tqdm_interval_sec)
                
        #         done = False
        #         while not done:
        #             # create obs dict and reshape each item to add batch dimension
        #             np_obs_dict = dict()
        #             for key, value in obs.items():
        #                 # Add batch dimension of size 1
        #                 np_obs_dict[key] = np.expand_dims(value, axis=0)
                    
        #             if self.past_action and (past_action is not None):
        #                 np_obs_dict['past_action'] = past_action[
        #                     :,-(self.n_obs_steps-1):].astype(np.float32)
                    
        #             # device transfer
        #             obs_dict = dict_apply(np_obs_dict, 
        #                 lambda x: torch.from_numpy(x).to(device=device))

        #             # run policy
        #             with torch.no_grad():
        #                 action_dict = policy.predict_action(obs_dict)

        #             # device_transfer
        #             np_action_dict = dict_apply(action_dict,
        #                 lambda x: x.detach().to('cpu').numpy())

        #             action = np_action_dict['action']
        #             if not np.all(np.isfinite(action)):
        #                 print(action)
        #                 raise RuntimeError("Nan or Inf action")
                    
        #             # step env
        #             env_action = action
        #             if self.abs_action:
        #                 env_action = self.undo_transform_action(action)

        #             obs, reward, done, info = env.step(env_action)
        #             rewards.append(reward)
        #             past_action = action

        #             # update pbar
        #             pbar.update(action.shape[1])
        #         pbar.close()

        #         all_rewards.append(np.array(rewards))
        #         all_seeds.append(train_idx)
        #         all_prefixs.append('train/')

        # Run test episodes
        for i in range(self.n_test):
            seed = self.test_start_seed + i
            enable_render = i < self.n_test_vis

            # Setup environment
            env.env.init_state = None
            env.seed(seed)
            if enable_render:
                filename = pathlib.Path(self.output_dir).joinpath(
                    'media', wv.util.generate_id() + ".mp4")
                filename.parent.mkdir(parents=False, exist_ok=True)
                filename = str(filename)
                # Note: video recording is disabled in this version

            # Run episode
            obs = env.reset()
            past_action = None
            policy.reset()
            rewards = []

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, 
                desc=f"Eval {env_name} Test {i+1}/{self.n_test}", 
                leave=False, 
                mininterval=self.tqdm_interval_sec)
            
            done = False
            while not done:
                # create obs dict and reshape each item to add batch dimension
                np_obs_dict = dict()
                for key, value in obs.items():
                    # Add batch dimension of size 1
                    np_obs_dict[key] = np.expand_dims(value, axis=0)
                
                if self.past_action and (past_action is not None):
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'][0] ## NOTE: (1, 8, 14) -> (8, 14)
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                ## if abs action, we need to transform from rot6d to axis-angle
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                rewards.append(reward)
                past_action = action

                self.env.env.env.render()

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            all_rewards.append(np.array(rewards))
            all_seeds.append(seed)
            all_prefixs.append('test/')

        # Log results
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        
        for i in range(len(all_rewards)):
            seed = all_seeds[i]
            prefix = all_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
