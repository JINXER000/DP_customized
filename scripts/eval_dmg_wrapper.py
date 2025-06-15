from typing import List, Optional
import numpy as np
import gym
from gym import spaces
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import collections

class DMGEnvWrapper(gym.Env):
    def __init__(self, 
        env_name: str,
        controller_name: str = "OSC_POSE",
        abs_action: bool = False,
        H: int = 84,
        W: int = 84,
        cam_names: List[str] = ["agentview", "birdview", "frontview", "robot0_eye_in_hand", "robot1_eye_in_hand"],
        shape_meta: Optional[dict] = None,
        init_state: Optional[np.ndarray] = None,
        render_obs_key: str = 'agentview_image',
        with_planning: bool = False
    ):
        # Create robosuite environment
        env_kwargs = {
            "env_name": env_name,
            "robots": ["Panda", "Panda"],
            "controller_configs": {
                "type": controller_name,
                "control_delta": not abs_action,
                "damping": 1.0,
                "kp": 150.0,
                "impedance_mode": "fixed",
                "position_limits": None,
                "orientation_limits": None,
                "uncouple_pos_ori": True,
                "interpolation": None,
                "ramp_ratio": 0.2,
            },
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "camera_names": cam_names,
            "camera_heights": H,
            "camera_widths": W,
            "use_object_obs": False,
        }
        
        self.env = EnvUtils.create_env_from_metadata(
            env_meta={"env_name": env_name, "env_kwargs": env_kwargs},
            render=False,
            render_offscreen=True,
            use_image_obs=True
        )
        
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False
        self.with_planning = with_planning
        
        # Setup spaces
        if shape_meta is not None:
            action_shape = shape_meta['action']['shape']
            action_space = spaces.Box(
                low=-1,
                high=1,
                shape=action_shape,
                dtype=np.float32
            )
            self.action_space = action_space

            observation_space = spaces.Dict()
            for key, value in shape_meta['obs'].items():
                shape = value['shape']
                min_value, max_value = -1, 1
                if key.endswith('image'):
                    min_value, max_value = 0, 1
                elif key.endswith('quat'):
                    min_value, max_value = -1, 1
                elif key.endswith('qpos'):
                    min_value, max_value = -1, 1
                elif key.endswith('pos'):
                    min_value, max_value = -1, 1
                else:
                    raise RuntimeError(f"Unsupported type {key}")
                
                this_space = spaces.Box(
                    low=min_value,
                    high=max_value,
                    shape=shape,
                    dtype=np.float32
                )
                observation_space[key] = this_space
            self.observation_space = observation_space
        else:
            # Default spaces if shape_meta not provided
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()
        
        if self.render_obs_key in raw_obs:
            self.render_cache = raw_obs[self.render_obs_key]

        if self.shape_meta is not None:
            obs = dict()
            for key in self.observation_space.keys():
                obs[key] = raw_obs[key]
            return obs
        else:
            return raw_obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self, with_planning=None):
        if with_planning is not None:
            self.with_planning = with_planning
            
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state
            raw_obs = self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                raw_obs = self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()

        # return obs
        obs = self.get_observation(raw_obs)
        return obs
    
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        img = np.moveaxis(self.render_cache, 0, -1)
        img = (img * 255).astype(np.uint8)
        return img

# For backward compatibility
class DMG_env_switchable(DMGEnvWrapper):
    pass 