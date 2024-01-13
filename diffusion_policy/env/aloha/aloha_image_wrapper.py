import numpy as np
import gym
from gym import spaces

from dm_control.rl.control import Environment

class AlohaImageWrapper(gym.Env):
    metadata = {
        "render.modes": ["rgb_array"], 
        "video.frames_per_second": 10
    }

    def __init__(
        self,
        env: Environment,
        shape_meta: dict,
        render_obs_key="angle",
    ):
        self.env = env
        self.render_obs_key = render_obs_key
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False

        # setup spaces
        action_shape = shape_meta["action"]["shape"]
        action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=action_shape,
            dtype=np.float32,
        )
        self.action_space = action_space

        observation_space = spaces.Dict({
            "qpos": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32),
            "images": spaces.Box(0.0, 1.0, shape=(3, 480, 640), dtype=np.float32),
        })
        self.observation_space = observation_space


    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env._task.get_observation(self.env._physics)

        self.render_cache = raw_obs["images"][self.render_obs_key]

        # raw obs is a dict, we need to extract key-val like shape_meta
        obs = dict()
        for key in self.shape_meta["obs"].keys():
            if key.endswith("images"):
                # h, w, c --> c, h, w
                # [0, 255] --> [0, 1]
                obs[key] = np.moveaxis(
                    raw_obs["images"][self.render_obs_key].astype(np.float32) / 255.0, -1, 0
                )
            else:
                obs[key] = raw_obs[key]
        return obs

    def seed(self, seed=None):
        assert isinstance(self.env._task._random, np.random.RandomState)
        self.env._task._random = np.random.RandomState(seed)
        self._seed = seed

    def reset(self):
        ts = self.env.reset()
        return self.get_observation(ts.observation)

    def step(self, action):
        ts = self.env.step(action)

        raw_obs = ts.observation
        reward = ts.reward
        done = ts.last()
        info = dict()

        obs = self.get_observation(raw_obs)
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        
        img = self.render_cache
        c, h, w = self.shape_meta['obs']['images']['shape']
        assert img.dtype == np.uint8, f"img.dtype: {img.dtype}"
        assert img.shape == (h, w, c), f"img.shape: {img.shape}"
        return img
