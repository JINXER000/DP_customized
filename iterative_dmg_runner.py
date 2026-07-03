import os
import pathlib
import time
import numpy as np
import copy
import json
import numpy as np
import torch
import dill
import hydra
import h5py
from collections import namedtuple


from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from scripts.robomimic_dmg_wrapper import DMG_env_switchable, to_camel_case
import cv2

# Timestep contract for reset_ts/step_ts. The upstream DMG wrapper turned these
# into abstract stubs, so the subclass here owns the return type.
ts_tuple = namedtuple("ts_tuple", ["observation", "reward", "done", "info"])

def collect_obs(obs_shape_meta, obs_history, t, obs):
    """
    Collect observations from the environment and store them in obs_history.
    Handles both RGB and low-dim observations according to shape_meta.
    RGB images are expected to be in channels-first format (C,H,W).
    """
    for k, v in obs_shape_meta.items():
        tgt_shape = v['shape']
        if len(tgt_shape)==3 and v["type"] == "rgb" and obs[k].shape!= tgt_shape:
            ## resize image
            img_hwc = np.transpose(obs[k].astype(np.float32), (1, 2, 0))
            img_resized = cv2.resize(img_hwc, (tgt_shape[1], tgt_shape[2]), interpolation=cv2.INTER_LINEAR)
            cur_obs = np.transpose(img_resized, (2, 0, 1))  # Convert to channels-first format
        else:
            cur_obs = obs[k].astype(np.float32)
        obs_history[k][t] = cur_obs
    return


def get_seq_obs(obs_history, t, n_obs_steps):
    """
    Get a sequence of observations for the policy.
    If we don't have enough history, pad with the first observation.
    """
    obs_dict_np = dict()
    if t < n_obs_steps - 1:
        # Pad with first observation if we don't have enough history
        for k, v in obs_history.items():
            obs_dict_np[k] = np.array([v[0]] * n_obs_steps, dtype=np.float32)
            obs_dict_np[k][n_obs_steps-t-1:n_obs_steps] = v[0:t+1]
    else:
        # Get the last n_obs_steps observations
        for k, v in obs_history.items():
            obs_dict_np[k] = v[t-n_obs_steps+1:t+1]
    return obs_dict_np




class Robosuite_Evaluator(DMG_env_switchable):
    def __init__(self, checkpoint_dict, output, max_timesteps,\
                 num_inference_steps,fps = 10, crf = 22, record = False, render_obs_key = ["robot0_eye_in_hand_image", "robot1_eye_in_hand_image"], **kwargs):
        self.max_timesteps = max_timesteps
        self.checkpoint_dict = checkpoint_dict
        self.output = output
        self.num_inference_steps = num_inference_steps
        self.fps = fps
        self.render_obs_key = render_obs_key
        self.hdf5_path = None
        self.hdf5_buffers = {}


        self.lfd_alg = 'DP'
        ## config image recording
        if record:
            self.video_recorder = VideoRecorder.create_h264(
                            fps=fps,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=crf,
                            thread_type='FRAME',
                            thread_count=1
                        )
            save_dir = output
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cur_time = time.strftime("%d_%H.%M.%S", time.localtime())
            self.file_path = os.path.join(save_dir,  'test_' +cur_time +'.mp4')
            self.hdf5_path = os.path.join(save_dir, 'test_' +cur_time +'.hdf5')
            print(f"{self.render_obs_key} will be saved to: {self.file_path}")
        else:
            self.file_path = None
            self.video_recorder = None

        self.env_initialized = False

    def _policy_step_offset(self):
        return (self.t - self.t_bc_start) % self.query_cycle

    def set_bc_controller(self):
        self.update_controllers(controller_name = "OSC_POSE", abs_action = self.lfd_abs_action)
        self.t_bc_start = self.t


    def initialize_env(self, env_name,  **kwargs):
        self.cur_env_name = env_name
        self.load_checkpoint(**kwargs)        
        self.ts = self.reset_all()

    def load_checkpoint(self, width = 84, height = 84, controller_name = "OSC_POSE", env_options=None, **kwargs):
        # load checkpoint
        payload = torch.load(open(self.checkpoint_dict[self.cur_env_name], 'rb'), pickle_module=dill)
        cfg = payload['cfg']

        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=self.output)
        workspace: BaseWorkspace
        # in case that model, ema_model & opt are not defined in __init__ (e.g. ddp)
        if "model" not in workspace.__dict__.keys():
            workspace.model = hydra.utils.instantiate(cfg.policy)
        if "ema_model" not in workspace.__dict__.keys() and cfg.training.use_ema:
            workspace.ema_model = copy.deepcopy(workspace.model)
        if "optimizer" not in workspace.__dict__.keys():
            workspace.optimizer = hydra.utils.instantiate(
                cfg.optimizer, workspace.model.parameters()
            )
        workspace.load_payload(payload, exclude_keys=["optimizer"], include_keys=None)

        # get policy from workspace
        if 'diffusion' in cfg.name:
            ## diffusion model
            policy: BaseImagePolicy
            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model

            self.device = torch.device('cuda')
            policy.eval().to(self.device)

            ## set inference params
            policy.num_inference_steps = self.num_inference_steps #16 # [DDIM inference iterations]
        else:
            raise RuntimeError("Unsupported policy type: ", cfg.name)
        
        self.policy= policy
        # hyper-parameters
        ## observation
        self.obs_shape_meta = cfg.task.shape_meta.obs

        ## setup action type
        self.action_dim = cfg.task.shape_meta.action.shape
        if self.action_dim[0] > 14:
            self.lfd_abs_action = True
            self.rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')
        else:
            self.lfd_abs_action = False
            self.rotation_transformer = None

        ## multi-step params for policy
        self.query_cycle = cfg.n_action_steps
        self.n_obs_steps = cfg.n_obs_steps

        ## setup environment
        env_name = to_camel_case(self.cur_env_name)
        ## TODO： switch between different skills.
        if self.env_initialized:
            raise NotImplementedError("policy switching is not supported yet")
        
        np.random.seed(int(time.time()))
        init_kwargs = dict(
            controller_name=controller_name,
            abs_action=self.lfd_abs_action,
            max_timesteps=self.max_timesteps,
            **kwargs,
        )
        if env_options is None:
            # DMG_env_switchable no longer accepts H/W/cam_names directly; it expects a
            # robosuite-style env_options dict. Reproduce the wrapper's previous defaults.
            env_options = dict(
                env_configuration="single-arm-parallel",
                robots=["Panda", "Panda"],
                camera_names=["agentview", "birdview", "frontview", "robot0_eye_in_hand", "robot1_eye_in_hand"],
                camera_heights=height,
                camera_widths=width,
                camera_segmentations="instance",
            )
        init_kwargs["env_options"] = env_options

        super().__init__(env_name, **init_kwargs)
        self.env_initialized = True

    def reset_to(self, state):
        ret = super().reset_to({"states" : state})
        self.t_bc_start = self.t
        return ret

    def reset_ts(self):

        self.raw_obs = self.env.reset()
        self.obs = self.get_observation(self.raw_obs)
        init_ts = ts_tuple(self.obs, 0, False, {})
        return init_ts
    
    def step_ts(self, action):
        self.raw_obs, reward, done, info = self.env.step(action)
        self.obs = self.get_observation(self.raw_obs)
        info["is_success"] = self.is_success()
        return ts_tuple(self.obs, reward, done, info)

    def reset_all(self):
        ts = self.reset_ts()


        ## obs history for extracting multi-step obs
        self.obs_history = dict()
        for key in self.obs_shape_meta.keys():
            self.obs_history[key] = np.zeros(
                (self.max_timesteps, *self.obs_shape_meta[key].shape),
                dtype=np.float32
            )

        if self.video_recorder is not None:
            render_keys = [self.render_obs_key] if isinstance(self.render_obs_key, str) else list(self.render_obs_key)
            for cam in render_keys:
                if cam not in self.obs_shape_meta:
                    raise KeyError(f"render_obs_key '{cam}' not found in obs_shape_meta")
                if self.obs_shape_meta[cam].get('type') != 'rgb':
                    raise ValueError(f"render_obs_key '{cam}' must have type 'rgb', got {self.obs_shape_meta[cam].get('type')!r}")
            self.hdf5_buffers = {cam: [] for cam in render_keys}
        else:
            self.hdf5_buffers = {}

        self.t = 0
        self.t_bc_start = 0
        return ts

    def record_frame(self, obs):

        if self.video_recorder is not None:
            try:
                if not self.video_recorder.is_ready():
                    self.video_recorder.start(self.file_path)

                render_keys = [self.render_obs_key] if isinstance(self.render_obs_key, str) else list(self.render_obs_key)
                cam_frames = []
                for cam_name in render_keys:
                    cam_frame = (np.moveaxis(obs[cam_name], 0, -1) * 255).astype(np.uint8)
                    if cam_name in self.hdf5_buffers:
                        self.hdf5_buffers[cam_name].append(cam_frame)
                    cam_frames.append(cam_frame)
                frame = cam_frames[0] if len(cam_frames) == 1 else np.concatenate(cam_frames, axis=1)

                self.video_recorder.write_frame(frame)
            except Exception as e:
                print(f"Warning: Failed to record video frame: {e}")
                import traceback
                traceback.print_exc()

    def replay_tamp_step(self, total_action):
        # self.ts = self.replay_tamp_step(total_action)

        start = time.time()

        obs = self.ts.observation
        collect_obs(self.obs_shape_meta, self.obs_history, self.t, obs)
        self.ts = self.step_ts(total_action)
        self.env.render()
        # limit frame rate if necessary
        elapsed = time.time() - start
        diff = 1 / self.max_framerate - elapsed
        if diff > 0:
            time.sleep(diff)
            
        self.record_frame(obs)
        self.t += 1
        return self.ts
    # def get_mj_pc_dict(self, **kwargs):
    #     return self.env.save_mj_observation(**kwargs)

    def inference_once(self, render = True):
        if self.t >= self.max_timesteps:
            return True
        with torch.inference_mode():
            # process previous ts
            obs = self.ts.observation
            collect_obs(self.obs_shape_meta, self.obs_history, self.t, obs)
            obs_dict_np = get_seq_obs(self.obs_history, self.t, self.n_obs_steps)
            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))

            # query policy to extract action: (B=1, Da)
            # t0 = time.perf_counter()
            if self._policy_step_offset() == 0:
                action_dict = self.policy.predict_action(obs_dict)
                self.np_action_seq = action_dict['action'][0].detach().to('cpu').numpy() # T,Da
                self.query_cycle = len(self.np_action_seq)  # sync to actual chunk size (may differ from cfg.n_action_steps)
            action = self.np_action_seq[self._policy_step_offset()]
            # t1 = time.perf_counter()

            self.ts = self.step_ts(action)

            self.t += 1

        if render:
            self.env.render()

        self.record_frame(obs)

        return self.ts.done

    @staticmethod
    def _outcome_prefixed_path(path, prefix):
        """Replace the provisional 'test_' token of a recording filename with `prefix`."""
        directory, filename = os.path.split(path)
        assert filename.startswith('test_'), f"unexpected recording name: {filename}"
        return os.path.join(directory, prefix + filename[len('test_'):])

    def _label_recordings_by_outcome(self):
        """Rename this run's recording artifacts to reflect the task outcome.

        The recorder writes to a provisional ``test_<stamp>`` name because success is
        only known once execution ends. Resolve the outcome with the same reward check
        the executor reports (``handle_rewards``) and rename the mp4 -- and its paired
        hdf5 -- to ``success_<stamp>`` / ``fail_<stamp>``. Best-effort: a missing or
        already-present target is reported and skipped so a valid run is never lost to
        a cosmetic rename.
        """
        if self.file_path is None:
            return
        prefix = 'success_' if self.handle_rewards() else 'fail_'
        renames = [('file_path', self.file_path)]
        if self.hdf5_path is not None and self.hdf5_buffers:
            renames.append(('hdf5_path', self.hdf5_path))
        for attr, src in renames:
            dst = self._outcome_prefixed_path(src, prefix)
            if not os.path.exists(src):
                print(f"Warning: recording artifact missing, cannot label: {src}")
                continue
            if os.path.exists(dst):
                print(f"Warning: refusing to overwrite existing artifact: {dst}")
                continue
            os.rename(src, dst)
            setattr(self, attr, dst)
            print(f"Recording labeled '{prefix.rstrip('_')}': {dst}")

    def exit(self):
     
        if self.video_recorder is not None:
            try:
                self.video_recorder.stop()
                print(f"Video saved to: {self.file_path}")
                
                # Verify the video file was created and is valid
                if os.path.exists(self.file_path):
                    file_size = os.path.getsize(self.file_path)
                    if file_size > 0:
                        print(f"Video file created successfully. Size: {file_size} bytes")
                        
                        # Check if the video needs to be converted to proper MP4 format
                        import subprocess
                        try:
                            # Use ffprobe to check the container format
                            result = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', self.file_path], 
                                                  capture_output=True, text=True)
                            if result.returncode == 0:
                                # Video is valid, try to ensure it's properly formatted
                                temp_path = self.file_path.replace('.mp4', '_temp.mp4')
                                subprocess.run(['ffmpeg', '-i', self.file_path, '-c', 'copy', '-f', 'mp4', temp_path], 
                                             capture_output=True)
                                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                                    os.replace(temp_path, self.file_path)
                                    print("Video converted to proper MP4 format")
                        except Exception as e:
                            print(f"Warning: Could not post-process video: {e}")
                    else:
                        print("Warning: Video file is empty!")
                else:
                    print("Warning: Video file was not created!")
                    
            except Exception as e:
                print(f"Error stopping video recorder: {e}")

        if self.hdf5_path is not None and self.hdf5_buffers:
            try:
                with h5py.File(self.hdf5_path, 'w') as hf:
                    hf.attrs['fps'] = self.fps
                    hf.attrs['max_timesteps'] = self.max_timesteps
                    num_recorded = 0
                    for cam_name, cam_frames in self.hdf5_buffers.items():
                        if not cam_frames:
                            continue
                        hf.create_dataset(cam_name, data=np.stack(cam_frames, axis=0), compression='gzip')
                        num_recorded = max(num_recorded, len(cam_frames))
                    hf.attrs['num_frames'] = num_recorded
                print(f"HDF5 saved to: {self.hdf5_path}")
            except Exception as e:
                print(f"Warning: Failed to write HDF5 file: {e}")

        self._label_recordings_by_outcome()
        return self.file_path



def wrapper_test():
    output = './data/eval/transfer_cup/'
    checkpoint_dict = {\
        # 'two_arm_three_piece_assembly': \
        #     #   'data/outputs/two_arm_assembly/latest.ckpt',
        #     'data/outputs/two_arm_assembly/epoch=2200-test_mean_score=0.720.ckpt',
        'two_arm_threading': \
        'data/outputs/two_arm_threading/1000demo0.500.ckpt',
            # 'data/outputs/two_arm_threading/latest.ckpt',
                       }
    env_names = list(checkpoint_dict.keys())

    max_timesteps = 500
    num_inference_steps = 10

    env_runer = Robosuite_Evaluator(checkpoint_dict, output, max_timesteps, num_inference_steps, record= True)
    
    for skill in env_names:
        env_runer.initialize_env(skill, width = 168, height = 168)
        for i in range(max_timesteps):
            done = env_runer.inference_once()
            task_success = env_runer.handle_rewards()
            if task_success:
                print('Task completed!')
                break
            # dp.append_image()

    env_runer.exit()


if __name__ == '__main__':
    # main()
    wrapper_test()
