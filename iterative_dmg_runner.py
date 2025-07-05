
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


from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

from scripts.robomimic_dmg_wrapper import DMG_env_switchable,to_camel_case
import cv2

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




class Robosuite_Evaluator():
    def __init__(self, checkpoint_dict, output, max_timesteps, \
                 num_inference_steps, with_planning= False, scale = 1.0):
        self.max_timesteps = max_timesteps
        self.checkpoint_dict = checkpoint_dict
        self.output = output
        self.num_inference_steps = num_inference_steps

        self.with_planning = with_planning  
        self.image_list = []


    def initialize_env(self, env_name, reset_grippers= True, **kwargs):
        self.cur_env_name = env_name
        self.load_checkpoint(**kwargs)        
        self.ts = self.reset_all(reset_grippers = reset_grippers)

    def load_checkpoint(self, width = 84, height = 84, controller_name = "OSC_POSE", **kwargs):
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

        ## multi-step params for policy
        self.query_cycle = cfg.n_action_steps
        self.n_obs_steps = cfg.n_obs_steps

        ## setup environment
        env_name = to_camel_case(self.cur_env_name)
        self.env = DMG_env_switchable(env_name, controller_name = controller_name, abs_action = False, H = height, W= width, cam_names = ["agentview", "birdview", "frontview", "robot0_eye_in_hand", "robot1_eye_in_hand"],)
        

    def reset_all(self, reset_grippers = True):
        ts = self.env.reset_ts(with_planning=self.with_planning)

            
        ## obs history for extracting multi-step obs
        self.obs_history = dict()
        for key in self.obs_shape_meta.keys():
            self.obs_history[key] = np.zeros(
                (self.max_timesteps, *self.obs_shape_meta[key].shape),
                dtype=np.float32
            )
        self.t = 0
        return ts

    def replay_tamp_step(self, total_action):
        self.ts = self.env.replay_tamp_step(total_action)
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
            if self.t % self.query_cycle == 0:
                action_dict = self.policy.predict_action(obs_dict)
                self.np_action_seq = action_dict['action'][0].detach().to('cpu').numpy() # T,Da
            action = self.np_action_seq[self.t % self.query_cycle]
            # t1 = time.perf_counter()

            self.ts = self.env.step_ts(action)

            self.t += 1

        if render:
            self.env.env.render()

        return self.ts.done

    def append_image(self):
        cam_high_image = self.env.image_recorder.cam_high_image
        # import cv2
        # cam_high_image = cv2.resize(cam_high_image, (240, 320))
        self.image_list.append({'cam_high':cam_high_image})

    def exit(self, save_dir):
        if len(self.image_list) == 0:
            print("No images to save.")
            return
        # save_videos(self.image_list, DT, video_path=os.path.join(save_dir, f'rollout.mp4'))
        import cv2
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cur_time = time.strftime("%d_%H.%M.%S", time.localtime())
        vid_save_path = os.path.join(save_dir,  'test_' +cur_time +'.avi')

        height, width, _ = self.image_list[0]['cam_high'].shape
        fps = 30  # Adjust based on your camera settings
        self.video_writer = cv2.VideoWriter(
            vid_save_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            fps,
            ( width,height)
        )
        for image in self.image_list:
            rgb_image = cv2.cvtColor(image['cam_high'], cv2.COLOR_BGR2RGB)
            # transpose image width and height
            self.video_writer.write(rgb_image)
        self.video_writer.release()
        print(f"Saved video to {vid_save_path}")



def wrapper_test():
    output = './data/eval/transfer_cup/'
    checkpoint_dict = {\
        # 'two_arm_three_piece_assembly': 'data/outputs/two_arm_assembly/latest.ckpt',
        'two_arm_threading': 'data/outputs/two_arm_threading/latest.ckpt',
                       }
    env_names = list(checkpoint_dict.keys())

    max_timesteps = 500
    num_inference_steps = 10

    env_runer = Robosuite_Evaluator(checkpoint_dict, output, max_timesteps, num_inference_steps)
    
    for skill in env_names:
        env_runer.initialize_env(skill, width = 168, height = 168)
        for i in range(max_timesteps):
            done = env_runer.inference_once()
            task_success = env_runer.env.handle_rewards()
            if task_success:
                print('Task completed!')
                break
            # dp.append_image()

    env_runer.exit(output)


if __name__ == '__main__':
    # main()
    wrapper_test()
