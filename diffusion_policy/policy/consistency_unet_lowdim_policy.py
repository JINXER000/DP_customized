from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
# from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

## new package
from diffusion_policy.model.consistency.karras_diffusion import KarrasDenoiser, karras_sample
from diffusion_policy.model.consistency.sampler import create_named_schedule_sampler, LossAwareSampler
from diffusion_policy.model.consistency.scripts_util import create_ema_and_scales_fn
from diffusion_policy.model.consistency.nn import update_ema
from diffusion_policy.model.consistency.fp16_utils import master_params_to_model_params, make_master_params, get_param_groups_and_shapes
import time

import functools
import ipdb
import copy

class ConsistencyUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler,
            ema_scale,
            sample,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond

        ''' configure ema '''
        self.ema_scale_fn = create_ema_and_scales_fn(
            target_ema_mode=ema_scale.target_ema_mode,
            start_ema=ema_scale.start_ema,
            scale_mode=ema_scale.scale_mode,
            start_scales=ema_scale.start_scales,
            end_scales=ema_scale.end_scales,
            total_steps=ema_scale.total_training_steps,
            distill_steps_per_iter=ema_scale.distill_steps_per_iter,
        )

        '''-- model follow DP --'''
        self.model = model
        self.param_groups_and_shapes = get_param_groups_and_shapes(
            self.model.named_parameters()
        )
        self.master_params = make_master_params(
            self.param_groups_and_shapes
        )

        '''-- target model --'''
        self.target_model = copy.deepcopy(self.model)
        self.target_model.requires_grad_(False)
        self.target_model.train()

        self.target_model_param_groups_and_shapes = get_param_groups_and_shapes(
            self.target_model.named_parameters()
        )
        self.target_model_master_params = make_master_params(
            self.target_model_param_groups_and_shapes
        )

        '''-- scheduler follow CM --'''
        self.diffusion = KarrasDenoiser(
            sigma_data=noise_scheduler.sigma_data,
            sigma_max=noise_scheduler.sigma_max,
            sigma_min=noise_scheduler.sigma_min,
            distillation=noise_scheduler.distillation,
            weight_schedule=noise_scheduler.weight_schedule,
        )
        self.schedule_sampler = create_named_schedule_sampler(noise_scheduler.schedule_sampler, self.diffusion)

        # self.mask_generator = LowdimMaskGenerator(
        #     action_dim=action_dim,
        #     obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
        #     max_n_obs_steps=n_obs_steps,
        #     fix_obs_steps=True,
        #     action_visible=False
        # )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        self.training_mode = ema_scale.training_mode ## new
        #
        self.sampler = sample.sampler
        self.generator = sample.generator
        self.ts = sample.ts
        self.clip_denoised = sample.clip_denoised
        self.sigma_min = noise_scheduler.sigma_min
        self.sigma_max = noise_scheduler.sigma_max

        self.s_churn = sample.s_churn
        self.s_tmin = sample.s_tmin
        self.s_tmax = float(sample.s_tmax)
        self.s_noise = sample.s_noise
        self.steps = sample.steps

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps



    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        ## generate action_sequence
        # tic = time.time()
        nsample = karras_sample(
            self.diffusion,
            self.model,
            (B, T, Da),
            cond_data,
            cond_mask,
            steps=self.steps,
            clip_denoised=self.clip_denoised,
            local_cond=local_cond,
            global_cond=global_cond,
            device=self.device,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            sampler=self.sampler,
            s_churn=self.s_churn,
            s_tmin=self.s_tmin,
            s_tmax=self.s_tmax,
            s_noise=self.s_noise,
            generator=None,
            ts=self.ts,
        ) ## 重点！！ CM 定制！
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    ''' --- Function: Compute Loss --- '''
    def compute_loss(self, batch, global_step):
        # normalize input
        assert 'valid_mask' not in batch

        ## -- 提取 batch 数据： obs 和 action -- ##
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs'] #[B, horizon, obs_dim]
        action = nbatch['action'] #[B, horizon, action_dim]

        ## -- 处理 obs 数据 -- ##
        local_cond = None
        global_cond = None
        trajectory = action ## 注意：trajectory 中仅包含 action 信息
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1) # [B, n_obs_steps * obs_dim]
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        ## == 这一部分适用于 diffusion model，并未被 consistency model 提及 == ##
        # # generate impainting mask
        # if self.pred_action_steps_only:
        #     condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        # else:
        #     condition_mask = self.mask_generator(trajectory.shape)
        # loss_mask = ~condition_mask

        '''---- compute loss ----'''

        ## -- Importance-sample timesteps for a batch
        t, weights = self.schedule_sampler.sample(trajectory.shape[0], self.device)

        ema, num_scales = self.ema_scale_fn(global_step) ## 注意：参数设置 is different between CD and CT

        ## -- 声明 compute loss 模式 -- ##
        ## -- -- 定义于 karra_diffusion.py / consistency_loss()
        if self.training_mode == "consistency_training":
            compute_losses = functools.partial(
                self.diffusion.consistency_losses,
                self.model,
                trajectory,
                num_scales,
                target_model=self.target_model,
                local_cond = local_cond,
                global_cond = global_cond,
            )
        else:
            raise ValueError(f"Warning training mode {self.training_mode}")

        ## -- 计算 loss -- ##
        losses = compute_losses() ## 重点

        ## 当 sampler = LossSecondMomentResampler 才会启动
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        ## -- loss 加权取平均 -- ##
        loss = (losses["loss"] * weights).mean()

        return loss


    def update_target_ema(self, global_step):

        ## Note: 此处针对原代码（consistency model）进行改动：
        ## 为防止 master_params 和 target_model_master_params 没有随着模型更新
        ## 此处 显性地实时地同步一遍 master_params 和 target_model_master_params
        self.master_params = make_master_params(
            self.param_groups_and_shapes
        )
        self.target_model_master_params = make_master_params(
            self.target_model_param_groups_and_shapes
        )

        target_ema, scales = self.ema_scale_fn(global_step)
        with torch.no_grad():
            update_ema(
                self.target_model_master_params,
                self.master_params,
                rate = target_ema,
            )
            master_params_to_model_params(
                self.target_model_param_groups_and_shapes,
                self.target_model_master_params,
            )
        # print(self.target_model_master_params)



