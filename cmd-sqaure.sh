## Memo: 使用 AAA=xxx 来覆盖 hydra config
## 例如： python train.py --config-name=train_diffusion_unet_image_workspace task.dataset_path=data/pusht


## === lowdim === ##

## pushT
# python train.py --config-name train_diffusion_unet_lowdim_workspace task=can_lowdim_abs

## === image === ##
python train.py --config-name train_diffusion_unet_ddim_image_workspace task=sim_transfer_cube_image

# python train.py --config-name train_diffusion_transformer_image_workspace task=square_image_abs


## === Evaluation === ##
# python eval.py --checkpoint data/outputs/2024.01.14/21.39.52_train_diffusion_unet_image_sim_transfer_cube_scripted/checkpoints/latest.ckpt -o data/eval/sim_cube_transfer_scripted/2024.01.14_21.39.52
