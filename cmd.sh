## Memo: 使用 AAA=xxx 来覆盖 hydra config
## 例如： python train.py --config-name=train_diffusion_unet_image_workspace task.dataset_path=data/pusht


## === lowdim === ##

## pushT
python train.py --config-name train_diffusion_unet_lowdim_workspace task=pusht_lowdim
## blockpush
python train.py --config-name train_diffusion_unet_lowdim_workspace task=blockpush_lowdim_seed
# Kitchen = not works
python train.py --config-name train_diffusion_unet_lowdim_workspace task=kitchen_lowdim_abs
# robotmimic
python train.py --config-name train_diffusion_unet_lowdim_workspace task=square_lowdim
python train.py --config-name train_diffusion_unet_lowdim_workspace task=can_lowdim
python train.py --config-name train_diffusion_unet_lowdim_workspace task=lift_lowdim
python train.py --config-name train_diffusion_unet_lowdim_workspace task=tool_hang_lowdim_abs
python train.py --config-name train_diffusion_unet_lowdim_workspace task=transport_lowdim_abs


## === image === ##
python train.py --config-name train_diffusion_unet_image_workspace task=lift_image_abs
python train.py --config-name train_diffusion_unet_image_workspace task=square_image_abs


## === lowdim DDIM === ##
python train.py --config-name train_diffusion_unet_lowdim_workspace_DDIM task=square_lowdim

## === lowdim Consistency Model === ##
python train.py --config-name train_diffusion_unet_lowdim_workspace_CM task=square_lowdim


## === Evaluation === ##
python eval.py --checkpoint data/outputs/2023.11.30/18.51.09_train_diffusion_unet_lowdim_pusht_lowdim/checkpoints/latest.ckpt -o data/eval/pusht_lowdim_20231103_185109
python eval.py --checkpoint data/outputs/2023.12.01/19.07.36_train_diffusion_unet_lowdim_transport_lowdim/checkpoints/latest.ckpt -o data/eval/transport_lowdim_20231203_190736
python eval.py --checkpoint data/outputs/2023.12.03/23.21.48_train_diffusion_unet_lowdim_square_lowdim/checkpoints/latest.ckpt -o data/eval/square_lowdim_20231203_232148
python eval.py --checkpoint data/outputs/2023.12.05/18.04.47_train_diffusion_unet_lowdim_square_lowdim/checkpoints/latest.ckpt -o data/eval/square_lowdim_20231205_180447_ddim

## === consistency model === ##
python train.py --config-name train_diffusion_unet_lowdim_workspace_CM task=square_lowdim
python train.py --config-name train_diffusion_unet_lowdim_workspace_CM task=pusht_lowdim

python eval.py --checkpoint data/outputs/2023.12.30/14.00.44_train_consistency_unet_lowdim_square_lowdim/checkpoints/latest.ckpt -o data/eval/square_cm_20231230_latest_r2
python eval.py --checkpoint data/outputs/2023.12.28/19.17.46_train_consistency_unet_lowdim_pusht_lowdim/checkpoints/latest.ckpt -o data/eval/pushT_cm_20231228_latest_one_step

