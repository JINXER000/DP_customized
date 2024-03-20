## Memo: 使用 AAA=xxx 来覆盖 hydra config
## 例如： python train.py --config-name=train_diffusion_unet_image_workspace task.dataset_path=data/pusht


## === lowdim === ##

## pushT
# python train.py --config-name train_diffusion_unet_lowdim_workspace task=can_lowdim_abs

## === image === ##
python train.py \
    --config-name train_diffusion_unet_ddim_image_workspace \
    task=sim_insertion_scripted

python train.py \
    --config-name train_diffusion_unet_ddim_image_workspace \
    task=sim_transfer_cube_human

# python train.py --config-name train_diffusion_transformer_image_workspace task=square_image_abs


## === Evaluation === ##
# python eval.py \
#     -c data/outputs/2024.01.17/14.18.59_train_diffusion_unet_image_sim_transfer_cube_human/checkpoints/epoch=0300-test_mean_score=2.600.ckpt \
#     -o data/eval/sim_cube_transfer_scripted/2024.01.17_14.18.59
