## Memo: 使用 AAA=xxx 来覆盖 hydra config
## 例如： python train.py --config-name=train_diffusion_unet_image_workspace task.dataset_path=data/pusht


python train.py \
    --config-name train_diffusion_unet_ddim_image_workspace \
    task=sim_transfer_cube_human

## == aloha real data == ##

CUDA_VISIBLE_DEVICES=2 python train.py \
    --config-name train_diffusion_unet_ddim_image_workspace_real \
    task=aloha_insert_10s 
    
    > log_insert.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python train.py \
    --config-name train_diffusion_unet_ddim_image_workspace_real \
    task=aloha_battery 
    
    > log_battery.log 2>&1 &

## == aloha eval == ##

python eval_aloha.py \
 -i ./data/outputs/2024.02.11/23.41.10_train_diffusion_unet_image_aloha_insert_10s.yaml/checkpoints/latest.ckpt \
 -o ./data/outputs/2024.02.11/23.41.10_train_diffusion_unet_image_aloha_insert_10s.yaml/
