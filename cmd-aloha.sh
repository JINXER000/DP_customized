## Memo: 使用 AAA=xxx 来覆盖 hydra config
## 例如： python train.py --config-name=train_diffusion_unet_image_workspace task.dataset_path=data/pusht


## == aloha real data - diffusion == ##

CUDA_VISIBLE_DEVICES=1 python train.py \
    --config-name train_diffusion_unet_ddim_image_workspace_real \
    task=aloha_insert_10s \
    # > ./data/log/DP_log_insert.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python train.py \
    --config-name train_diffusion_unet_ddim_image_workspace_real \
    task=aloha_insert_10s_random_init \
    # > ./data/log/DP_log_insert_random.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 python train.py \
    --config-name train_diffusion_unet_ddim_image_workspace_real \
    task=aloha_battery \
    # > ./data/log/DP_log_battery.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 python train.py \
    --config-name train_diffusion_unet_ddim_image_workspace_real \
    task=aloha_ziploc \
    # > ./data/log/DP_log_battery.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python train.py \
    --config-name train_diffusion_unet_ddim_image_workspace_real \
    task=aloha_towel \


CUDA_VISIBLE_DEVICES=1 python train.py \
    --config-name train_diffusion_unet_ddim_image_workspace_real \
    task=aloha_screwdriver \

CUDA_VISIBLE_DEVICES=2 python train.py \
    --config-name train_diffusion_unet_ddim_image_workspace_real \
    task=aloha_starbucks 

CUDA_VISIBLE_DEVICES=2 python train.py \
    --config-name train_diffusion_transformer_image_workspace \
    task=aloha_conveyor 


CUDA_VISIBLE_DEVICES=3 python train.py \
    --config-name train_diffusion_unet_ddim_image_workspace_real \
    task=aloha_hang_pants 


## == aloha real data - consistency == ##


CUDA_VISIBLE_DEVICES=2 python train.py \
    --config-name train_consistency_unet_image_workspace_aloha \
    task=aloha_insert_10s_random_init


CUDA_VISIBLE_DEVICES=3 python train.py \
    --config-name train_consistency_unet_image_workspace_aloha \
    task=aloha_ziploc 

CUDA_VISIBLE_DEVICES=2 python train.py \
    --config-name train_consistency_unet_image_workspace_aloha \
    task=aloha_towel

CUDA_VISIBLE_DEVICES=3 python train.py \
    --config-name train_consistency_unet_image_workspace_aloha \
    task=aloha_battery


CUDA_VISIBLE_DEVICES=3 python train.py \
    --config-name train_consistency_unet_image_workspace_aloha \
    task=aloha_insert_10s


CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-name train_consistency_unet_image_workspace_aloha \
    task=aloha_screwdriver 


CUDA_VISIBLE_DEVICES=3 python train.py \
    --config-name train_consistency_unet_image_workspace_aloha \
    task=aloha_starbucks

CUDA_VISIBLE_DEVICES=3 python train.py \
    --config-name train_consistency_unet_image_workspace_aloha \
    task=aloha_conveyor



## == aloha simulation - diffusion == ##
CUDA_VISIBLE_DEVICES=2 python train.py \
    --config-name train_diffusion_unet_ddim_image_workspace \
    task=sim_insertion_scripted

CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-name train_diffusion_unet_ddim_image_workspace \
    task=sim_transfer_cube_human



## == aloha simulation - consistency == ##

CUDA_VISIBLE_DEVICES=1 python train.py \
    --config-name train_consistency_unet_image_workspace_aloha_sim \
    task=sim_insertion_scripted

CUDA_VISIBLE_DEVICES=3 python train.py \
    --config-name train_consistency_unet_image_workspace_aloha_sim \
    task=sim_transfer_cube_scripted




## == aloha eval == ##
python eval_aloha.py \
 -i ./data/outputs/2024.02.11/23.41.10_train_diffusion_unet_image_aloha_insert_10s.yaml/checkpoints/latest.ckpt \
 -o ./data/outputs/2024.02.11/23.41.10_train_diffusion_unet_image_aloha_insert_10s.yaml/

python eval_aloha.py \
 -i ./data/outputs/2024.03.18/16.08.45_train_diffusion_unet_image_aloha_insert_10s_random_init/checkpoints/latest.ckpt \
 -o ./data/outputs/2024.03.18/16.08.45_train_diffusion_unet_image_aloha_insert_10s_random_init/

 ## == demo == ##

python eval_aloha.py \
    -i /home/xuhang/Desktop/xh-codes/Diffusion-Policy/data/outputs/2024.04.18/19.40.33_train_diffusion_unet_image_aloha_screwdriver/checkpoints/latest.ckpt \
    -o nan

python eval_aloha.py \
    -i /home/xuhang/Desktop/xh-codes/Diffusion-Policy/data/outputs/2024.04.18/19.42.00_train_consistency_unet_image_aloha_screwdriver/checkpoints/latest.ckpt \
    -o nan

## == aloha eval - conveyor == ##

python eval_aloha.py \
 -i /home/xuhang/Desktop/xh-codes/Diffusion-Policy/data/outputs/2024.06.06/23.48.30_train_consistency_unet_image_aloha_conveyor/checkpoints/latest.ckpt \
 -o nan

 python eval_aloha.py \
 -i /home/xuhang/Desktop/xh-codes/Diffusion-Policy/data/outputs/2024.06.06/23.49.27_train_diffusion_unet_image_aloha_conveyor/checkpoints/latest.ckpt \
 -o nan


 ## == aloha eval - starbucks == ##

python eval_aloha.py \
 -i /home/xuhang/Desktop/xh-codes/Diffusion-Policy/data/outputs/2024.06.08/12.53.50_train_consistency_unet_image_aloha_starbucks/checkpoints/latest.ckpt \
 -o nan

 python eval_aloha.py \
 -i /home/xuhang/Desktop/xh-codes/Diffusion-Policy/data/outputs/2024.06.08/12.52.52_train_diffusion_unet_image_aloha_starbucks/checkpoints/latest.ckpt  \
 -o nan

### aloha eval - hang_pants ###


python eval_aloha_hitl.py \
 -i /ssd1/xuhang/dp_ckpt/2024.12.08/00.51.22_train_diffusion_unet_image_aloha_hang_pants/checkpoints/latest.ckpt  \
 -o /home/xuhang/Desktop/xh-codes/Diffusion-Policy/data/eval/hang_pants/ \
 -t 1000

 python eval_aloha_hitl.py \
 -i /ssd1/xuhang/dp_ckpt/2024.12.08/00.53.02_train_consistency_unet_image_aloha_hang_pants/checkpoints/latest.ckpt  \
 -o /home/xuhang/Desktop/xh-codes/Diffusion-Policy/data/eval/hang_pants/ \
 -t 1000

### train aloha cyberport with longer horizon
 CUDA_VISIBLE_DEVICES=4 python train.py --config-name train_diffusion_unet_ddim_image_workspace_real task=aloha_transfer_tape    horizon=32 n_obs_steps=1 n_action_steps=25 