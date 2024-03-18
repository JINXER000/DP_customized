## == aloha train == ##
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python train.py \
    --config-name train_diffusion_unet_ddim_image_workspace_real \
    task=aloha_insert_10s_random_init


## == aloha eval == ##
python eval_aloha.py \
 -i ./data/outputs/2024.02.11/23.41.10_train_diffusion_unet_image_aloha_insert_10s.yaml/checkpoints/latest.ckpt \
 -o ./data/outputs/2024.02.11/23.41.10_train_diffusion_unet_image_aloha_insert_10s.yaml/
