for i in {1..5}
do
    python eval_aloha.py \
    -i /home/xuhang/Desktop/xh-codes/Diffusion-Policy/data/outputs/2024.06.08/12.53.50_train_consistency_unet_image_aloha_starbucks/checkpoints/latest.ckpt \
    -o nan \
    -md 400

    /home/xuhang/miniforge3/envs/aloha/bin/python ~/interbotix_ws/src/aloha/aloha_scripts/sleep.py

    echo "Done $i"
done
