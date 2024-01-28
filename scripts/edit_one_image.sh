python main.py  --edit_one_image \
                --config church.yml \
                --exp ./runs/test1 \
                --t_0 500 \
                --n_inv_step 500 \
                --n_test_step 500 \
                --n_iter 1 \
                --img_path /data/kylee/projects/sound-guided-semantic-image-manipulation/data/gan_data/2.png \
                --model_path /data/kylee/projects/sound-guided-semantic-image-manipulation/checkpoint/church_fire_cl1.3_id0_l20_sl0.8_vgg0.1_niter20_for40_gen6_b16_FT_church_outdoor_church_fire_t500_ninv40_ngen6_id0.0_l10.3_lr8e-06_Church_on_fire/6.pth
