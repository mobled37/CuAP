export CUDA_VISIBLE_DEVICES=0
clip_weight=1.3
id_weight=0.3
l1_weight=0.3
l2_weight=0
# soundclip_weight=0.8
vgg_weight=0.3
n_iter=15
S_for=40
S_gen=6
t_0=500

for soundclip_weight in 0.8 0; do
    for edit_attr in giggling CuAP_giggling CuPL_giggling ACT_giggling; do
        python main.py --optimization          \
        --config /data/kylee/projects/sound-guided-semantic-image-manipulation/configs/celeba.yml    \
        --audio_path audiosample/giggling.wav \
        --model_path /data/kylee/projects/sound-guided-semantic-image-manipulation/pretrained/diffusionclip/celeba_hq.ckpt \
        --comment ${edit_attr}  \
        --exp ./runs/main_${edit_attr}_sound1_ai3_cl${clip_weight}_id${id_weight}_l2${l2_weight}_sl${soundclip_weight}_vgg${vgg_weight}_niter${n_iter}_for${S_for}_gen${S_gen}      \
        --edit_attr ${edit_attr} \
        --audio_encoder_path soundclip/pretrained/audio_encoder/audio_encoder49.pth \
        --results_dir ./results/exp10 \
        --do_train 1             \
        --do_test 1              \
        --n_train_img 68         \
        --n_test_img 340          \
        --n_iter ${n_iter}               \
        --t_0 ${t_0}               \
        --n_inv_step ${S_for}          \
        --n_train_step ${S_gen}         \
        --n_test_step ${S_for}         \
        --lr_clip_finetune 8e-6  \
        --clip_loss_w ${clip_weight}          \
        --id_loss_w ${id_weight}            \
        --l1_loss_w ${l1_weight}            \
        --l2_loss_w ${l2_weight}            \
        --soundclip_loss_w ${soundclip_weight}     \
        --vgg_loss_w ${vgg_weight} \
        --audio_intensity 3
    done
done

for soundclip_weight in 0.8 0; do
    for edit_attr in sobbing CuAP_sobbing CuPL_sobbing ACT_sobbing; do
        python main.py --optimization          \
        --config /data/kylee/projects/sound-guided-semantic-image-manipulation/configs/celeba.yml    \
        --audio_path audiosample/sobbing.wav \
        --model_path /data/kylee/projects/sound-guided-semantic-image-manipulation/pretrained/diffusionclip/celeba_hq.ckpt \
        --comment ${edit_attr}  \
        --exp ./runs/main_${edit_attr}_sound1_ai3_cl${clip_weight}_id${id_weight}_l2${l2_weight}_sl${soundclip_weight}_vgg${vgg_weight}_niter${n_iter}_for${S_for}_gen${S_gen}      \
        --edit_attr ${edit_attr} \
        --audio_encoder_path soundclip/pretrained/audio_encoder/audio_encoder49.pth \
        --results_dir ./results/exp10 \
        --do_train 1             \
        --do_test 1              \
        --n_train_img 68         \
        --n_test_img 340          \
        --n_iter ${n_iter}               \
        --t_0 ${t_0}               \
        --n_inv_step ${S_for}          \
        --n_train_step ${S_gen}         \
        --n_test_step ${S_for}         \
        --lr_clip_finetune 8e-6  \
        --clip_loss_w ${clip_weight}          \
        --id_loss_w ${id_weight}            \
        --l1_loss_w ${l1_weight}            \
        --l2_loss_w ${l2_weight}            \
        --soundclip_loss_w ${soundclip_weight}     \
        --vgg_loss_w ${vgg_weight} \
        --audio_intensity 3
    done
done