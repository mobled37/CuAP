export CUDA_VISIBLE_DEVICES=0
edit_attribute="church_fire"
clip_weight=1.3
id_weight=0
l1_weight=0.3
l2_weight=0
soundclip_weight=0.8
vgg_weight=0.1
n_iter=7
S_for=40
S_gen=6
t_0=500
audio_intensity=3

python main.py --optimization          \
--config /data/kylee/projects/sound-guided-semantic-image-manipulation/configs/church.yml    \
--audio_path audiosample/fire.wav \
--model_path /data/kylee/projects/sound-guided-semantic-image-manipulation/pretrained/diffusionclip/church_outdoor.ckpt \
--comment ${edit_attribute}   \
--exp ./runs/large_${edit_attribute}_cl${clip_weight}_id${id_weight}_l2${l2_weight}_sl${soundclip_weight}_vgg${vgg_weight}_niter${n_iter}_for${S_for}_gen${S_gen}_b16      \
--edit_attr ${edit_attribute}  \
--audio_encoder_path soundclip/pretrained/audio_encoder/audio_encoder49.pth \
--results_dir ./results/exp10 \
--do_train 1             \
--do_test 1              \
--n_train_img 100         \
--n_test_img 100          \
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
--audio_intensity ${audio_intensity}

# Basic setting, (S_for, S_gen) = (40, 6)
# Setting change to (S_for, S_gen) = (200, 40)
# --n_inv_step 40          \
# --n_train_step 6         \
# --n_test_step 40
