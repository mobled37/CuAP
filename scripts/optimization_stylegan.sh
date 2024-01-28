python optimization/run_optimization.py \
    --lambda_similarity 0.002 \
    --lambda_identity 0.0 \
    --truncation 0.7 \
    --lr 0.1 \
    --audio_path ./audiosample/explosion.wav \
    --ckpt ./pretrained_models/landscape.pt \
    --stylegan_size 256 
