#!/bin/bash

python -m scripts.inference --unet_config_path "configs/unet/stage2_512.yaml" --inference_ckpt_path "checkpoints/latentsync_unet.pt" --inference_steps 20 --guidance_scale 1.5 --enable_deepcache --video_path "assets/thl2_trimmed.mp4" --audio_path "assets/thl_trimmed.wav" --video_out_path "video_out.mp4"
