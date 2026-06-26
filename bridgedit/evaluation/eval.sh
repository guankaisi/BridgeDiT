source /miniforge/etc/profile.d/conda.sh
conda activate myenv
CUDA_VISIBLE_DEVICES=1,2,3 python eval.py \
    --generated_video /mnt/task_runtime/t2av/baselines/codi_result/codi_video \
    --generated_audio /mnt/task_runtime/t2av/baselines/codi_result/codi_audio \
    --metrics desync
# conda activate myenv
# python eval.py \
#     --generated_video /mnt/task_runtime/t2av/baselines/javis_result_480p \
#     --generated_audio /mnt/task_runtime/t2av/baselines/javis_result_480p \
#     --video_caption_path /mnt/task_runtime/t2av/text_favd/test.jsonl\
#     --audio_caption_path /mnt/task_runtime/t2av/text_favd/test.jsonl\
#     --metrics clipsim
    
 