import torch
from audioldm_eval import EvaluationHelper

def Cal_FAD(target_audio, generated_audio):
    device = torch.device(f"cuda:{0}")
    evaluator = EvaluationHelper(16000, device,backbone='cnn14')
    metrics = evaluator.main(
        target_audio,
        generated_audio
    )

if __name__ == "__main__":
    # GPU acceleration is preferred
    device = torch.device(f"cuda:{0}")

    generation_result_path = "/mnt/task_runtime/t2av/baselines/result/codi_audio"
    target_audio_path = "/mnt/task_runtime/dataset/favdbench/video/test"

    # Initialize a helper instance
    evaluator = EvaluationHelper(16000, device,backbone='cnn14')

    # Perform evaluation, result will be print out and saved as json
    metrics = evaluator.main(
        generation_result_path,
        target_audio_path
    )