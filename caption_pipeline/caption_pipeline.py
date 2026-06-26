from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch
import os
import json
from tqdm import tqdm
import librosa
import argparse
import warnings

from paths import model_path

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
# 设置 NCCL_DEBUG 环境变量为 ERROR，禁用详细日志
os.environ["NCCL_DEBUG"] = "ERROR"

video_caption_template = """You are an expert video analyst and prompt engineer. Your goal is to watch a video and create a highly effective, descriptive prompt that can be used by a text-to-video generation model to recreate the visual essence and physical dynamics of the scene.
Your generated prompt should be a dense, continuous paragraph rich with visual details. Follow this thinking process:
1.  **Scene and Atmosphere**:
    * Describe the core environment and the overall mood. Translate any auditory feeling (e.g., tension from music) into visual terms (e.g., `high-contrast lighting, deep shadows`).
2.  **Subjects and Details**:
    * Identify the main subjects and objects. Describe them with specific visual adjectives (`a weathered blacksmith with soot-stained hands`, `a glowing orange piece of iron`).
3.  **Key Actions, Cinematography, and Physical Dynamics**:
    * Describe the sequence of most important actions.
    * Describe the cinematography (shot type, angle, movement).
    * **[NEW CORE RULE] Describe the "Visual Counterpart" of Sound**: Instead of describing sound itself, describe the physical actions that *create* sound. Focus on impact, interaction, and motion that implies sound.
        * **For Speech/Vocalization**: Detail the mouth movements, facial expressions, and throat or chest movements (`a lion opens its massive jaws wide, a deep roar building in its chest`).
        * **For Impacts**: Describe the collision, the reaction, and the result (`a heavy hammer strikes the glowing iron, sending a shower of bright orange sparks flying into the air; the metal visibly deforms under the blow`).
        * **For Movement/Friction**: Detail the interaction between surfaces (`a car's tires screech, leaving black rubber marks on the asphalt as it drifts around a corner`).
        * **For Natural Forces**: Describe the effect of the force on the environment (`trees bend and sway violently under the force of the wind, loose leaves are whipped into a frenzy`).
4.  **Visual Style and Quality**:
    * Specify the artistic style (`photorealistic`, `cinematic`), lighting (`dramatic, warm light from the forge`), and visual quality (`highly detailed, 8K`).
**Final Instruction**: Synthesize all these visual and physical elements into a single, rich, and coherent paragraph. Your entire output should be a prompt that visually and dynamically directs an AI. **Do NOT describe sound itself, but rather the physics of its creation.**
Based on the video, the prompt for video generation is:"""

audio_caption_template = """ You are an expert audio analyst and prompt engineer. Your goal is to listen to an audio clip and create a **concise and efficient** prompt that allows a text-to-audio model to recreate the scene.
Based on the audio, the concise prompt for audio generation is (please in english):"""

rewrite_template = """Combine the provided Video Caption (visuals) and Audio Caption (sounds) into a single unified description.
**Constraints:**
- The output must explicitly include information from both modalities.
- Do not hallucinate information not present in the source captions.
- Keep the sentence structure concise.
**Source Data:**
- Video Caption: {video_caption}
- Audio Caption: {audio_caption}
**Response:**
Audio-Video Caption:"""

# Default resolved via paths.model_path() at runtime.
MODEL_PATH = None

def load_mllm_model(model_path=MODEL_PATH):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        limit_mm_per_prompt={"image": 10, "video": 10},
        dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return llm, processor

def load_llm(model_path=MODEL_PATH):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=8,
        dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return llm, tokenizer

def load_audio_llm_model(model_path=MODEL_PATH):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        limit_mm_per_prompt={"audio": 1},
        dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return llm, processor

def process_video_input(video_files_list, processor):
    video_inputs_list = []
    for video_file in video_files_list:
        video_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                    {"type": "text", "text": video_caption_template},
                    {
                        "type": "video", 
                        "video": video_file,
                        "total_pixels": 1280 * 28 * 28, "min_pixels": 16 * 28 * 28
                    }
                ]
            },
        ]
        prompt = processor.apply_chat_template(
            video_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        _, video_inputs, video_kwargs = process_vision_info(video_messages, return_video_kwargs=True)
        # Check and remove unnecessary keys from video_kwargs
        mm_data = {}
        mm_data["video"] = video_inputs
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
        video_inputs_list.append(llm_inputs)
    return video_inputs_list

def build_qwen_prompt(question):
    # Qwen2-Audio 在 vLLM 中的官方 Prompt 格式
    # 必须显式包含 <|AUDIO|> 占位符以及起止符
    audio_placeholder = "<|audio_bos|><|AUDIO|><|audio_eos|>"
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{audio_placeholder}\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=1024,
    stop_token_ids=[],
)

def generate_video_caption(video_js=None, batch_size=1, mllm=None, processor=None):
    video_files_list = list(video_js.keys())
    print("Lens of video_js: ", len(video_js))
    for i in tqdm(range(0, len(video_files_list), batch_size)):
        video_batch_inputs = process_video_input(video_files_list[i:i+batch_size], processor)
        outputs = llm.generate(video_batch_inputs, sampling_params=sampling_params, use_tqdm=False)
        gen_captions = [output.outputs[0].text for output in outputs]
        for j in range(batch_size):
            try:
                video_id = video_files_list[i+j]
                video_js[video_id]['video_caption'] = gen_captions[j]
            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                del video_js[video_id]
                continue
    return video_js

def rewrite_caption(video_js, batch_size=1, llm=None, tokenizer=None):
    keys_list = list(video_js.keys())
    for i in tqdm(range(0, len(video_files_list), batch_size)):
        batch_keys_list = keys_list[i : i+batch_size]
        video_captions = [video_js[key]['video_caption'] for key in batch_keys_list]
        audio_captions = [video_js[key]['audio_caption'] for key in batch_keys_list]
        llm_inputs = []
        for idx,video_caption in enumerate(video_captions):
            llm_messages = [
                    {"role": "system", "content": "You are a helpful assisstant for caption."},
                    {"role": "user", "content": rewrite_template.format(
                            video_caption=video_caption,
                            audio_caption=audio_captions[idx]
                        )
                    }
                ]
            llms_input = tokenizer.apply_chat_template(
                llm_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            llm_inputs.append(llms_input)
        llm_caption_outputs = llm.generate(llm_inputs, sampling_params, use_tqdm=False)
        av_captions = [output.outputs[0].text for output in llm_caption_outputs]
        for j in range(batch_size):
            video_id = batch_keys_list[j]
            video_js[video_id]['audio_video_caption'] = av_captions[j]
    return video_js


def process_audio_input(audio_path):
    # 使用 librosa 读取音频，sr=None 保持原始采样率 (或者指定 sr=16000)
    # Qwen2-Audio 通常对采样率适应性较好，但 vLLM 内部会处理
    y, sr = librosa.load(audio_path, sr=None) 
    return (y, sr)

def build_qwen_prompt(question):
    # Qwen2-Audio 在 vLLM 中的官方 Prompt 格式
    # 必须显式包含 <|AUDIO|> 占位符以及起止符
    audio_placeholder = "<|audio_bos|><|AUDIO|><|audio_eos|>"
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{audio_placeholder}\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt

def generate_audio_caption(video_js, video_files_list, batch_size=1, llm=None):
    for i in tqdm(range(0, len(video_files_list), batch_size)):
        batch_files = video_files_list[i : i+batch_size]
        
        inputs_batch = []
        valid_indices = [] # 记录这一批里哪些文件是有效的，方便回填结果

        for j, video_file in enumerate(batch_files):
            try:
                # 准备音频数据
                audio_data = process_audio_input(video_file) # (y, sr)
                
                # 准备 Prompt
                prompt_text = build_qwen_prompt(audio_caption_template)

                # 构造 vLLM 标准输入格式
                inputs_batch.append({
                    "prompt": prompt_text,
                    "multi_modal_data": {
                        "audio": audio_data 
                    }
                })
                valid_indices.append(j)
            except Exception as e:
                print(f"Error loading {video_file}: {e}")
        
        if not inputs_batch:
            continue

        # 3. 批量生成
        sampling_params = SamplingParams(temperature=0.2, max_tokens=256)
        outputs = llm.generate(inputs_batch, sampling_params=sampling_params, use_tqdm=False)
        
        # 4. 回填结果
        for k, output in enumerate(outputs):
            # 找到原始对应的 video_file
            original_idx = valid_indices[k]
            video_id = batch_files[original_idx]
            
            caption = output.outputs[0].text
            video_js[video_id]['audio_caption'] = caption

    return video_js



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--audio_llm_path", type=str, default=None)
    parser.add_argument("--mllm_path", type=str, default=None)
    parser.add_argument("--llm_path", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="recaption/avsync-test-0129-final.json")
    parser.add_argument("--input_json", type=str, default=None,
                        help="Optional JSON keyed by video paths with raw captions.")
    args = parser.parse_args()

    mllm_path = args.mllm_path or model_path("Qwen2.5-VL-72B-Instruct")
    llm_path = args.llm_path or model_path("Qwen2.5-72B-Instruct")
    audio_llm_path = args.audio_llm_path or model_path("Qwen2-Audio-7B-Instruct")

    output_file = args.output_file
    batch_size = args.batch_size

    video_path = args.video_path

    # 准备文件列表
    # video_files_list = [
    #     os.path.join(args.video_path, file) 
    #     for file in os.listdir(args.video_path) 
    #     if file.endswith(('.mp4', '.wav', '.mp3'))
    # ]
    
    # video_js = {f: {} for f in video_files_list}
    input_json = args.input_json or os.path.join(os.path.dirname(__file__), "recaption/avsync-test-72B-captions.json")
    with open(input_json, 'r') as f:
        video_js = json.load(f)
    video_files_list = list(video_js.keys())
    # mllm, processor = load_mllm_model(mllm_path)
    # video_js =generate_video_caption(mllm=mllm,processor=processor, video_path=video_path, batch_size=batch_size)
    # del mllm, processor
    # audio_llm, processor = load_audio_llm_model(audio_llm_path)
    # video_js = generate_audio_caption(video_js, video_files_list, batch_size=batch_size, llm=audio_llm)
    # del audio_llm, processor
    llm, tokenizer = load_llm(llm_path)

    video_js = rewrite_caption(video_js, batch_size=batch_size, llm=llm, tokenizer=tokenizer)
    with open(output_file, 'w') as f:
        json.dump(video_js, f, indent=4, ensure_ascii=False)


