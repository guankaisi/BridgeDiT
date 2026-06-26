"""
CRR.py — Cross-Referential Rewriter caption framework
(ECCV 2026 Submission #1528, "Taming Text-to-Sounding Video Generation
via Advanced Modality Condition and Interaction").

Pipeline:
  Training:  Sounding Video
              --(Qwen2.5-VL)-->  T_V^raw
              --(Qwen2-Audio)--> T_A^raw
              --(Qwen2.5-72B Semantic Checker, cross-reference)--> Semantic Anchors
              --(Qwen2.5-72B Cross-Modal Rewriter)--> (T_V, T_A)

  Inference: User prompt
              --(Qwen2.5-72B Semantic Checker, context inference)--> Semantic Anchors
              --(Qwen2.5-72B Cross-Modal Rewriter)--> (T_V, T_A)
"""

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch
import os
import re
import json
from tqdm import tqdm
import librosa
import argparse
import warnings

from vllm_utils import get_tensor_parallel_size
from paths import model_path

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
os.environ["NCCL_DEBUG"] = "ERROR"


# ============================================================
# Prompt Templates (verbatim from paper Appendix C.6 / D.6)
# ============================================================

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

# ---- Semantic Checker — Training Stage (paper, App. D.6) ----
semantic_checker_train_template = """You are a multimodal semantic analyst. You will receive two raw captions describing the same video clip: a Video Raw Caption and an Audio Raw Caption. These may contain inconsistencies or hallucinations (especially in the audio caption).

Your task is to cross-reference both captions and extract ONLY the semantically grounded information into a structured JSON called Semantic Anchors.

Follow these rules strictly:
1. Cross-Reference: Use the video caption as the grounding reference. Every sound event in the output MUST have a corresponding visual action in the video caption. If the audio caption describes a sound with no visual source, DISCARD it.
2. Identify Conflicts: If the audio caption contradicts the video caption (e.g., audio says "bird chirping" but video shows only mechanical actions), flag and remove the conflicting audio description.
3. Extract Attributes: For each verified event, extract: Entity (who/what), Environment (where), Action (what visual action occurs), Sound (what sound this action produces).

Output ONLY a JSON object in the following format, with no additional text:
{{
  "anchors": [
    {{
      "entity": "...",
      "environment": "...",
      "action": "...",
      "sound": "..."
    }}
  ],
  "discarded": ["list of audio descriptions removed due to no visual grounding"]
}}

Video Raw Caption: {video_raw_caption}
Audio Raw Caption: {audio_raw_caption}"""

# ---- Semantic Checker — Inference Stage (paper, App. D.6) ----
semantic_checker_infer_template = """You are a multimodal semantic analyst. You will receive a short user prompt describing a scene. Your task is to infer the implicit visual and acoustic events from this brief description and produce a structured JSON called Semantic Anchors.

Follow these rules strictly:
1. Context Inference: Deduce the likely entities, environment, physical actions, and resulting sounds from the user's description. Stay faithful to the prompt — do not invent unrelated events.
2. Physical Grounding: Every sound event MUST be caused by a plausible physical action. Do not add sounds without a clear causal source.
3. Richness: Expand the brief prompt into detailed attributes. For example, "a man hammering iron" should yield specific environment (workshop), specific action (hammer striking heated metal), and specific sound (metallic clang with sparks hissing).

Output ONLY a JSON object in the following format, with no additional text:
{{
  "anchors": [
    {{
      "entity": "...",
      "environment": "...",
      "action": "...",
      "sound": "..."
    }}
  ]
}}

User Prompt: {user_prompt}"""

# ---- Cross-Modal Rewriter (paper, App. D.6) ----
cross_modal_rewriter_template = """You are an expert caption writer for a Text-to-Sounding-Video generation system. You will receive a JSON object called Semantic Anchors containing verified entity-action-sound correspondences.

Your task is to generate TWO separate, dense captions from these anchors: a Video Caption and an Audio Caption.

Follow these rules strictly:
1. Strict Grounding: Use ONLY the information in the provided Semantic Anchors. Do NOT introduce any events, entities, or sounds not present in the anchors.
2. Modality Purity:
   - Video Caption: Describe ONLY visual elements — appearance, lighting, motion, camera angles, colors, textures, and physical dynamics. Do NOT use any sound-related words (e.g., "loud", "echo", "clang", "sound", "noisy").
   - Audio Caption: Describe ONLY auditory elements — sound type, rhythm, timbre, volume, pitch, and acoustic environment. Do NOT use any visual words (e.g., "red", "bright", "camera", "shadow").
3. Dense and Cinematic: Write rich, detailed descriptions suitable for training generation models. Use precise vocabulary and dynamic phrasing (e.g., instead of "a man hits metal", write "a weathered blacksmith raises a heavy hammer above the glowing iron, muscles tensing before the forceful downward strike sends a shower of bright orange sparks into the air").
4. Temporal Alignment: Both captions must describe the same temporal sequence. Visual actions and their corresponding sounds should align in order and timing.

Semantic Anchors:
{semantic_anchors_json}

Output in the following format:
CRR Video Caption: [your pure-video description]
CRR Audio Caption: [your pure-audio description]"""


# ============================================================
# Sampling parameters
# ============================================================

# For MLLM (video) raw caption generation — same as caption_pipeline.py
mllm_sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=1024,
    stop_token_ids=[],
)

# For Audio LLM raw caption generation — same as caption_pipeline.py
audio_sampling_params = SamplingParams(
    temperature=0.2,
    max_tokens=256,
)

# For text LLM (Semantic Checker + Cross-Modal Rewriter)
llm_sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.9,
    repetition_penalty=1.05,
    max_tokens=1536,
    stop_token_ids=[],
)


# ============================================================
# Model Loaders
# ============================================================

def load_mllm_model(model_path):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=get_tensor_parallel_size(4),
        limit_mm_per_prompt={"image": 10, "video": 10},
        dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return llm, processor


def load_audio_llm_model(model_path):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=get_tensor_parallel_size(4),
        limit_mm_per_prompt={"audio": 1},
        dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return llm, processor


def load_llm(model_path):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=get_tensor_parallel_size(1),
        dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return llm, tokenizer


# ============================================================
# Stage 1: Raw caption generation
# ============================================================

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
                    "total_pixels": 1280 * 28 * 28,
                    "min_pixels": 16 * 28 * 28,
                },
            ]},
        ]
        prompt = processor.apply_chat_template(
            video_messages, tokenize=False, add_generation_prompt=True
        )
        _, video_inputs, video_kwargs = process_vision_info(
            video_messages, return_video_kwargs=True
        )
        mm_data = {"video": video_inputs}
        llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
        video_inputs_list.append(llm_inputs)
    return video_inputs_list


def generate_video_raw_caption(video_js, batch_size=1, mllm=None, processor=None):
    """Stage 1a: Qwen2.5-VL produces T_V^raw for each sounding video."""
    video_files_list = list(video_js.keys())
    print(f"[CRR] Generating raw video captions for {len(video_files_list)} videos.")
    for i in tqdm(range(0, len(video_files_list), batch_size), desc="Video Raw Caption"):
        batch = video_files_list[i:i + batch_size]
        try:
            video_batch_inputs = process_video_input(batch, processor)
        except Exception as e:
            print(f"[CRR] Error processing video batch starting at {i}: {e}")
            continue
        outputs = mllm.generate(
            video_batch_inputs, sampling_params=mllm_sampling_params, use_tqdm=False
        )
        gen_captions = [o.outputs[0].text for o in outputs]
        for j, video_id in enumerate(batch):
            try:
                video_js[video_id]['video_raw_caption'] = gen_captions[j]
            except Exception as e:
                print(f"[CRR] Error storing video caption for {video_id}: {e}")
    return video_js


def process_audio_input(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    return (y, sr)


def build_qwen_audio_prompt(question):
    audio_placeholder = "<|audio_bos|><|AUDIO|><|audio_eos|>"
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{audio_placeholder}\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt


def generate_audio_raw_caption(video_js, batch_size=1, llm=None):
    """Stage 1b: Qwen2-Audio produces T_A^raw for each sounding video."""
    video_files_list = list(video_js.keys())
    print(f"[CRR] Generating raw audio captions for {len(video_files_list)} videos.")
    for i in tqdm(range(0, len(video_files_list), batch_size), desc="Audio Raw Caption"):
        batch_files = video_files_list[i:i + batch_size]
        inputs_batch = []
        valid_indices = []
        for j, video_file in enumerate(batch_files):
            try:
                audio_data = process_audio_input(video_file)
                prompt_text = build_qwen_audio_prompt(audio_caption_template)
                inputs_batch.append({
                    "prompt": prompt_text,
                    "multi_modal_data": {"audio": audio_data},
                })
                valid_indices.append(j)
            except Exception as e:
                print(f"[CRR] Error loading audio {video_file}: {e}")

        if not inputs_batch:
            continue

        outputs = llm.generate(
            inputs_batch, sampling_params=audio_sampling_params, use_tqdm=False
        )
        for k, output in enumerate(outputs):
            video_id = batch_files[valid_indices[k]]
            video_js[video_id]['audio_raw_caption'] = output.outputs[0].text
    return video_js


# ============================================================
# Stage 2: Semantic Checker (F_sc)
# ============================================================

def parse_semantic_anchors(text):
    """Robustly extract a JSON object from LLM output (strip code fences,
    take the first {...} block on failure)."""
    text = text.strip()
    # Strip markdown code fences if any
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {"anchors": [], "discarded": []}


def run_semantic_checker_train(video_js, batch_size=1, llm=None, tokenizer=None):
    """Stage 2 (training): cross-reference raw V/A captions to extract grounded
    Semantic Anchors A. Hallucinated audio events without visual grounding are
    discarded."""
    keys_list = list(video_js.keys())
    print(f"[CRR] Semantic Checker (training) on {len(keys_list)} samples.")
    for i in tqdm(range(0, len(keys_list), batch_size), desc="Semantic Checker (Train)"):
        batch_keys = keys_list[i:i + batch_size]
        llm_inputs = []
        for key in batch_keys:
            v_raw = video_js[key].get('video_raw_caption', '')
            a_raw = video_js[key].get('audio_raw_caption', '')
            messages = [
                {"role": "system",
                 "content": "You are a multimodal semantic analyst. Output ONLY valid JSON with no additional text."},
                {"role": "user",
                 "content": semantic_checker_train_template.format(
                     video_raw_caption=v_raw,
                     audio_raw_caption=a_raw,
                 )},
            ]
            llm_input = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            llm_inputs.append(llm_input)

        outputs = llm.generate(llm_inputs, llm_sampling_params, use_tqdm=False)
        for j, output in enumerate(outputs):
            video_id = batch_keys[j]
            raw_text = output.outputs[0].text
            anchors = parse_semantic_anchors(raw_text)
            video_js[video_id]['semantic_anchors'] = anchors
            video_js[video_id]['semantic_anchors_raw'] = raw_text
    return video_js


def run_semantic_checker_infer(user_prompts, batch_size=1, llm=None, tokenizer=None):
    """Stage 2 (inference): expand brief user prompts into Semantic Anchors via
    context inference (no cross-reference; one-side input only).

    Args:
        user_prompts: dict {prompt_id: {'user_prompt': '...'}}
    """
    keys_list = list(user_prompts.keys())
    print(f"[CRR] Semantic Checker (inference) on {len(keys_list)} prompts.")
    for i in tqdm(range(0, len(keys_list), batch_size), desc="Semantic Checker (Infer)"):
        batch_keys = keys_list[i:i + batch_size]
        llm_inputs = []
        for key in batch_keys:
            u_prompt = user_prompts[key].get('user_prompt', '')
            messages = [
                {"role": "system",
                 "content": "You are a multimodal semantic analyst. Output ONLY valid JSON with no additional text."},
                {"role": "user",
                 "content": semantic_checker_infer_template.format(
                     user_prompt=u_prompt,
                 )},
            ]
            llm_input = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            llm_inputs.append(llm_input)

        outputs = llm.generate(llm_inputs, llm_sampling_params, use_tqdm=False)
        for j, output in enumerate(outputs):
            prompt_id = batch_keys[j]
            raw_text = output.outputs[0].text
            anchors = parse_semantic_anchors(raw_text)
            user_prompts[prompt_id]['semantic_anchors'] = anchors
            user_prompts[prompt_id]['semantic_anchors_raw'] = raw_text
    return user_prompts


# ============================================================
# Stage 3: Cross-Modal Rewriter (F_cr)
# ============================================================

def parse_crr_output(text):
    """Parse 'CRR Video Caption: ...\\nCRR Audio Caption: ...' format."""
    video_caption, audio_caption = "", ""
    v_match = re.search(
        r"CRR\s*Video\s*Caption\s*:\s*(.*?)(?=CRR\s*Audio\s*Caption\s*:|$)",
        text, re.DOTALL | re.IGNORECASE,
    )
    a_match = re.search(
        r"CRR\s*Audio\s*Caption\s*:\s*(.*)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if v_match:
        video_caption = v_match.group(1).strip()
    if a_match:
        audio_caption = a_match.group(1).strip()
    return video_caption, audio_caption


def run_cross_modal_rewriter(samples, batch_size=1, llm=None, tokenizer=None):
    """Stage 3: Generate disentangled, modality-pure (T_V, T_A) caption pairs
    strictly from Semantic Anchors. Shared by training and inference paths."""
    keys_list = list(samples.keys())
    print(f"[CRR] Cross-Modal Rewriter on {len(keys_list)} samples.")
    for i in tqdm(range(0, len(keys_list), batch_size), desc="Cross-Modal Rewriter"):
        batch_keys = keys_list[i:i + batch_size]
        llm_inputs = []
        for key in batch_keys:
            anchors = samples[key].get('semantic_anchors', {"anchors": []})
            anchors_json_str = json.dumps(anchors, ensure_ascii=False, indent=2)
            messages = [
                {"role": "system",
                 "content": "You are an expert caption writer. Strictly follow the modality-purity rules."},
                {"role": "user",
                 "content": cross_modal_rewriter_template.format(
                     semantic_anchors_json=anchors_json_str,
                 )},
            ]
            llm_input = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            llm_inputs.append(llm_input)

        outputs = llm.generate(llm_inputs, llm_sampling_params, use_tqdm=False)
        for j, output in enumerate(outputs):
            sid = batch_keys[j]
            text = output.outputs[0].text
            v_cap, a_cap = parse_crr_output(text)
            samples[sid]['crr_video_caption'] = v_cap
            samples[sid]['crr_audio_caption'] = a_cap
            samples[sid]['crr_rewriter_raw'] = text
    return samples


# ============================================================
# End-to-End Pipelines
# ============================================================

def run_crr_training_pipeline(
    video_js,
    mllm_path,
    audio_llm_path,
    llm_path,
    batch_size=1,
    cache_path=None,
):
    """Full training-stage CRR pipeline (Figure 2(a) of the paper):
        Sounding Video --(MLLM,AudioLLM)--> (T_V^raw, T_A^raw)
                       --F_sc--> Semantic Anchors
                       --F_cr--> (T_V, T_A)
    Models are loaded sequentially and freed between stages to save GPU memory.
    """
    # --- Stage 1a: video raw caption ---
    mllm, processor = load_mllm_model(mllm_path)
    video_js = generate_video_raw_caption(
        video_js, batch_size=batch_size, mllm=mllm, processor=processor
    )
    del mllm, processor
    torch.cuda.empty_cache()
    if cache_path:
        with open(cache_path, 'w') as f:
            json.dump(video_js, f, indent=2, ensure_ascii=False)

    # --- Stage 1b: audio raw caption ---
    audio_llm, audio_processor = load_audio_llm_model(audio_llm_path)
    video_js = generate_audio_raw_caption(
        video_js, batch_size=batch_size, llm=audio_llm
    )
    del audio_llm, audio_processor
    torch.cuda.empty_cache()
    if cache_path:
        with open(cache_path, 'w') as f:
            json.dump(video_js, f, indent=2, ensure_ascii=False)

    # --- Stage 2 + Stage 3: Semantic Checker + Cross-Modal Rewriter ---
    # Both share the same text LLM (Qwen2.5-72B), so we load it once.
    llm, tokenizer = load_llm(llm_path)
    video_js = run_semantic_checker_train(
        video_js, batch_size=batch_size, llm=llm, tokenizer=tokenizer
    )
    video_js = run_cross_modal_rewriter(
        video_js, batch_size=batch_size, llm=llm, tokenizer=tokenizer
    )
    del llm, tokenizer
    torch.cuda.empty_cache()

    return video_js


def run_crr_inference_pipeline(
    user_prompts,
    llm_path,
    batch_size=1,
):
    """Full inference-stage CRR pipeline (Figure 2(b)):
        Concise user prompt --F_sc--> Semantic Anchors --F_cr--> (T_V, T_A).
    Bridges the training-inference distribution gap via prompt expansion.
    """
    llm, tokenizer = load_llm(llm_path)
    user_prompts = run_semantic_checker_infer(
        user_prompts, batch_size=batch_size, llm=llm, tokenizer=tokenizer
    )
    user_prompts = run_cross_modal_rewriter(
        user_prompts, batch_size=batch_size, llm=llm, tokenizer=tokenizer
    )
    del llm, tokenizer
    torch.cuda.empty_cache()
    return user_prompts


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "infer"], default="train",
                        help="train: full pipeline on sounding videos; "
                             "infer: user prompt -> (T_V, T_A).")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--input_json", type=str, default=None,
                        help="Optional pre-existing JSON. "
                             "Train mode: dict keyed by video paths. "
                             "Infer mode: dict keyed by id, each with a 'user_prompt' field "
                             "(or simply id->'user_prompt' string).")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--audio_llm_path", type=str, default=None)
    parser.add_argument("--mllm_path", type=str, default=None)
    parser.add_argument("--llm_path", type=str, default=None)
    parser.add_argument("--output_file", type=str,
                        default="recaption/crr_output.json")
    parser.add_argument("--cache_file", type=str, default=None,
                        help="Optional intermediate cache for raw captions in training mode.")
    args = parser.parse_args()

    args.audio_llm_path = args.audio_llm_path or model_path("Qwen2-Audio-7B-Instruct")
    args.mllm_path = args.mllm_path or model_path("Qwen2.5-VL-72B-Instruct")
    args.llm_path = args.llm_path or model_path("Qwen2.5-72B-Instruct")

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.mode == "train":
        # Build / load the dict of sounding videos
        if args.input_json and os.path.exists(args.input_json):
            with open(args.input_json, 'r') as f:
                video_js = json.load(f)
        else:
            video_files_list = [
                os.path.join(args.video_path, file)
                for file in os.listdir(args.video_path)
                if file.endswith(('.mp4', '.wav', '.mp3', '.mkv', '.mov'))
            ]
            video_js = {f: {} for f in video_files_list}

        video_js = run_crr_training_pipeline(
            video_js=video_js,
            mllm_path=args.mllm_path,
            audio_llm_path=args.audio_llm_path,
            llm_path=args.llm_path,
            batch_size=args.batch_size,
            cache_path=args.cache_file,
        )

        with open(args.output_file, 'w') as f:
            json.dump(video_js, f, indent=4, ensure_ascii=False)
        print(f"[CRR] Training-stage CRR captions saved to {args.output_file}")

    else:  # infer
        if not args.input_json or not os.path.exists(args.input_json):
            raise ValueError(
                "In 'infer' mode, --input_json must point to a JSON of "
                "{id: {'user_prompt': '...'}} or {id: 'user_prompt'}."
            )
        with open(args.input_json, 'r') as f:
            user_prompts = json.load(f)

        # Auto-wrap if values are plain strings
        for k, v in list(user_prompts.items()):
            if isinstance(v, str):
                user_prompts[k] = {'user_prompt': v}

        user_prompts = run_crr_inference_pipeline(
            user_prompts=user_prompts,
            llm_path=args.llm_path,
            batch_size=args.batch_size,
        )

        with open(args.output_file, 'w') as f:
            json.dump(user_prompts, f, indent=4, ensure_ascii=False)
        print(f"[CRR] Inference-stage CRR captions saved to {args.output_file}")