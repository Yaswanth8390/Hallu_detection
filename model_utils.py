"""
Model loading + generation utilities for the Jacobian hallucination-detection pipeline.

Target model: Qwen/Qwen2.5-7B-Instruct
Requires a GPU with ~16-20GB VRAM for fp16/bf16 inference (or load in 8-bit/4-bit on smaller GPUs).
"""

from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class GenerationResult:
    prompt: str
    prompt_ids: torch.Tensor          # (prompt_len,)
    generated_ids: torch.Tensor       # (gen_len,) -- newly generated tokens only
    generated_text: str
    full_ids: torch.Tensor            # (prompt_len + gen_len,)


def load_model(dtype: torch.dtype = torch.bfloat16, device: str = "cuda", load_in_8bit: bool = True):
    """Load Qwen2.5-7B-Instruct and its tokenizer.

    load_in_8bit=True (default) quantizes weights to ~7-8GB via bitsandbytes --
    needed on a single 15GB T4, since bf16 weights alone (~14GB) leave almost
    no headroom for the forward+backward passes the Jacobian/grounding code
    needs. Set False if you have a bigger GPU (A100/L4/etc.) and want full
    precision -- Jacobian magnitudes are somewhat sensitive to quantization,
    so prefer False when you have the VRAM for it.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    quant_config = BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None
    # 8-bit weights need accelerate to place them -- always use device_map="auto"
    # in that case (it'll put everything on GPU 0 if that's all that's needed).
    device_map = "auto" if (device == "auto" or load_in_8bit) else None
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device_map,
        quantization_config=quant_config,
    )
    if device_map is None:
        model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_answer(model, tokenizer, question: str, device: str = "cuda",
                     max_new_tokens: int = 32) -> GenerationResult:
    """Greedy-decode a short answer to `question` using the chat template.

    Greedy decoding is used deliberately: we want a single, reproducible
    generation path to attribute (sampling would make the Jacobian analysis
    non-deterministic across runs).
    """
    messages = [
        {"role": "system", "content": "Answer the question concisely and factually, in one short sentence."},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    out = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0, prompt_ids.shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return GenerationResult(
        prompt=prompt,
        prompt_ids=prompt_ids[0].detach().cpu(),
        generated_ids=gen_ids.detach().cpu(),
        generated_text=gen_text,
        full_ids=out[0].detach().cpu(),
    )


def token_strings(tokenizer, ids: torch.Tensor) -> List[str]:
    return [tokenizer.decode([t]) for t in ids.tolist()]
