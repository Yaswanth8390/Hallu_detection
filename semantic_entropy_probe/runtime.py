"""Model loading, prompt formatting, generation, and dataset access for SEP."""

from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class GenerationResult:
    prompt_ids: torch.Tensor
    generated_ids: torch.Tensor
    generated_text: str


def input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def load_model(device: str = "cuda", load_in_8bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    quantization_config = (
        BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None
    )
    device_map = "auto" if device == "auto" or load_in_8bit else None
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device_map,
        quantization_config=quantization_config,
    )
    if device_map is None:
        model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def encode_question(tokenizer, question: str) -> torch.Tensor:
    messages = [
        {
            "role": "system",
            "content": "Answer the question concisely and factually, in one short sentence.",
        },
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(prompt, return_tensors="pt").input_ids[0]


@torch.no_grad()
def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 48,
                    do_sample: bool = False, temperature: float = 1.0,
                    top_p: float = 1.0) -> GenerationResult:
    prompt_ids = encode_question(tokenizer, question)
    input_ids = prompt_ids.to(input_device(model)).unsqueeze(0)
    generation_options = {}
    if do_sample:
        generation_options.update(temperature=temperature, top_p=top_p)
    output = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        **generation_options,
    )[0]
    generated_ids = output[input_ids.shape[1]:].detach().cpu()
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    if not generated_text or generated_ids.numel() == 0:
        raise RuntimeError("Model generated an empty answer")
    return GenerationResult(prompt_ids.cpu(), generated_ids, generated_text)


def load_questions(limit: int, split: str = "validation") -> list[str]:
    dataset = load_dataset("truthful_qa", "generation", split=split)
    questions = [row["question"] for row in dataset]
    return questions[:limit]


def sequence_log_probability(model, prompt_ids: torch.Tensor,
                             generated_ids: torch.Tensor) -> float:
    """Sum realized token log-probabilities under the original model."""
    if generated_ids.numel() == 0:
        raise ValueError("Cannot score an empty generated answer")
    device = input_device(model)
    prompt_ids = prompt_ids.reshape(-1).to(device)
    generated_ids = generated_ids.reshape(-1).to(device)
    sequence = torch.cat((prompt_ids, generated_ids)).unsqueeze(0)
    positions = torch.arange(
        prompt_ids.numel() - 1,
        prompt_ids.numel() + generated_ids.numel() - 1,
        device=device,
    )
    with torch.no_grad():
        logits = model(input_ids=sequence, use_cache=False).logits[0].float()
        log_probs = logits[positions].log_softmax(dim=-1)
        return float(log_probs.gather(1, generated_ids.unsqueeze(1)).sum().item())
