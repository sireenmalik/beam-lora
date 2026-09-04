"""
serve.py
--------
OpenAI-compatible /v1/chat/completions endpoint backed by Qwen 2.5 0.5B + your
trained LoRA adapter (./beam-lora/).

Drop-in for beammeup's src/model.js when MODEL_PROVIDER=openai.

Run:
  uvicorn serve:app --host 127.0.0.1 --port 8000
"""

import time
import uuid
import torch
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./beam-lora"
MAX_NEW_TOKENS_DEFAULT = 150

print("loading tokenizer...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("loading base model (Qwen 2.5 0.5B)...")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32).to("cpu")

print(f"loading LoRA adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base, ADAPTER_PATH).to("cpu")
model.eval()
print("READY")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = MAX_NEW_TOKENS_DEFAULT
    top_p: float | None = None
    n: int | None = 1
    stream: bool | None = False
    stop: Any | None = None
    chat_template_kwargs: dict | None = None

    class Config:
        extra = "allow"


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


app = FastAPI(title="beammeup-edge LLM", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "base_model": BASE_MODEL, "adapter": ADAPTER_PATH}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "beam-lora", "object": "model", "created": int(time.time()), "owned_by": "local"}],
    }


def _run_inference(messages, max_new_tokens):
    msgs = [{"role": m.role, "content": m.content} for m in messages]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")
    prompt_tokens = int(ids["input_ids"].shape[1])
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    completion_tokens = int(out.shape[1] - prompt_tokens)
    text = tok.decode(out[0][prompt_tokens:], skip_special_tokens=True).strip()
    return text, prompt_tokens, completion_tokens


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail="streaming not implemented")
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    text, ptok, ctok = _run_inference(req.messages, req.max_tokens or MAX_NEW_TOKENS_DEFAULT)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=req.model,
        choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=text), finish_reason="stop")],
        usage=Usage(prompt_tokens=ptok, completion_tokens=ctok, total_tokens=ptok + ctok),
    )


@app.post("/chat/completions", response_model=ChatCompletionResponse)
def chat_completions_no_v1(req: ChatCompletionRequest):
    return chat_completions(req)
