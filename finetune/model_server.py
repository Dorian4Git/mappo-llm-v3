"""
model_server.py — Serve Fine-Tuned Model for Inference
=======================================================

Two inference modes:
    A) Ollama mode: Register GGUF model, query via Ollama API
    B) Direct HuggingFace mode: Load merged model, infer locally

Provides a unified generate() API identical to LLMBridge.

Usage:
    server = ModelServer(mode="huggingface", model_path="data/models/merged")
    response = server.generate(prompt)
"""

import json
import os
from typing import Optional


class ModelServer:
    """
    Unified inference server for fine-tuned models.

    Supports both Ollama and direct HuggingFace inference with the
    same API as LLMBridge for seamless hot-swap.
    """

    def __init__(
        self,
        mode: str = "huggingface",
        model_path: str = "data/models/merged",
        ollama_model_name: str = "mappo-expert",
        ollama_host: str = "http://localhost:11434/api/generate",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ):
        self.mode = mode
        self.model_path = model_path
        self.ollama_model_name = ollama_model_name
        self.ollama_host = ollama_host
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._model = None
        self._tokenizer = None
        self._device = None

        if mode == "huggingface":
            self._load_hf_model()

    def _load_hf_model(self):
        """Load the merged HuggingFace model for direct inference."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[ModelServer] Loading HF model from {self.model_path}...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        self._device = next(self._model.parameters()).device

        print(f"[ModelServer] Model loaded on {self._device}")

    def generate(self, prompt: str) -> str:
        """
        Generate a response from the fine-tuned model.

        Args:
            prompt: Input prompt.

        Returns:
            Generated text (expected to be JSON).
        """
        if self.mode == "huggingface":
            return self._generate_hf(prompt)
        elif self.mode == "ollama":
            return self._generate_ollama(prompt)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _generate_hf(self, prompt: str) -> str:
        """Generate using direct HuggingFace inference."""
        import torch

        # Format with the same template used during fine-tuning
        formatted = f"### Input:\n{prompt}\n\n### Response:\n"

        inputs = self._tokenizer(formatted, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=max(self.temperature, 0.01),
                do_sample=(self.temperature > 0),
                top_p=1.0,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only the generated part
        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        return response

    def _generate_ollama(self, prompt: str) -> str:
        """Generate using Ollama API."""
        import requests

        payload = {
            "model": self.ollama_model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "top_p": 1.0,
            },
        }

        response = requests.post(self.ollama_host, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["response"]

    def generate_parsed(self, prompt: str) -> Optional[dict]:
        """
        Generate and parse as JSON.

        Returns parsed dict or None if parsing fails.
        """
        try:
            raw = self.generate(prompt)
            clean = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(clean)
        except (json.JSONDecodeError, Exception) as e:
            print(f"[ModelServer] Failed to parse response: {e}")
            return None

    def benchmark(self, prompts: list[str], n_runs: int = 3) -> dict:
        """
        Benchmark inference latency.

        Returns dict with avg_latency, p50, p95, p99 in seconds.
        """
        import time
        latencies = []

        for prompt in prompts:
            for _ in range(n_runs):
                start = time.perf_counter()
                self.generate(prompt)
                latencies.append(time.perf_counter() - start)

        import numpy as np
        latencies = np.array(latencies)

        return {
            "n_queries": len(latencies),
            "avg_latency_s": float(latencies.mean()),
            "p50_s": float(np.percentile(latencies, 50)),
            "p95_s": float(np.percentile(latencies, 95)),
            "p99_s": float(np.percentile(latencies, 99)),
            "total_s": float(latencies.sum()),
        }


if __name__ == "__main__":
    print("[ModelServer] Testing...")
    print("  To test HF mode: python -m finetune.model_server --mode huggingface")
    print("  To test Ollama:   python -m finetune.model_server --mode ollama")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="ollama")
    parser.add_argument("--model-path", default="data/models/merged")
    args = parser.parse_args()

    server = ModelServer(mode=args.mode, model_path=args.model_path)
    result = server.generate_parsed(
        "ENVIRONMENT STATE: Agents stuck for 100 steps. "
        "Bottleneck: pickaxe crafting. "
        "What reward shaping adjustment would resolve this?"
    )
    print(f"Response: {result}")
