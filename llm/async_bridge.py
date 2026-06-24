"""
async_bridge.py — Async Communication Layer for LLM Inference
==============================================================

Threading-based async bridge between the RL training loop and the LLM
inference engine. Avoids CUDA fork issues by using threads instead of
multiprocessing.

Supports:
    - Non-blocking query submission
    - Blocking query with timeout
    - Model hot-swap (Ollama ↔ HuggingFace)
    - Retry logic with exponential backoff

Usage:
    bridge = LLMBridge(backend="ollama", model_name="qwen2.5:7b")
    result = bridge.query_sync(prompt, timeout=30)
    # or
    bridge.query_async(prompt, callback=on_result)
"""

import json
import time
import threading
import queue
import requests
from typing import Optional, Callable


class LLMBridge:
    """
    Async communication bridge to an LLM inference backend.

    Supports Ollama HTTP API and (future) direct HuggingFace inference.
    """

    def __init__(
        self,
        backend: str = "ollama",
        model_name: str = "qwen2.5:7b",
        host: str = "http://localhost:11434/api/generate",
        timeout: int = 30,
        max_retries: int = 3,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 42,
    ):
        self.backend = backend
        self.model_name = model_name
        self.host = host
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed

        # Async query infrastructure
        self._request_queue: queue.Queue = queue.Queue()
        self._result_store: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

        # HuggingFace model (loaded on demand for hot-swap)
        self._hf_model = None
        self._hf_tokenizer = None
        self._lora_disabled = False

        self._start_worker()

    def _start_worker(self):
        """Start the background worker thread for async queries."""
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="LLMBridge-Worker"
        )
        self._worker_thread.start()

    def _worker_loop(self):
        """Background worker that processes queued LLM requests."""
        while self._running:
            try:
                request = self._request_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            query_id = request["id"]
            prompts = request["prompts"]
            callback = request.get("callback")
            require_json = request.get("require_json", False)

            try:
                # If backend is huggingface, we can batch natively
                if len(prompts) > 1 and (self.backend == "huggingface" or self.backend == "huggingface_peft"):
                    results = self._query_huggingface_batch(prompts)
                else:
                    results = [self._execute_query(p, require_json=require_json) for p in prompts]
                
                is_batch = request.get("is_batch", False)
                if callback:
                    callback(results if is_batch else results[0])
            except Exception as e:
                print(f"[LLMBridge] Async query failed: {e}")
                if callback:
                    callback(None)
            finally:
                self._request_queue.task_done()

    def query_async(
        self,
        prompt: str,
        callback: Optional[Callable[[Optional[str]], None]] = None,
        require_json: bool = False
    ) -> str:
        """
        Asynchronous query. Returns immediately with a query_id.
        """
        query_id = f"query_{id(prompt)}_{self.seed}"
        self._request_queue.put({
            "id": query_id,
            "prompts": [prompt],
            "callback": callback,
            "require_json": require_json,
            "is_batch": False
        })
        return query_id

    def query_batch_async(self, prompts: list, callback: Callable, require_json: bool = False):
        """Submits a batch of prompts to the background worker."""
        query_id = f"batch_{id(prompts)}_{self.seed}"
        self._request_queue.put({
            "id": query_id,
            "prompts": prompts,
            "callback": callback,
            "require_json": require_json,
            "is_batch": True
        })

    def _execute_query(self, prompt: str, require_json: bool = False) -> str:
        """
        Execute a single LLM query with retries.

        Returns:
            Raw text response from the LLM.
        """
        if self.backend == "ollama":
            return self._query_ollama(prompt, require_json=require_json)
        elif self.backend == "huggingface_peft" or self.backend == "huggingface":
            return self._query_huggingface(prompt)
        elif self.backend == "gemini":
            return self._query_gemini(prompt)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _execute_batch(self, prompts: list) -> list:
        if self.backend == "huggingface_peft" or self.backend == "huggingface":
            return self._query_huggingface_batch(prompts)
        else:
            # Fallback for Ollama/Gemini (sequential)
            return [self._execute_query(p) for p in prompts]

    def _query_huggingface_batch(self, prompts: list) -> list:
        if self._hf_model is None or self._hf_tokenizer is None:
            raise RuntimeError("HuggingFace model not loaded. Call swap_model() first.")
            
        formatted_prompts = []
        for prompt in prompts:
            if hasattr(self._hf_tokenizer, "apply_chat_template"):
                if "### CURRENT STATE:" in prompt:
                    parts = prompt.split("### CURRENT STATE:")
                    system_content = parts[0].strip()
                    user_content = "### CURRENT STATE:\n" + parts[1].strip()
                else:
                    system_content = "You are a helpful assistant orchestrating RL agents."
                    user_content = prompt.strip()
                    
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
                formatted_prompts.append(self._hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            else:
                formatted_prompts.append(prompt)

        # We can just batch the inputs
        self._hf_tokenizer.padding_side = 'left' 
        if self._hf_tokenizer.pad_token is None:
            self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token
            
        inputs = self._hf_tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(self._hf_model.device)
        
        if getattr(self, "_lora_disabled", False) and hasattr(self._hf_model, "disable_adapter"):
            with self._hf_model.disable_adapter():
                outputs = self._hf_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=1.0,
                    top_p=1.0,
                    do_sample=False
                )
        else:
            outputs = self._hf_model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=1.0,
                top_p=1.0,
                do_sample=False
            )
        responses = []
        for i, output in enumerate(outputs):
            response = self._hf_tokenizer.decode(output[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            responses.append(response)
        return responses

    def _query_gemini(self, prompt: str) -> str:
        """Query Google Gemini API."""
        import os
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai is not installed. Please run: pip install google-genai"
            )

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        client = genai.Client(api_key=api_key)
        
        # Use gemini-2.5-flash as default if model name is empty
        model_id = self.model_name if self.model_name else "gemini-2.5-flash"
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                )
                return response.text
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt
                print(f"[LLMBridge] Gemini attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        raise RuntimeError(
            f"Gemini query failed after {self.max_retries} retries: {last_error}"
        )

    def _query_ollama(self, prompt: str, require_json: bool = False) -> str:
        """Query Ollama HTTP API with retry logic."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": self.seed,
            },
        }

        if require_json:
            payload["format"] = "json"

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.host, json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()["response"]
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"[LLMBridge] Ollama attempt {attempt + 1} failed: {e}. "
                      f"Retrying in {wait_time}s...")
                time.sleep(wait_time)

        raise RuntimeError(
            f"LLM query failed after {self.max_retries} retries: {last_error}"
        )

    def _query_huggingface(self, prompt: str) -> str:
        """Query a locally loaded HuggingFace model."""
        if self._hf_model is None or self._hf_tokenizer is None:
            raise RuntimeError(
                "HuggingFace model not loaded. Call swap_model() first."
            )

        if hasattr(self._hf_tokenizer, "apply_chat_template"):
            if "### CURRENT STATE:" in prompt:
                parts = prompt.split("### CURRENT STATE:")
                system_content = parts[0].strip()
                user_content = "### CURRENT STATE:\n" + parts[1].strip()
            else:
                system_content = "You are a helpful assistant orchestrating RL agents."
                user_content = prompt.strip()
                
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            prompt = self._hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self._hf_tokenizer(prompt, return_tensors="pt").to(
            self._hf_model.device
        )
        
        if getattr(self, "_lora_disabled", False) and hasattr(self._hf_model, "disable_adapter"):
            with self._hf_model.disable_adapter():
                outputs = self._hf_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=max(self.temperature, 0.01),
                    top_p=self.top_p,
                    do_sample=(self.temperature > 0),
                )
        else:
            outputs = self._hf_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=max(self.temperature, 0.01),  # avoid div by zero
                top_p=self.top_p,
                do_sample=(self.temperature > 0),
            )
        response = self._hf_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return response

    def query_sync(self, prompt: str, timeout: Optional[int] = None, require_json: bool = False) -> str:
        """
        Synchronous (blocking) query. Blocks until the LLM responds.

        Args:
            prompt: The prompt to send.
            timeout: Optional override for timeout.
            require_json: Whether to enforce JSON formatting via logits processor.
        """
        if timeout:
            self.timeout = timeout
        return self._execute_query(prompt, require_json=require_json)



    def get_async_result(self, query_id: str, timeout: float = 0.0) -> Optional[dict]:
        """
        Check if an async query has completed.

        Args:
            query_id: The ID returned by query_async().
            timeout: How long to wait (0 = non-blocking check).

        Returns:
            Result dict with 'success' and 'response'/'error' keys, or None if not ready.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if query_id in self._result_store:
                    return self._result_store.pop(query_id)
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    def swap_model(self, model_path: str, backend: str = "huggingface"):
        """
        Hot-swap the LLM model. Used to switch from generic Ollama to fine-tuned model.

        Args:
            model_path: Path to HuggingFace model or Ollama model name.
            backend: "ollama", "huggingface", or "huggingface_peft".
        """
        print(f"[LLMBridge] Swapping model to: {model_path} (backend={backend})")

        if backend == "huggingface":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.backend = "huggingface"
            print(f"[LLMBridge] HuggingFace model loaded: {model_path}")
            
        elif backend == "huggingface_peft":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel
            
            base_model = "Qwen/Qwen2.5-7B-Instruct"
            print(f"[LLMBridge] Loading base model {base_model} in 4-bit...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self._hf_tokenizer = AutoTokenizer.from_pretrained(base_model)
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
            )
            print(f"[LLMBridge] Loading LoRA adapter from {model_path}...")
            self._hf_model = PeftModel.from_pretrained(base, model_path)
            self.backend = "huggingface"

        elif backend == "ollama":
            self.model_name = model_path
            self.backend = "ollama"
            self._hf_model = None
            self._hf_tokenizer = None
            print(f"[LLMBridge] Switched to Ollama model: {model_path}")

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def disable_lora(self):
        """Dynamically disable the LoRA adapter to run base model inference."""
        self._lora_disabled = True
        print("[LLMBridge] LoRA adapter disabled. Using base model.")

    def enable_lora(self):
        """Dynamically enable the LoRA adapter."""
        self._lora_disabled = False
        print("[LLMBridge] LoRA adapter enabled.")

    def close(self):
        """Stop the worker thread."""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)


if __name__ == "__main__":
    # Quick test — requires Ollama running
    bridge = LLMBridge()
    try:
        result = bridge.query_sync("Say hello in JSON format: {\"greeting\": \"...\"}")
        print(f"Response: {result}")
    except Exception as e:
        print(f"Test failed (Ollama not running?): {e}")
    finally:
        bridge.close()
