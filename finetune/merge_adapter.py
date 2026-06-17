"""
merge_adapter.py — Merge QLoRA Adapter into Base Model
========================================================

Merges the LoRA adapter weights back into the base model to produce
a standalone model for inference without PEFT dependency.

Optionally converts to GGUF format for Ollama import.

Usage:
    python -m finetune.merge_adapter
    python -m finetune.merge_adapter --to-gguf
"""

import os
import argparse
import yaml


def load_config() -> dict:
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "llm_config.yaml"
    )
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f).get("finetune", {})
    return {}


def merge_adapter(
    adapter_dir: str = "data/models/qlora_adapter",
    output_dir: str = "data/models/merged",
    base_model: str = None,
):
    """
    Merge LoRA adapter into base model weights.

    Args:
        adapter_dir: Path to the saved LoRA adapter.
        output_dir: Where to save the merged model.
        base_model: Base model ID (loaded from config if None).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    config = load_config()
    if base_model is None:
        base_model = config.get("base_model", "Qwen/Qwen2.5-7B")

    print(f"[Merge] Base model: {base_model}")
    print(f"[Merge] Adapter: {adapter_dir}")
    print(f"[Merge] Output: {output_dir}")

    # Load base model in full precision for merging
    print("[Merge] Loading base model (fp16)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Load and merge adapter
    print("[Merge] Loading and merging LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()

    # Save merged model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"[Merge] Merged model saved to {output_dir}")


def convert_to_gguf(
    model_dir: str = "data/models/merged",
    output_path: str = "data/models/mappo-expert.gguf",
    quantization: str = "q4_k_m",
):
    """
    Convert merged model to GGUF format for Ollama.

    Requires llama.cpp's convert script to be available.
    """
    import subprocess

    print(f"[GGUF] Converting {model_dir} → {output_path}")
    print(f"[GGUF] Quantization: {quantization}")

    # This requires llama.cpp to be installed
    # The convert command varies by version
    cmd = [
        "python", "-m", "llama_cpp.convert",
        model_dir,
        "--outfile", output_path,
        "--outtype", quantization,
    ]

    print(f"[GGUF] Running: {' '.join(cmd)}")
    print("[GGUF] NOTE: This requires llama-cpp-python or llama.cpp installed.")
    print("[GGUF] If this fails, you can manually convert using:")
    print(f"  python convert_hf_to_gguf.py {model_dir} --outtype {quantization}")

    try:
        subprocess.run(cmd, check=True)
        print(f"[GGUF] Converted → {output_path}")

        # Create Ollama Modelfile
        modelfile_path = os.path.join(os.path.dirname(output_path), "Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(f'FROM {output_path}\n')
            f.write('PARAMETER temperature 0.0\n')
            f.write('PARAMETER top_p 1.0\n')
            f.write('SYSTEM "You are an expert RL reward designer specialized in cooperative MARL environments."\n')

        print(f"[GGUF] Ollama Modelfile → {modelfile_path}")
        print(f"[GGUF] To import into Ollama:")
        print(f"  ollama create mappo-expert -f {modelfile_path}")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[GGUF] Conversion failed: {e}")
        print("[GGUF] Manual conversion required. See instructions above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge QLoRA adapter into base model")
    parser.add_argument("--adapter", type=str, default="data/models/qlora_adapter")
    parser.add_argument("--output", type=str, default="data/models/merged")
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--to-gguf", action="store_true",
                        help="Also convert to GGUF for Ollama")
    parser.add_argument("--gguf-quant", type=str, default="q4_k_m")
    args = parser.parse_args()

    merge_adapter(
        adapter_dir=args.adapter,
        output_dir=args.output,
        base_model=args.base_model,
    )

    if args.to_gguf:
        convert_to_gguf(
            model_dir=args.output,
            quantization=args.gguf_quant,
        )
