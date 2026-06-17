"""
run_finetune.py — Launch the Full Fine-Tuning Pipeline
=======================================================
Runs dataset generation, formatting, and QLoRA training in sequence.

Usage:
    python scripts/run_finetune.py
    python scripts/run_finetune.py --max-steps 1  # dry run
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Fine-Tuning Pipeline")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip dataset generation (use existing raw data)")
    parser.add_argument("--skip-format", action="store_true",
                        help="Skip dataset formatting (use existing train/val splits)")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Max QLoRA training steps (-1 for full)")
    parser.add_argument("--merge", action="store_true",
                        help="Also merge adapter after training")
    args = parser.parse_args()

    # Step 1: Generate dataset from trajectories
    if not args.skip_generate:
        print("=" * 60)
        print("STEP 1: Generating dataset from trajectories")
        print("=" * 60)
        from finetune.dataset_generator import main as gen_main
        sys.argv = ["dataset_generator",
                     "--input", "data/trajectories",
                     "--output", "data/datasets/raw"]
        gen_main()
    else:
        print("Skipping dataset generation.")

    # Step 2: Format for fine-tuning
    if not args.skip_format:
        print("\n" + "=" * 60)
        print("STEP 2: Formatting dataset for QLoRA")
        print("=" * 60)
        from finetune.format_dataset import main as fmt_main
        sys.argv = ["format_dataset",
                     "--input", "data/datasets/raw/bottleneck_examples.jsonl",
                     "--output", "data/datasets"]
        fmt_main()
    else:
        print("Skipping dataset formatting.")

    # Step 3: QLoRA fine-tuning
    print("\n" + "=" * 60)
    print("STEP 3: QLoRA Fine-Tuning")
    print("=" * 60)
    from finetune.train_qlora import train_qlora
    train_qlora(
        dataset_path="data/datasets/train.jsonl",
        val_path="data/datasets/val.jsonl",
        max_steps=args.max_steps,
    )

    # Step 4: Merge adapter (optional)
    if args.merge:
        print("\n" + "=" * 60)
        print("STEP 4: Merging LoRA Adapter")
        print("=" * 60)
        from finetune.merge_adapter import merge_adapter
        merge_adapter()

    print("\n✅ Fine-tuning pipeline complete!")
