import argparse
import json
import os
import time

import torch

from workers.bsroformer_worker import load_model_impl, separate_impl, load_models_registry


def parse_args():
    parser = argparse.ArgumentParser(description="Run BS-RoFormer models on one input file")
    parser.add_argument("--input", required=True, help="Input audio file path")
    parser.add_argument("--models-dir", required=True, help="Models directory (contains models.json)")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--overlap", type=int, default=2, help="Overlap factor")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--use-tta", action="store_true", help="Enable TTA")
    parser.add_argument("--fast", action="store_true", help="Use vectorized chunking")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of models (0 = all)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "report.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models_registry(args.models_dir)

    results = []
    model_names = list(models.keys())
    if args.limit and args.limit > 0:
        model_names = model_names[:args.limit]

    for name in model_names:
        start = time.time()
        status = "ok"
        error = None
        try:
            model, config, model_info = load_model_impl(name, args.models_dir)
            separate_impl(
                model,
                config,
                args.input,
                os.path.join(args.output_dir, name),
                model_info,
                device=device,
                overlap=args.overlap,
                batch_size=args.batch_size,
                use_tta=args.use_tta,
                output_format="wav",
                pcm_type="FLOAT",
                extract_instrumental=False,
                selected_stems=None,
                two_stems=None,
                use_fast=args.fast,
            )
        except Exception as exc:
            status = "error"
            error = str(exc)
        elapsed = round(time.time() - start, 2)
        results.append(
            {
                "model": name,
                "status": status,
                "elapsed_s": elapsed,
                "error": error,
            }
        )

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
