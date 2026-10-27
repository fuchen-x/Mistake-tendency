from __future__ import annotations

import argparse
import json

from .config import ensure_output_dirs, load_config
from .generate_profiles import generate_profiles
from .predict_errors import predict_errors
from .preprocess import preprocess_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mistake-tendency stage-1 experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Build clean temporal splits")
    preprocess.add_argument("--config", required=True)

    profiles = subparsers.add_parser("profiles", help="Infer student profiles")
    profiles.add_argument("--config", required=True)
    profiles.add_argument("--provider")
    profiles.add_argument("--limit", type=int)
    profiles.add_argument("--overwrite", action="store_true")

    predict = subparsers.add_parser("predict", help="Predict held-out wrong options")
    predict.add_argument("--config", required=True)
    predict.add_argument("--provider")
    predict.add_argument("--condition", required=True)
    predict.add_argument("--split", choices=["validation", "test"], default="test")
    predict.add_argument("--profiles")
    predict.add_argument("--limit-students", type=int)
    predict.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    if args.command == "preprocess":
        manifest = preprocess_dataset(config)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    elif args.command == "profiles":
        path = generate_profiles(config, args.provider, args.limit, args.overwrite)
        print(path)
    elif args.command == "predict":
        path = predict_errors(
            config,
            condition=args.condition,
            split=args.split,
            provider=args.provider,
            profiles_path=args.profiles,
            limit_students=args.limit_students,
            overwrite=args.overwrite,
        )
        print(path)


if __name__ == "__main__":
    main()
