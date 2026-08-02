from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from llm_scratch_lab.core.components import create_default_registry
from llm_scratch_lab.core.config import load_experiment_config
from llm_scratch_lab.core.registry import ComponentKind
from llm_scratch_lab.tokenization import load_tokenizer_pipeline_config, run_pipeline
from llm_scratch_lab.training import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-lab")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Run a configured training experiment")
    train.add_argument("--config", required=True)
    train.add_argument("--device")
    train.add_argument("--max-updates", type=int)
    train.add_argument("--resume")
    train.add_argument("--tracker", action="append", choices=["tensorboard", "wandb"])

    tokenizer = commands.add_parser("tokenizer", help="Build tokenizer artifacts")
    tokenizer.add_argument("mode", choices=["corpus", "train", "export", "all"])
    tokenizer.add_argument("--config", required=True)

    components = commands.add_parser("components", help="Inspect registered components")
    component_commands = components.add_subparsers(dest="component_command", required=True)
    list_components = component_commands.add_parser("list")
    list_components.add_argument("--kind", choices=["model", "data", "method", "optimizer"])
    describe = component_commands.add_parser("describe")
    describe.add_argument("kind", choices=["model", "data", "method", "optimizer"])
    describe.add_argument("name")
    return parser


def _run_train(args: argparse.Namespace) -> None:
    config, context = load_experiment_config(args.config)
    runtime_updates: dict[str, Any] = {}
    if args.device is not None:
        runtime_updates["device"] = args.device
    if args.max_updates is not None:
        runtime_updates["max_updates"] = args.max_updates
    if runtime_updates:
        config = config.model_copy(
            update={"runtime": config.runtime.model_copy(update=runtime_updates)}
        )
    if args.tracker is not None:
        config = config.model_copy(
            update={"tracking": config.tracking.model_copy(update={"backends": args.tracker})}
        )
    registry = create_default_registry()
    for kind in ("model", "data", "method", "optimizer"):
        registry.validate_config(kind, getattr(config, kind))
    run_dir = run_experiment(config, context, registry, resume=args.resume)
    print(run_dir)


def _run_components(args: argparse.Namespace) -> None:
    registry = create_default_registry()
    if args.component_command == "list":
        kind = args.kind
        entries = registry.list(kind)
        for category, category_entries in entries.items():
            print(f"[{category}]")
            for name, entry in sorted(category_entries.items()):
                print(f"{name}: {entry.description}")
        return
    kind: ComponentKind = args.kind
    entry = registry.get(kind, args.name)
    print(entry.description)
    print(json.dumps(entry.config_type.model_json_schema(), ensure_ascii=False, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.command == "train":
        _run_train(args)
    elif args.command == "tokenizer":
        config, base_dir = load_tokenizer_pipeline_config(args.config)
        output = run_pipeline(args.mode, config, base_dir)
        print(output)
    else:
        _run_components(args)


if __name__ == "__main__":
    main()
