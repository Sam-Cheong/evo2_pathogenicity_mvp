import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evo2_mvp.environment import inspect_environment
from evo2_mvp.utils import save_json, load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/environment.json")
    p.add_argument("--check-network", action="store_true")
    p.add_argument("--require-evo2", action="store_true")
    p.add_argument("--config", default="configs/mvp.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)["encoder"]
    backend = cfg.get("backend", "nvidia_api")
    report = inspect_environment(args.check_network, backend=backend, model=cfg.get("model"))
    save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if args.require_evo2 and not report["ready_for_selected_backend_attempt"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

