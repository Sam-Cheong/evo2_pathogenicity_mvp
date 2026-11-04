"""Execute and checkpoint a Jupyter notebook with this .venv's real kernel."""
from __future__ import annotations

import argparse
import faulthandler
from datetime import datetime, timezone
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="HyenaDNA_d256_experiment.ipynb")
    parser.add_argument("--output", default="HyenaDNA_d256_experiment.executed.ipynb")
    parser.add_argument("--timeout", type=int, default=7200, help="Per-cell timeout in seconds")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    for variable, folder in {"JUPYTER_RUNTIME_DIR": "jupyter_runtime", "JUPYTER_CONFIG_DIR": "jupyter_config",
                             "JUPYTER_DATA_DIR": "jupyter_data", "IPYTHONDIR": "ipython",
                             "HF_HOME": "huggingface"}.items():
        path = root / "work" / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    # Windows temporary/runtime files stay in the writable project as well.
    temporary = root / "work" / "notebook_tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(temporary)
    os.environ["PYTHONUTF8"] = "1"
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from evo2_mvp.utils import save_json

    notebook = nbformat.read(root / args.notebook, as_version=4)
    target = root / args.output
    status = root / "outputs" / "hyenadna_notebook_execution.json"
    started = datetime.now(timezone.utc).isoformat()
    state = {"status": "starting", "started_at_utc": started, "notebook": str(root / args.notebook),
             "executed_notebook": str(target), "kernel_python": sys.executable, "successful_cells": []}

    def persist():
        temporary_notebook = target.with_suffix(target.suffix + ".tmp")
        nbformat.write(notebook, temporary_notebook)
        os.replace(temporary_notebook, target)
        save_json(status, state)

    def on_start(cell, cell_index, **kwargs):
        state.update(status="running", current_cell=cell_index, current_cell_id=cell.get("id"))
        print(f"CELL {cell_index} START {cell.get('id', '')}", flush=True)
        persist()

    def on_executed(cell, cell_index, **kwargs):
        errors = [o for o in cell.get("outputs", []) if o.get("output_type") == "error"]
        if errors:
            state.update(status="failed", error=errors[-1].get("ename", "CellError") + ": " + errors[-1].get("evalue", ""))
        else:
            state["successful_cells"].append(cell_index)
        print(f"CELL {cell_index} {'FAILED' if errors else 'DONE'}", flush=True)
        persist()

    manager = KernelManager(kernel_name="python3", ip="127.0.0.1",
                            connection_file=str(root / "work/jupyter_runtime" / f"kernel-{os.getpid()}.json"))
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=args.timeout, allow_errors=False,
                            resources={"metadata": {"path": str(root)}},
                            on_cell_start=on_start, on_cell_executed=on_executed)
    persist()
    stack_log = (root / "work/notebook_kernel_stack.log").open("w", encoding="utf-8")
    faulthandler.dump_traceback_later(45, repeat=True, file=stack_log)
    try:
        client.execute(cleanup_kc=True)
        state.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat())
        persist()
        print(f"NOTEBOOK COMPLETED: {target}", flush=True)
    except BaseException as exc:
        state.update(status="failed", failed_at_utc=datetime.now(timezone.utc).isoformat(),
                     error=type(exc).__name__ + ": " + str(exc))
        persist()
        print(f"NOTEBOOK FAILED. Partial outputs saved to {target}", file=sys.stderr, flush=True)
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        stack_log.close()
        # With an externally supplied manager, nbclient leaves cleanup to us.
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)


if __name__ == "__main__":
    main()
