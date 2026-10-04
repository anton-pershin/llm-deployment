"""Deployment CLI: resolves the deployment configuration for a model-hardware
pair and starts the vLLM server (constitution spec sections 3.2, 4.2, 4.3).

Usage:
    python deploy.py model=<model identifier> hardware=<hardware identifier>
"""

import sys
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "config" / "deployment_configurations"


def load_deployment_configuration(model: str, hardware: str) -> dict:
    """Look up the deployment configuration for the model-hardware pair."""
    if not CONFIG_DIR.is_dir():
        raise FileNotFoundError(
            f"deployment configuration library not found: {CONFIG_DIR}"
        )
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        entry = yaml.safe_load(path.read_text())
        if entry.get("model") == model and entry.get("hardware") == hardware:
            return entry
    raise KeyError(f"unsupported pair: (model={model!r}, hardware={hardware!r})")


def build_vllm_command(config: dict) -> list:
    """Build the vLLM server command line from a deployment configuration."""
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        config["model"],
    ]
    for key, value in config["vllm_options"].items():
        if key == "device":
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def main() -> int:
    args = sys.argv[1:]
    kwargs = {}
    for arg in args:
        if "=" not in arg:
            print(
                f"error: malformed argument {arg!r} (expected key=value)",
                file=sys.stderr,
            )
            return 2
        key, value = arg.split("=", 1)
        kwargs[key] = value

    model = kwargs.get("model")
    hardware = kwargs.get("hardware")
    if not model or not hardware:
        print("error: both 'model' and 'hardware' must be specified", file=sys.stderr)
        return 2

    try:
        config = load_deployment_configuration(model, hardware)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    cmd = build_vllm_command(config)
    print(
        f"starting deployment for (model={model}, hardware={hardware}): {' '.join(cmd)}"
    )

    import subprocess

    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
