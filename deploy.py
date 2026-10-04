"""Deployment CLI: resolves the deployment configuration for a model-hardware
pair and starts the vLLM server (constitution spec sections 3.2, 4.2, 4.3).

Usage:
    python deploy.py model=<model identifier> hardware=<hardware identifier>
"""

import os
import subprocess
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
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def build_vllm_env(config: dict) -> dict:
    """Build the environment for the vLLM subprocess from the configuration."""
    env = {}
    for key, value in config.get("environment", {}).items():
        env[key] = resolve_env_values(value)
    return env


def resolve_env_values(value):
    """Resolve supported placeholders in an environment variable value.

    Supports lists (joined with ':') and the '{sys_prefix}' placeholder
    (resolved to the current Python environment's prefix).
    """
    if isinstance(value, list):
        return ":".join(resolve_env_values(item) for item in value)
    if isinstance(value, str):
        return value.replace("{sys_prefix}", sys.prefix)
    return str(value)


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
    env = build_vllm_env(config)
    print(
        f"starting deployment for (model={model}, hardware={hardware}): {' '.join(cmd)}"
    )
    if env:
        print(f"with environment: {env}")

    result = subprocess.run(cmd, check=False, env={**os.environ, **env})
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
