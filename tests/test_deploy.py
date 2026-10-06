"""Tests for the baseline skeleton KISS spec (01-baseline-skeleton-kiss-spec)."""

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_CONFIG_DIR = REPO_ROOT / "config" / "deployment_configurations"

SUPPORTED_MODEL = "Qwen/Qwen3-0.6B"
SUPPORTED_HARDWARE = "huawei-cpu"

UNSUPPORTED_MODEL = "org/does-not-exist"
UNSUPPORTED_HARDWARE = "not-a-hardware"


def _run_deploy(*args: str) -> subprocess.CompletedProcess:
    """Run deploy.py with the given Hydra CLI args."""
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "deploy.py"), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


# --- T2: unsupported pair must fail clearly and not start a server (R4, B1, B2) ---


def test_t2_unsupported_model(capsys):
    result = _run_deploy(f"model={UNSUPPORTED_MODEL}", f"hardware={SUPPORTED_HARDWARE}")
    assert result.returncode != 0
    assert UNSUPPORTED_MODEL in (result.stderr + result.stdout)
    assert "unsupported" in (result.stderr + result.stdout).lower()


def test_t2_unsupported_hardware(capsys):
    result = _run_deploy(f"model={SUPPORTED_MODEL}", f"hardware={UNSUPPORTED_HARDWARE}")
    assert result.returncode != 0
    assert UNSUPPORTED_HARDWARE in (result.stderr + result.stdout)
    assert "unsupported" in (result.stderr + result.stdout).lower()


# --- T3: deployment-configuration library shape (R2) ---


def test_t3_config_directory_non_empty():
    files = list(DEPLOY_CONFIG_DIR.glob("*.yaml"))
    assert files, f"no deployment configurations in {DEPLOY_CONFIG_DIR}"


def test_t3_every_entry_defines_identifiers_and_options():
    for path in DEPLOY_CONFIG_DIR.glob("*.yaml"):
        entry = yaml.safe_load(path.read_text())
        assert isinstance(entry.get("model"), str) and entry["model"], path
        assert isinstance(entry.get("hardware"), str) and entry["hardware"], path
        assert "vllm_options" in entry, path


def test_t3_supported_pair_entry_exists():
    entries = [yaml.safe_load(p.read_text()) for p in DEPLOY_CONFIG_DIR.glob("*.yaml")]
    matching = [
        e
        for e in entries
        if e.get("model") == SUPPORTED_MODEL and e.get("hardware") == SUPPORTED_HARDWARE
    ]
    assert matching, (
        f"no deployment configuration for ({SUPPORTED_MODEL}, {SUPPORTED_HARDWARE})"
    )


# --- Spec 02: the (Qwen/Qwen3-0.6B, rtx3090-oct-22) pair (T1, T2) ---

RTX3090_MODEL = "Qwen/Qwen3-0.6B"
RTX3090_HARDWARE = "rtx3090-oct-22"
RTX3090_SLUG = "qwen_qwen3_0_6b_rtx3090_oct_22"


def _pair_slug(model: str, hardware: str) -> str:
    """Pair slug derived per the constitution spec section 4.3 rule."""
    import re

    return re.sub(r"[^a-z0-9]", "_", f"{model}_{hardware}".lower())


def test_t1_slug_rule_reproduces_documented_example():
    assert _pair_slug("Qwen/Qwen3-0.6B", "huawei-cpu") == "qwen_qwen3_0_6b_huawei_cpu"


def test_t1_new_pair_entry_exists_under_the_slug_name():
    path = DEPLOY_CONFIG_DIR / f"{_pair_slug(RTX3090_MODEL, RTX3090_HARDWARE)}.yaml"
    assert path.is_file(), f"missing deployment configuration {path.name}"
    entry = yaml.safe_load(path.read_text())
    assert entry["model"] == RTX3090_MODEL
    assert entry["hardware"] == RTX3090_HARDWARE


def test_t1_new_pair_is_the_baseline_strategy():
    path = DEPLOY_CONFIG_DIR / f"{_pair_slug(RTX3090_MODEL, RTX3090_HARDWARE)}.yaml"
    entry = yaml.safe_load(path.read_text())
    # R2: the baseline deployment strategy carries no extra deployment options.
    assert entry["vllm_options"] == {}


def test_t1_new_pair_setup_script_exists():
    path = DEPLOY_CONFIG_DIR / f"{_pair_slug(RTX3090_MODEL, RTX3090_HARDWARE)}.sh"
    assert path.is_file(), f"missing environment setup script {path.name}"


def test_t2_unsupported_near_miss_hardware():
    result = _run_deploy(f"model={RTX3090_MODEL}", "hardware=rtx3090-24gb")
    assert result.returncode != 0
    assert "rtx3090-24gb" in (result.stderr + result.stdout)
    assert "unsupported" in (result.stderr + result.stdout).lower()


def test_t2_unsupported_model_on_the_new_hardware():
    result = _run_deploy(f"model={UNSUPPORTED_MODEL}", f"hardware={RTX3090_HARDWARE}")
    assert result.returncode != 0
    assert "unsupported" in (result.stderr + result.stdout).lower()


def test_t2_cpu_entry_is_untouched():
    path = DEPLOY_CONFIG_DIR / f"{_pair_slug(SUPPORTED_MODEL, SUPPORTED_HARDWARE)}.yaml"
    entry = yaml.safe_load(path.read_text())
    assert entry["model"] == SUPPORTED_MODEL
    assert entry["hardware"] == SUPPORTED_HARDWARE
    assert entry["vllm_options"], path
