"""Tests for the baseline skeleton KISS spec (01-baseline-skeleton-kiss-spec)."""

import re
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


# --- T1/T2 of spec 03: the rtx3090-oct-22 deployment configurations (R1, R2, R3, R7, B1, B2) ---

RTX3090_HARDWARE = "rtx3090-oct-22"
RTX3090_8B_MODEL = "Qwen/Qwen3-8B"
RTX3090_8B_WEIGHTS_REPO = "JunHowie/Qwen3-8B-GPTQ-Int8"

NEAR_MISS_HARDWARE = "rtx3090-24gb"
UNSUPPORTED_QUANTIZED_MODEL = "Qwen/Qwen3-8B-FP8"


def _pair_slug(model: str, hardware: str) -> str:
    """The pair-slug rule of the constitution spec, section 4.3."""
    return re.sub(r"[^a-z0-9]", "_", f"{model}_{hardware}".lower())


def _entry_path(model: str, hardware: str) -> Path:
    """The library file the pair-slug rule requires for this pair."""
    return DEPLOY_CONFIG_DIR / f"{_pair_slug(model, hardware)}.yaml"


def _entry(model: str, hardware: str) -> dict:
    path = _entry_path(model, hardware)
    assert path.is_file(), f"missing library entry {path.name}"
    entry = yaml.safe_load(path.read_text())
    assert entry.get("model") == model, path
    assert entry.get("hardware") == hardware, path
    return entry


def test_t1_rtx3090_8b_entry():
    entry = _entry(RTX3090_8B_MODEL, RTX3090_HARDWARE)
    assert entry["vllm_options"] == {
        "max_model_len": 24576,
        "gpu_memory_utilization": 0.6,
        "dtype": "float16",
        "linear_backend": "marlin",
        "max_num_seqs": 8,
        "compilation_config": '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8],"compile_sizes":[1,2,3,4,5,6,7,8]}',
        "override_generation_config": '{"temperature":0.0}',
    }, "the entry records explicit small-batch graph and generation settings"
    setup_script = _entry_path(RTX3090_8B_MODEL, RTX3090_HARDWARE).with_suffix(".sh")
    assert setup_script.is_file()
    assert RTX3090_8B_WEIGHTS_REPO in setup_script.read_text(), (
        "the setup script prepares the 4-bit release this deployment serves"
    )


def test_t2_near_miss_hardware_identifier_rejected():
    result = _run_deploy(f"model={RTX3090_8B_MODEL}", f"hardware={NEAR_MISS_HARDWARE}")
    assert result.returncode != 0
    assert "unsupported" in (result.stderr + result.stdout).lower()


def test_t2_unsupported_model_identifier_rejected():
    for model in (UNSUPPORTED_QUANTIZED_MODEL, RTX3090_8B_WEIGHTS_REPO):
        result = _run_deploy(f"model={model}", f"hardware={RTX3090_HARDWARE}")
        assert result.returncode != 0, model
        assert "unsupported" in (result.stderr + result.stdout).lower(), model


def test_t2_cpu_entry_still_present_and_unchanged():
    entry = _entry(SUPPORTED_MODEL, SUPPORTED_HARDWARE)
    assert "vllm_options" in entry
