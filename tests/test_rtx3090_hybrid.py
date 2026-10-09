"""CPU-only integration tests with real sharded safetensors checkpoints."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

SCRIPT = (
    Path(__file__).parents[1]
    / "config/deployment_configurations/qwen_qwen3_8b_rtx3090_oct_22.py"
)


def helper():
    assert SCRIPT.is_file(), "offline preparation helper has not been implemented"
    spec = importlib.util.spec_from_file_location("hybrid", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sources(tmp_path):
    awq, original = tmp_path / "awq", tmp_path / "original"
    awq.mkdir()
    original.mkdir()
    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "num_hidden_layers": 2,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
    }
    (original / "config.json").write_text(json.dumps(config))
    config["quantization_config"] = {
        "quant_method": "awq",
        "bits": 4,
        "group_size": 8,
        "zero_point": True,
        "version": "gemm",
    }
    (awq / "config.json").write_text(json.dumps(config))
    (awq / "tokenizer.json").write_text('{"test": true}')
    (awq / "generation_config.json").write_text('{"eos_token_id": 1}')
    packed, dense = {}, {}
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        for projection in ("q", "k", "v", "o"):
            name = prefix + f"self_attn.{projection}_proj."
            rows = 4 if projection in ("k", "v") else 8
            dense[name + "weight"] = torch.full((rows, 8), 3.0, dtype=torch.bfloat16)
            for part in ("qweight", "qzeros", "scales"):
                dtype = torch.float16 if part == "scales" else torch.int32
                packed[name + part] = torch.full((8, 1), 2, dtype=dtype)
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ):
            shape = (4,) if "self_attn" in norm else (8,)
            packed[prefix + norm + ".weight"] = torch.full(
                shape, 2.0, dtype=torch.float16
            )
            dense[prefix + norm + ".weight"] = torch.full(
                shape, 3.0, dtype=torch.bfloat16
            )
        for projection in ("gate", "up", "down"):
            for part in ("qweight", "qzeros", "scales"):
                dtype = torch.float16 if part == "scales" else torch.int32
                packed[prefix + f"mlp.{projection}_proj.{part}"] = torch.full(
                    (8, 2), 7, dtype=dtype
                )
    packed["model.embed_tokens.weight"] = torch.full((10, 8), 2.0, dtype=torch.float16)
    index = {}
    for number, layer in enumerate(("model.layers.0.", "model.layers.1.")):
        name = f"model-{number}.safetensors"
        tensors = {
            k: v
            for k, v in packed.items()
            if k.startswith(layer) or (number == 0 and "embed_tokens" in k)
        }
        save_file(tensors, awq / name)
        index.update({k: name for k in tensors})
        save_file(
            {k: v for k, v in dense.items() if k.startswith(layer)}, original / name
        )
    (awq / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    return awq, original, tmp_path / "output", packed, dense


@pytest.mark.parametrize(
    "fault",
    [
        "missing_weight",
        "wrong_shape",
        "wrong_dtype",
        "wrong_architecture",
        "wrong_layout",
        "not_awq",
        "invalid_selection",
        "missing_packed",
        "missing_norm",
        "missing_shard",
    ],
)
def test_invalid_sources_fail_without_output(sources, fault):
    awq, original, output, _, _ = sources
    selection = "qkv"
    if fault in ("wrong_architecture", "wrong_layout", "not_awq"):
        path = awq / "config.json"
        config = json.loads(path.read_text())
        if fault == "wrong_architecture":
            config["model_type"] = "llama"
        elif fault == "wrong_layout":
            config["hidden_size"] = 16
        else:
            config["quantization_config"]["quant_method"] = "gptq"
        path.write_text(json.dumps(config))
    elif fault == "invalid_selection":
        selection = "bad"
    elif fault == "missing_shard":
        (awq / "model-1.safetensors").unlink()
    else:
        path = (
            awq if fault in ("missing_packed", "missing_norm") else original
        ) / "model-0.safetensors"
        data = load_file(path)
        name = "model.layers.0.self_attn.q_proj.weight"
        if fault == "missing_packed":
            name = name.replace("weight", "scales")
        if fault == "missing_norm":
            name = "model.layers.0.input_layernorm.weight"
        if fault in ("missing_weight", "missing_packed", "missing_norm"):
            del data[name]
        elif fault == "wrong_shape":
            data[name] = torch.ones(1, dtype=torch.bfloat16)
        else:
            data[name] = data[name].to(torch.int32)
        save_file(data, path)
    before = fingerprint(awq), fingerprint(original)
    with pytest.raises((ValueError, FileNotFoundError)):
        helper().prepare(awq, original, output, selection)
    assert not output.exists()
    assert not list(output.parent.glob(".output-*"))
    assert (fingerprint(awq), fingerprint(original)) == before


def test_cli_and_existing_exclusions(sources):
    awq, original, output, _, _ = sources
    path = awq / "config.json"
    config = json.loads(path.read_text())
    config["quantization_config"]["modules_to_not_convert"] = ["lm_head"]
    path.write_text(json.dumps(config))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(awq),
            str(original),
            str(output),
            "--attention-projections",
            "all",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "model.safetensors").is_file()
    config = json.loads((output / "config.json").read_text())
    assert config["quantization_config"]["modules_to_not_convert"] == [
        "lm_head",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]


@pytest.mark.parametrize("source", ["awq", "original"])
def test_output_cannot_be_inside_source(sources, source):
    awq, original, _, _, _ = sources
    output = (awq if source == "awq" else original) / "nested" / "output"
    before = fingerprint(awq), fingerprint(original)
    with pytest.raises(ValueError):
        helper().prepare(awq, original, output)
    assert (fingerprint(awq), fingerprint(original)) == before


@pytest.mark.parametrize(
    "missing", ["config.json", "model.safetensors", "preparation_manifest.json"]
)
def test_incomplete_cached_output_is_rejected(sources, missing):
    awq, original, output, _, _ = sources
    module = helper()
    module.prepare(awq, original, output)
    (output / missing).unlink()
    before = fingerprint(output)
    with pytest.raises(ValueError):
        module.prepare(awq, original, output)
    assert fingerprint(output) == before


def test_write_failure_is_atomic(sources, monkeypatch):
    awq, original, output, _, _ = sources
    module = helper()

    def broken_save(weights, path, **kwargs):
        Path(path).write_bytes(b"partial write")
        raise OSError("disk full")

    monkeypatch.setattr(module, "save_file", broken_save)
    with pytest.raises(OSError, match="disk full"):
        module.prepare(awq, original, output)
    assert not output.exists()
    assert not list(output.parent.glob(".output-*"))


def fingerprint(directory):
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}


def test_repeat_is_cached_and_sources_immutable(sources):
    awq, original, output, _, _ = sources
    before = fingerprint(awq), fingerprint(original)
    module = helper()
    assert module.prepare(awq, original, output) == output
    built = fingerprint(output)
    assert module.prepare(awq, original, output) == output
    assert fingerprint(output) == built
    assert (fingerprint(awq), fingerprint(original)) == before
    manifest = json.loads((output / "preparation_manifest.json").read_text())
    assert manifest["awq"] == str(awq.resolve())
    assert manifest["original"] == str(original.resolve())
    assert manifest["attention_projections"] == "qkv"
    with pytest.raises(ValueError):
        module.prepare(awq, original, output, "all")
    assert fingerprint(output) == built


def test_all_restores_output_projection(sources):
    awq, original, output, _, dense = sources
    helper().prepare(awq, original, output, attention_projections="all")
    actual = load_file(output / "model.safetensors")
    for layer in range(2):
        name = f"model.layers.{layer}.self_attn.o_proj."
        assert torch.equal(actual[name + "weight"], dense[name + "weight"])
        assert all(
            name + part not in actual for part in ("qweight", "qzeros", "scales")
        )
    config = json.loads((output / "config.json").read_text())
    assert config["quantization_config"]["modules_to_not_convert"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]


def test_qkv_restores_attention_preserving_calibrated_mlp(sources):
    awq, original, output, packed, dense = sources
    helper().prepare(awq, original, output)
    actual = load_file(output / "model.safetensors")
    for name, tensor in packed.items():
        replacement = name.replace("qweight", "weight")
        selected = any(f"self_attn.{p}_proj." in name for p in ("q", "k", "v"))
        if selected:
            assert name not in actual
            assert (
                torch.equal(actual[replacement], dense[replacement])
                if name.endswith("qweight")
                else True
            )
        elif any(
            n in name
            for n in ("input_layernorm", "self_attn.q_norm", "self_attn.k_norm")
        ):
            assert torch.equal(actual[name], dense[name])
            assert actual[name].dtype == torch.bfloat16
        else:
            assert actual[name].dtype == tensor.dtype
            assert torch.equal(actual[name], tensor)
    config = json.loads((output / "config.json").read_text())
    assert config["quantization_config"]["modules_to_not_convert"] == [
        "q_proj",
        "k_proj",
        "v_proj",
    ]
    assert config["torch_dtype"] == "bfloat16"
    assert config["quantization_config"]["quant_method"] == "awq"
    assert not (output / "model.safetensors.index.json").exists()
    assert (output / "tokenizer.json").read_bytes() == (
        awq / "tokenizer.json"
    ).read_bytes()
