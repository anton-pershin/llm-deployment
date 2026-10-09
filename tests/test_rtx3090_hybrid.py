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


@pytest.fixture
def gptq_sources(tmp_path):
    donors, data = [], []
    for bits in (4, 8):
        donor = tmp_path / f"gptq{bits}"
        donor.mkdir()
        config = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "num_hidden_layers": 2,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "vocab_size": 16,
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "gptq",
                "bits": bits,
                "group_size": 128,
                "sym": True,
                "desc_act": False,
                "checkpoint_format": "gptq",
                "lm_head": False,
            },
        }
        (donor / "config.json").write_text(json.dumps(config))
        for filename in ("tokenizer.json", "generation_config.json"):
            (donor / filename).write_text(json.dumps({"donor": bits}))
        weights = {}
        for layer in range(2):
            prefix = f"model.layers.{layer}."
            for area, projections in (
                ("self_attn", ("q", "k", "v", "o")),
                ("mlp", ("gate", "up", "down")),
            ):
                for projection in projections:
                    rows = (
                        64
                        if projection in ("k", "v")
                        else (256 if projection in ("gate", "up") else 128)
                    )
                    columns = 256 if projection == "down" else 128
                    name = prefix + f"{area}.{projection}_proj."
                    weights[name + "qweight"] = torch.full(
                        (columns * bits // 32, rows), 12345 + bits, dtype=torch.int32
                    )
                    weights[name + "qzeros"] = torch.full(
                        (columns // 128, rows * bits // 32), bits, dtype=torch.int32
                    )
                    weights[name + "scales"] = torch.full(
                        (columns // 128, rows), bits / 10, dtype=torch.bfloat16
                    )
                    weights[name + "g_idx"] = (
                        torch.arange(columns, dtype=torch.int32) // 128
                    )
            for norm in (
                "input_layernorm",
                "post_attention_layernorm",
                "self_attn.q_norm",
                "self_attn.k_norm",
            ):
                weights[prefix + norm + ".weight"] = torch.full(
                    (64 if "self_attn" in norm else 128,), bits, dtype=torch.bfloat16
                )
        for name, shape in (
            ("model.embed_tokens.weight", (16, 128)),
            ("lm_head.weight", (16, 128)),
            ("model.norm.weight", (128,)),
        ):
            weights[name] = torch.full(shape, bits, dtype=torch.bfloat16)
        index = {}
        for shard in range(2):
            filename = f"model-{shard}.safetensors"
            selected = {
                k: v
                for k, v in weights.items()
                if (f"layers.{shard}." in k or (shard == 0 and "layers." not in k))
            }
            save_file(selected, donor / filename)
            index.update(dict.fromkeys(selected, filename))
        (donor / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": index})
        )
        donors.append(donor)
        data.append(weights)
    return *donors, tmp_path / "output", *data


@pytest.mark.parametrize("indexed", [True, False])
def test_gptq_splices_packed_tensors_and_fused_metadata(gptq_sources, indexed):
    mlp4, attention8, output, four, eight = gptq_sources
    if not indexed:
        for donor in (mlp4, attention8):
            (donor / "model.safetensors.index.json").unlink()
    before = fingerprint(mlp4), fingerprint(attention8)
    assert helper().prepare_gptq(mlp4, attention8, output) == output
    actual = load_file(output / "model.safetensors")
    assert actual.keys() == eight.keys()
    for name, tensor in actual.items():
        expected = four[name] if ".mlp." in name else eight[name]
        assert tensor.dtype == expected.dtype
        assert torch.equal(tensor, expected), name
    config = json.loads((output / "config.json").read_text())
    assert config["quantization_config"] == {
        "quant_method": "gptq",
        "checkpoint_format": "gptq",
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "desc_act": False,
        "lm_head": False,
        "dynamic": {
            r"+:^model\.layers\.\d+\.self_attn\.(?:qkv_proj|o_proj)$": {
                "bits": 8,
                "group_size": 128,
                "sym": True,
                "desc_act": False,
            }
        },
    }
    for filename in ("tokenizer.json", "generation_config.json"):
        assert (output / filename).read_bytes() == (attention8 / filename).read_bytes()
    assert not (output / "model.safetensors.index.json").exists()
    assert (fingerprint(mlp4), fingerprint(attention8)) == before


@pytest.mark.parametrize("donor", [0, 1])
@pytest.mark.parametrize(
    "key,value",
    [
        ("quant_method", "awq"),
        ("bits", 3),
        ("group_size", 64),
        ("sym", False),
        ("desc_act", True),
        ("checkpoint_format", "gptq_v2"),
        ("lm_head", True),
        ("dynamic", {".*": {"bits": 2}}),
    ],
)
def test_gptq_rejects_invalid_quantization(gptq_sources, donor, key, value):
    sources = gptq_sources[:2]
    output = gptq_sources[2]
    path = sources[donor] / "config.json"
    config = json.loads(path.read_text())
    config["quantization_config"][key] = value
    path.write_text(json.dumps(config))
    before = tuple(fingerprint(p) for p in sources)
    with pytest.raises(ValueError):
        helper().prepare_gptq(*sources, output)
    assert not output.exists()
    assert tuple(fingerprint(p) for p in sources) == before


@pytest.mark.parametrize("donor", [0, 1])
@pytest.mark.parametrize(
    "fault", ["architecture", "layout", "missing_projection", "shape"]
)
def test_gptq_rejects_invalid_architecture_or_projection(gptq_sources, donor, fault):
    sources, output = gptq_sources[:2], gptq_sources[2]
    if fault in ("architecture", "layout"):
        path = sources[donor] / "config.json"
        config = json.loads(path.read_text())
        config["model_type" if fault == "architecture" else "hidden_size"] = (
            "llama" if fault == "architecture" else 256
        )
        path.write_text(json.dumps(config))
    else:
        path = sources[donor] / "model-0.safetensors"
        weights = load_file(path)
        name = (
            "model.layers.0."
            + ("mlp.gate_proj." if donor == 0 else "self_attn.o_proj.")
            + "qweight"
        )
        if fault == "missing_projection":
            del weights[name]
            index_path = sources[donor] / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            del index["weight_map"][name]
            index_path.write_text(json.dumps(index))
        else:
            weights[name] = torch.ones(1, dtype=torch.int32)
        save_file(weights, path)
    with pytest.raises(ValueError):
        helper().prepare_gptq(*sources, output)
    assert not output.exists()


@pytest.mark.parametrize("fault", ["missing_norm", "dense_projection", "invalid_g_idx"])
def test_gptq_rejects_incomplete_or_ambiguous_weights(gptq_sources, fault):
    mlp4, attention8, output, _, _ = gptq_sources
    path = attention8 / "model-0.safetensors"
    weights = load_file(path)
    if fault == "missing_norm":
        del weights["model.layers.0.post_attention_layernorm.weight"]
        (attention8 / "model.safetensors.index.json").unlink()
    elif fault == "dense_projection":
        weights["model.layers.0.self_attn.q_proj.weight"] = torch.ones(128, 128)
        (attention8 / "model.safetensors.index.json").unlink()
    else:
        weights["model.layers.0.self_attn.q_proj.g_idx"] += 1
    save_file(weights, path)
    with pytest.raises(ValueError):
        helper().prepare_gptq(mlp4, attention8, output)
    assert not output.exists()


def test_gptq_cache_and_atomic_failure(gptq_sources, monkeypatch):
    mlp4, attention8, output, _, _ = gptq_sources
    module = helper()
    before = fingerprint(mlp4), fingerprint(attention8)
    real_save = module.save_file

    def broken_save(weights, path, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(module, "save_file", broken_save)
    with pytest.raises(OSError, match="disk full"):
        module.prepare_gptq(mlp4, attention8, output)
    assert not output.exists()
    assert not list(output.parent.glob(".output-*"))
    monkeypatch.setattr(module, "save_file", real_save)
    module.prepare_gptq(mlp4, attention8, output)
    built = fingerprint(output)
    assert module.prepare_gptq(mlp4, attention8, output) == output
    assert fingerprint(output) == built
    assert (fingerprint(mlp4), fingerprint(attention8)) == before
    (output / "model.safetensors").unlink()
    with pytest.raises(ValueError):
        module.prepare_gptq(mlp4, attention8, output)


@pytest.mark.parametrize("donor", [0, 1])
def test_gptq_output_outside_donors(gptq_sources, donor):
    mlp4, attention8, _, _, _ = gptq_sources
    source = (mlp4, attention8)[donor]
    before = fingerprint(source)
    with pytest.raises(ValueError):
        helper().prepare_gptq(mlp4, attention8, source / "output")
    assert fingerprint(source) == before


def test_gptq_cli(gptq_sources):
    mlp4, attention8, output, _, _ = gptq_sources
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(mlp4),
            str(attention8),
            str(output),
            "--format",
            "gptq",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "model.safetensors").is_file()


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
