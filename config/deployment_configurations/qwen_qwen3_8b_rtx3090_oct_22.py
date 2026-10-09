"""Offline Qwen3 AWQ restoration or mixed GPTQ checkpoint preparation.

Run with AWQ_SNAPSHOT ORIGINAL_BF16_SNAPSHOT OUTPUT [--attention-projections all].
For GPTQ use MLP4_SNAPSHOT ATTENTION8_SNAPSHOT OUTPUT --format gptq.
No downloads, calibration, GPU use, or writes to source snapshots are performed.
"""

import json
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def tensors(path, names=None):
    """Read indexed or sorted shards, loading only requested original tensors."""
    index = path / "model.safetensors.index.json"
    mapping = json.loads(index.read_text())["weight_map"] if index.exists() else None
    shards = (
        sorted(set(mapping.values()))
        if mapping
        else sorted(p.name for p in path.glob("*.safetensors"))
    )
    result, seen = {}, {}
    for shard in shards:
        if Path(shard).name != shard:
            raise ValueError(f"Invalid shard filename: {shard}")
        with safe_open(path / shard, framework="pt", device="cpu") as source:
            for name in source.keys():  # noqa: SIM118 -- safe_open is not iterable
                if name in seen or (mapping is not None and mapping.get(name) != shard):
                    raise ValueError(f"Duplicate or misindexed tensor: {name}")
                seen[name] = shard
                if names is None or name in names:
                    result[name] = source.get_tensor(name)
    if not seen or (mapping is not None and seen != mapping):
        raise ValueError(f"Empty or incomplete checkpoint: {path}")
    return result


def prepare(awq, original, output, attention_projections="qkv"):
    """Build atomically; reuse only a complete output with matching provenance."""
    awq, original, output = (Path(p).absolute() for p in (awq, original, output))
    if any(
        output.resolve().is_relative_to(source.resolve()) for source in (awq, original)
    ):
        raise ValueError("Output must be outside source snapshots")
    if attention_projections not in ("qkv", "all"):
        raise ValueError("attention_projections must be qkv or all")
    manifest = {
        "awq": str(awq.resolve()),
        "original": str(original.resolve()),
        "attention_projections": attention_projections,
        "format_version": 1,
    }
    if output.exists():
        record = output / "preparation_manifest.json"
        if (
            record.is_file()
            and json.loads(record.read_text()) == manifest
            and all(
                (output / name).is_file()
                for name in ("config.json", "model.safetensors")
            )
        ):
            return output
        raise ValueError(f"Conflicting or incomplete output: {output}")
    config = json.loads((awq / "config.json").read_text())
    base = json.loads((original / "config.json").read_text())
    layout = (
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    )
    for source in (config, base):
        if source.get("model_type") != "qwen3" or source.get("architectures") != [
            "Qwen3ForCausalLM"
        ]:
            raise ValueError("Expected Qwen3ForCausalLM architecture")
    if any(
        not isinstance(config.get(k), int) or config[k] <= 0 or config[k] != base.get(k)
        for k in layout
    ):
        raise ValueError("Incompatible or invalid Qwen3 layout")
    quantization = config.get("quantization_config", {})
    if quantization.get("quant_method") != "awq" or quantization.get("bits") != 4:
        raise ValueError("Expected a four-bit AWQ checkpoint")
    weights = tensors(awq)
    projections = (
        ("q", "k", "v", "o") if attention_projections == "all" else ("q", "k", "v")
    )
    required, optional, remove = {}, {}, []
    hidden, head = config["hidden_size"], config["head_dim"]
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            name = prefix + norm + ".weight"
            if name not in weights or tuple(weights[name].shape) != (hidden,):
                raise ValueError(f"Missing or invalid calibrated norm: {name}")
        required[prefix + "input_layernorm.weight"] = (hidden,)
        for norm in ("q_norm", "k_norm"):
            optional[prefix + f"self_attn.{norm}.weight"] = (head,)
        for projection in projections:
            name = prefix + f"self_attn.{projection}_proj."
            keys = [name + part for part in ("qweight", "qzeros", "scales")]
            if not all(k in weights for k in keys) or name + "weight" in weights:
                raise ValueError(f"Missing or invalid packed attention: {name}")
            remove.extend(keys)
            rows = (
                hidden
                if projection == "o"
                else head
                * config[
                    "num_key_value_heads"
                    if projection in ("k", "v")
                    else "num_attention_heads"
                ]
            )
            columns = (
                head * config["num_attention_heads"] if projection == "o" else hidden
            )
            required[name + "weight"] = (rows, columns)
    dense = tensors(original, required.keys() | optional.keys())
    if required.keys() - dense.keys():
        raise ValueError(
            f"Missing original weights: {sorted(required.keys() - dense.keys())}"
        )
    for name, tensor in dense.items():
        if (
            not tensor.is_floating_point()
            or tuple(tensor.shape) != (required | optional)[name]
        ):
            raise ValueError(f"Invalid original tensor: {name}")
        weights[name] = tensor.to(torch.bfloat16)
    for name in remove:
        del weights[name]
    config["torch_dtype"] = "bfloat16"
    exclusions = quantization.get("modules_to_not_convert") or []
    quantization["modules_to_not_convert"] = list(
        dict.fromkeys([*exclusions, *(f"{p}_proj" for p in projections)])
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        save_file(weights, temporary / "model.safetensors", metadata={"format": "pt"})
        (temporary / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        for path in awq.iterdir():
            if (
                path.is_file()
                and path.name != "config.json"
                and not path.name.endswith(".index.json")
                and path.suffix in (".json", ".txt", ".model", ".jinja")
            ):
                shutil.copyfile(path, temporary / path.name)
        (temporary / "preparation_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        if output.exists():
            raise ValueError(f"Output appeared during preparation: {output}")
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def prepare_gptq(mlp4, attention8, output):
    """Copy GPTQ attention8/MLP4 without calibration or repacking."""
    mlp4, attention8, output = (Path(p).absolute() for p in (mlp4, attention8, output))
    if any(
        output.resolve().is_relative_to(source.resolve())
        for source in (mlp4, attention8)
    ):
        raise ValueError("Output must be outside source snapshots")
    manifest = {
        "mlp4": str(mlp4.resolve()),
        "attention8": str(attention8.resolve()),
        "format": "gptq",
        "format_version": 1,
    }
    if output.exists():
        record = output / "preparation_manifest.json"
        if (
            record.is_file()
            and json.loads(record.read_text()) == manifest
            and all(
                (output / name).is_file()
                for name in ("config.json", "model.safetensors")
            )
        ):
            return output
        raise ValueError(f"Conflicting or incomplete output: {output}")
    config = json.loads((attention8 / "config.json").read_text())
    base = json.loads((mlp4 / "config.json").read_text())
    layout = (
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    )
    if any(
        type(config.get(k)) is not int or config[k] <= 0 or config[k] != base.get(k)
        for k in layout
    ):
        raise ValueError("Incompatible or invalid Qwen3 layout")
    for source, bits in ((base, 4), (config, 8)):
        if source.get("model_type") != "qwen3" or source.get("architectures") != [
            "Qwen3ForCausalLM"
        ]:
            raise ValueError("Expected Qwen3ForCausalLM architecture")
        quant = source.get("quantization_config", {})
        expected = {
            "quant_method": "gptq",
            "bits": bits,
            "group_size": 128,
            "sym": True,
            "desc_act": False,
        }
        if (
            any(
                type(quant.get(k)) is not type(v) or quant[k] != v
                for k, v in expected.items()
            )
            or quant.get("checkpoint_format", "gptq") != "gptq"
            or quant.get("lm_head", False) is not False
            or quant.get("dynamic")
        ):
            raise ValueError(
                "Expected uniform symmetric GPTQ group128 without act order"
            )
    weights = tensors(attention8)
    four = tensors(mlp4)
    for donor, bits, area, projections in (
        (weights, 8, "self_attn", ("q", "k", "v", "o")),
        (four, 4, "mlp", ("gate", "up", "down")),
    ):
        for layer in range(config["num_hidden_layers"]):
            for projection in projections:
                name = f"model.layers.{layer}.{area}.{projection}_proj."
                rows = (
                    config["head_dim"] * config["num_key_value_heads"]
                    if projection in ("k", "v")
                    else config["intermediate_size"]
                    if projection in ("gate", "up")
                    else config["hidden_size"]
                )
                columns = (
                    config["intermediate_size"]
                    if projection == "down"
                    else config["head_dim"] * config["num_attention_heads"]
                    if projection == "o"
                    else config["hidden_size"]
                )
                if name + "weight" in donor:
                    raise ValueError(f"Unexpected dense projection: {name}")
                shapes = {
                    "qweight": (columns * bits // 32, rows),
                    "qzeros": (columns // 128, rows * bits // 32),
                    "scales": (columns // 128, rows),
                    "g_idx": (columns,),
                }
                for part, shape in shapes.items():
                    tensor = donor.get(name + part)
                    if (
                        tensor is None
                        or tuple(tensor.shape) != shape
                        or (
                            not tensor.is_floating_point()
                            if part == "scales"
                            else tensor.dtype != torch.int32
                        )
                    ):
                        raise ValueError(
                            f"Missing or invalid packed projection: {name + part}"
                        )
                if not torch.equal(
                    donor[name + "g_idx"],
                    torch.arange(columns, dtype=torch.int32) // 128,
                ):
                    raise ValueError(f"Invalid non-act-order group indices: {name}")
    floating = {"model.norm.weight": (config["hidden_size"],)}
    for layer in range(config["num_hidden_layers"]):
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ):
            floating[f"model.layers.{layer}.{norm}.weight"] = (
                config["head_dim"] if "self_attn" in norm else config["hidden_size"],
            )
    for name, shape in floating.items():
        tensor = weights.get(name)
        if (
            tensor is None
            or not tensor.is_floating_point()
            or tuple(tensor.shape) != shape
        ):
            raise ValueError(f"Missing or invalid floating tensor: {name}")
    weights.update({k: v for k, v in four.items() if ".mlp." in k})
    config["quantization_config"] = {
        "quant_method": "gptq",
        "checkpoint_format": "gptq",
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "desc_act": False,
        "lm_head": False,
        # vLLM 0.30 GPTQConfig override_quant_config sees fused module names.
        "dynamic": {
            r"+:^model\.layers\.\d+\.self_attn\.(?:qkv_proj|o_proj)$": {
                "bits": 8,
                "group_size": 128,
                "sym": True,
                "desc_act": False,
            }
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        save_file(weights, temporary / "model.safetensors", metadata={"format": "pt"})
        (temporary / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        for path in attention8.iterdir():
            if (
                path.is_file()
                and path.name != "config.json"
                and not path.name.endswith(".index.json")
                and path.suffix in (".json", ".txt", ".model", ".jinja")
            ):
                shutil.copyfile(path, temporary / path.name)
        (temporary / "preparation_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        if output.exists():
            raise ValueError(f"Output appeared during preparation: {output}")
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("awq", type=Path)
    parser.add_argument("original", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--format", choices=("awq", "gptq"), default="awq")
    parser.add_argument(
        "--attention-projections", choices=("qkv", "all"), default="qkv"
    )
    args = parser.parse_args()
    if args.format == "gptq":
        print(prepare_gptq(args.awq, args.original, args.output))
    else:
        print(prepare(args.awq, args.original, args.output, args.attention_projections))
