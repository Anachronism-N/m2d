"""Launch one native Wan-Dancer stage on one node/eight ranks; default is dry-run.

Manifest: absolute image_path, music_path, prompt_path, integer seed; Local also
requires absolute global_video_path. Optional integer cfg_scale, sigma_shift,
num_inference_steps, height, width, num_frames override official CLI defaults.
All logs, scratch and videos live under OUTPUT_ROOT/RUN_ID/STAGE, never in git.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess

from apply_official_patch import verify_source

PUBLISH_ROOT = Path(__file__).resolve().parents[3]
COMMON_WEIGHTS = (
    "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "google/umt5-xxl/tokenizer_config.json", "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
)
OPTIONS = ("cfg_scale", "sigma_shift", "num_inference_steps", "height", "width", "num_frames")


def absolute_path(value, label, is_dir=False):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(label + " must be absolute")
    path = path.resolve(strict=True)
    valid = path.is_dir() if is_dir else path.is_file() and path.stat().st_size > 0
    if not valid:
        raise ValueError(label + " must be an existing " + ("directory" if is_dir else "nonempty file"))
    return path


def private_output(path, source):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("output-root must be absolute")
    path = path.resolve()
    if path.is_relative_to(source) or path.is_relative_to(PUBLISH_ROOT):
        raise ValueError("Outputs must be outside source and publish checkout")
    parent = path
    while not parent.exists():
        parent = parent.parent
    result = subprocess.run(["git", "-C", str(parent), "rev-parse", "--show-toplevel"],
                            capture_output=True, text=True, check=False)
    if result.returncode == 0:
        raise ValueError("Outputs must not be inside any git worktree")
    return path


def build_launch(manifest_path, run_id, stage, source_root, python_path, model_root, output_root):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("run-id must be a safe single path component")
    if stage not in ("global", "local"):
        raise ValueError("stage must be global or local")
    source = absolute_path(source_root, "source-root", is_dir=True)
    absolute_path(python_path, "python")
    # Preserve the venv executable symlink rather than resolving to system Python.
    python = Path(python_path).parent.resolve(strict=True) / Path(python_path).name
    if not os.access(python, os.X_OK):
        raise ValueError("Python interpreter is not executable")
    metadata, _ = verify_source(source, patched=True)
    manifest_path = absolute_path(manifest_path, "manifest")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")
    required = ["image_path", "music_path", "prompt_path"]
    if stage == "local":
        required.append("global_video_path")
    missing = set(required + ["seed"]) - set(manifest)
    if missing:
        raise ValueError("Missing required manifest keys: " + ", ".join(sorted(missing)))
    allowed = set(required + ["seed"] + list(OPTIONS))
    if set(manifest) - allowed:
        raise ValueError("Unexpected manifest keys: " + ", ".join(sorted(set(manifest) - allowed)))
    inputs = {key: absolute_path(manifest[key], key) for key in required}
    seed = manifest["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for key in OPTIONS:
        if key in manifest and (type(manifest[key]) is not int or manifest[key] < (0 if key == "cfg_scale" else 1)):
            raise ValueError(key + " must be an integer in its valid range")
    if manifest.get("num_frames", 149) != 149:
        raise ValueError("Native temporal/keyframe layout requires num_frames=149")
    models = absolute_path(model_root, "model-root", is_dir=True)
    weights = [stage + "_model.safetensors", *COMMON_WEIGHTS]
    weight_stats = {}
    for name in weights:
        path = absolute_path(models / name, name)
        stat = path.stat()
        weight_stats[name] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    output = private_output(output_root, source)
    run_dir = output / run_id / stage
    if run_dir.exists():
        raise ValueError("Run/stage directory already exists; refusing output overwrite")
    final_name = inputs["image_path"].name.split(".")[0] + "_" + inputs["music_path"].name.split(".")[0]
    if any(char in final_name for char in ("/", "\\", "\n", "\r")):
        raise ValueError("Input basenames cannot be represented by the official output naming")
    basename = (final_name + "_seed" + str(seed) + "_" + run_id + ".mp4" if stage == "global"
                else final_name + "_" + run_id + "_seed" + str(seed) + ".mp4")
    output_folder = run_dir / "videos"
    command = [str(python), "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--nproc_per_node=8", str(source / ("gen_video/gen_video_" + stage + ".py"))]
    for key, path in inputs.items():
        command.extend(["--" + key, str(path)])
    command.extend(["--seed", str(seed), "--timestamp", run_id, "--output_folder", str(output_folder)])
    for key in OPTIONS:
        if key in manifest:
            command.extend(["--" + key, str(manifest[key])])
    environment = dict(os.environ)
    environment.update({"WANDANCER_MODEL_ROOT": str(models), "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(source)})
    # Do not inherit a previous distributed job's rank/world-size as launcher topology.
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    report = {"run_id": run_id, "stage": stage, "node": socket.gethostname(),
              "source_root": str(source), "model_root": str(models), "python": str(python),
              "official_commit": metadata["official_commit"], "patch_sha256": metadata["patch_sha256"],
              "source_files": metadata["files"], "manifest_path": str(manifest_path),
              "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "manifest": manifest,
              "weights_stat_only_not_hash_verified": weight_stats, "cwd": str(run_dir),
              "command": command, "expected_output": str(output_folder / basename),
              "quality_status": "NOT_EVALUATED", "execution_status": "DRY_RUN"}
    return command, environment, report


def execute_launch(command, environment, report):
    run_dir = Path(report["cwd"])
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "videos").mkdir()
    report["started_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report["execution_status"] = "RUNNING"
    receipt = run_dir / "run_receipt.json"
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    try:
        # find_spec does not import diffsynth or load models; verify node-local provenance.
        probe = "import importlib.util,pathlib,sys; p=pathlib.Path(importlib.util.find_spec('diffsynth').origin).resolve(); assert p.is_relative_to(pathlib.Path(sys.argv[1]).resolve()), str(p); print(p)"
        with (run_dir / "stdout.log").open("x") as log:
            imported = subprocess.run([command[0], "-c", probe, report["source_root"]],
                                      cwd=run_dir, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                      check=False)
            if imported.returncode:
                raise RuntimeError("Node-local diffsynth import provenance check failed")
            result = subprocess.run(command, cwd=run_dir, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        report["returncode"] = result.returncode
        output = Path(report["expected_output"])
        if result.returncode != 0 or output.is_symlink() or not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("Native stage failed or expected output is missing/empty")
        report["execution_status"] = "GENERATED_NOT_QUALITY_VALIDATED"
        report["output_size_bytes"] = output.stat().st_size
    except BaseException as error:
        report["execution_status"] = "FAILED"
        report["error"] = str(error)
        raise
    finally:
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        receipt.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True, choices=("global", "local"))
    parser.add_argument("--source-root", type=Path, default=Path("/opt/wandancer20261021/source"))
    parser.add_argument("--python", type=Path, default=Path("/opt/venvs/wandancer20261021/bin/python"))
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    command, environment, report = build_launch(args.manifest, args.run_id, args.stage,
                                               args.source_root, args.python, args.model_root, args.output_root)
    if args.execute:
        report = execute_launch(command, environment, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
