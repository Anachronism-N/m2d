"""Offline stdlib tests. Set WANDANCER_OFFICIAL_SOURCE to the pinned clean checkout."""
import ast
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import apply_official_patch as applier
import run_native

OFFICIAL_SOURCE = Path(os.environ.get("WANDANCER_OFFICIAL_SOURCE",
    str(Path(__file__).resolve().parents[3].parent / "Wan-Dancer")))


def functions(text):
    return {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}


def ast_dump(node):
    return ast.dump(node, include_attributes=False)


class OfficialPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata, cls.patch_text = applier.verify_source(OFFICIAL_SOURCE)
        cls.originals = {name: (OFFICIAL_SOURCE / name).read_text() for name in cls.metadata["files"]}
        cls.modified = applier.render_patch(cls.originals, cls.patch_text)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        for name in [*self.metadata["files"], *self.metadata["unchanged_files"]]:
            destination = self.source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(OFFICIAL_SOURCE / name, destination)
        self.models = self.root / "models"
        for name in ["global_model.safetensors", "local_model.safetensors", *run_native.COMMON_WEIGHTS]:
            target = self.models / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"offline fixture, not actual weights")
        self.manifest = {}
        for key in ("image_path", "music_path", "prompt_path", "global_video_path"):
            target = self.root / (key + ";not-a-shell-command")
            target.write_bytes(b"offline input")
            self.manifest[key] = str(target)
        self.manifest["seed"] = 0
        self.manifest_path = self.root / "manifest.json"

    def apply_fixture(self):
        applier.apply_patch(self.source, self.root / "patch_receipt.json")
        return applier.verify_source(self.source, patched=True)

    def launch(self, stage="global", **changes):
        self.apply_fixture()
        manifest = dict(self.manifest)
        if stage == "global":
            manifest.pop("global_video_path")
        manifest.update(changes)
        self.manifest_path.write_text(json.dumps(manifest))
        return run_native.build_launch(self.manifest_path, "offline-case", stage, self.source,
                                       Path(sys.executable), self.models, self.root / "private")

    def test_compile_and_patched_hashes(self):
        for name, text in self.modified.items():
            compile(text, name, "exec")
            self.assertEqual(applier.sha256(text.encode()), self.metadata["files"][name]["patched_sha256"])

    def test_check_only_does_not_write(self):
        report = applier.apply_patch(self.source)
        self.assertFalse(report["applied"])
        applier.verify_source(self.source)

    def test_apply_then_refuse_second_application(self):
        self.apply_fixture()
        receipt = json.loads((self.root / "patch_receipt.json").read_text())
        self.assertEqual(receipt["patch_sha256"], self.metadata["patch_sha256"])
        with self.assertRaises(ValueError):
            applier.apply_patch(self.source, self.root / "another_receipt.json")

    def test_unknown_source_change_refused_without_partial_write(self):
        name = "gen_video/gen_video_local.py"
        (self.source / name).write_text(self.originals[name] + "\n# unknown change\n")
        with self.assertRaises(ValueError):
            applier.apply_patch(self.source, self.root / "patch_receipt.json")
        self.assertEqual((self.source / "gen_video/gen_video_global.py").read_text(),
                         self.originals["gen_video/gen_video_global.py"])
        self.assertFalse((self.root / "patch_receipt.json").exists())

    def test_unknown_pipeline_change_refused(self):
        (self.source / "diffsynth/pipelines/wan_video_new.py").write_text("unknown")
        with self.assertRaises(ValueError):
            applier.apply_patch(self.source)

    def test_patch_hash_tampering_refused(self):
        with patch.object(applier, "sha256", return_value="not-the-pinned-hash"):
            with self.assertRaises(ValueError):
                applier.load_metadata()

    def test_wrong_context_refused(self):
        originals = dict(self.originals)
        originals["gen_video/gen_video_global.py"] = originals["gen_video/gen_video_global.py"].replace(
            '        device="cuda",', '        device="cpu",', 1)
        with self.assertRaises(ValueError):
            applier.render_patch(originals, self.patch_text)

    def test_protected_functions_and_defaults_unchanged(self):
        for name, original in self.originals.items():
            before, after = functions(original), functions(self.modified[name])
            protected = ["parse_args", "get_music_base_feature", "crop_and_resize", "gen_video_single", "main"]
            if name.endswith("global.py"):
                protected.append("gen_video")
            else:
                protected.extend(["get_music_clip_149f", "get_music_features", "process_global_video_firstlastframe"])
            for function in protected:
                self.assertEqual(ast_dump(before[function]), ast_dump(after[function]), (name, function))
            self.assertEqual(ast_dump(ast.parse(original).body[-1]), ast_dump(ast.parse(self.modified[name]).body[-1]))

    def test_all_model_configs_explicit_no_download_metadata(self):
        for name, text in self.modified.items():
            function = functions(text)["init_dit_model"]
            configs = [node for node in ast.walk(function) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Name) and node.func.id == "ModelConfig"]
            self.assertEqual(len(configs), 5)
            for config in configs:
                keys = {keyword.arg for keyword in config.keywords}
                self.assertIn("path", keys)
                self.assertFalse(keys & {"model_id", "origin_file_pattern"})
            call = next(node for node in ast.walk(function) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained")
            old_call = next(node for node in ast.walk(functions(self.originals[name])["init_dit_model"])
                            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "from_pretrained")
            for old_keyword in old_call.keywords:
                if old_keyword.arg not in ("model_configs", "tokenizer_config"):
                    keyword = next(k for k in call.keywords if k.arg == old_keyword.arg)
                    self.assertEqual(ast_dump(old_keyword), ast_dump(keyword))

    def test_local_cfg_scale_reaches_input_config(self):
        node = functions(self.modified["gen_video/gen_video_local.py"])["gen_video"]
        self.assertEqual(node.args.args[-1].arg, "cfg_scale")
        self.assertEqual(node.args.defaults[-1].value, 5)
        class FakeTensor:
            def to(self, **kwargs):
                return self
        captured = []
        scope = {"np": SimpleNamespace(load=lambda path: []),
                 "torch": SimpleNamespace(from_numpy=lambda feature: FakeTensor(), bfloat16="bfloat16"),
                 "gen_video_single": lambda pipe, prompt, config: captured.append(config)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "isolated gen_video", "exec"), scope)
        scope["gen_video"](None, "audio", "feature", "prompt", "output", cfg_scale=0)
        self.assertEqual(captured[0]["cfg_scale"], 0)
        scope["gen_video"](None, "audio", "feature", "prompt", "output")
        self.assertEqual(captured[1]["cfg_scale"], 5)
        process = functions(self.modified["gen_video/gen_video_local.py"])["process"]
        call = next(n for n in ast.walk(process) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "gen_video")
        self.assertEqual(ast_dump(next(k.value for k in call.keywords if k.arg == "cfg_scale")),
                         ast_dump(ast.parse("args.cfg_scale", mode="eval").body))

    def test_rank0_keyframe_save_before_dist_init(self):
        process = functions(self.modified["gen_video/gen_video_local.py"])["process"]
        guard = next(n for n in process.body if isinstance(n, ast.If) and 'os.environ.get' in ast.unparse(n.test))
        saves = []
        scope = {"os": os, "keyframes_list": [["frame"], ["frame"]], "music_folder": str(self.root),
                 "FPS": 30, "save_video": lambda *args, **kwargs: saves.append(args)}
        module = compile(ast.Module(body=[guard], type_ignores=[]), "isolated save guard", "exec")
        for rank in range(8):
            saves.clear()
            with patch.dict(os.environ, {"RANK": str(rank)}):
                exec(module, scope)
            self.assertEqual(len(saves), 2 if rank == 0 else 0)
        init_index = next(i for i, n in enumerate(process.body) if isinstance(n, ast.Assign)
                          and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                          and n.value.func.id == "init_dit_model")
        for node in process.body[:init_index]:
            self.assertNotIn("dist.", ast.unparse(node))
        self.assertTrue(ast.unparse(process.body[init_index + 1]).startswith("dist.barrier("))
        self.assertTrue(ast.unparse(process.body[-1]).startswith("dist.barrier("))
        concat_guard = process.body[-2]
        self.assertEqual(ast.unparse(concat_guard.test), "dist.get_rank() == 0")
        old = functions(self.originals["gen_video/gen_video_local.py"])["process"]
        start = next(i for i, node in enumerate(old.body) if isinstance(node, ast.Assign)
                     and ast.unparse(node) == "music_path = args.music_path")
        self.assertEqual([ast_dump(n) for n in concat_guard.body], [ast_dump(n) for n in old.body[start:]])

    def test_launcher_single_node_safe_argv_and_private_cwd(self):
        command, environment, report = self.launch(cfg_scale=0)
        self.assertIn("--nnodes=1", command)
        self.assertIn("--nproc_per_node=8", command)
        self.assertEqual(command[command.index("--cfg_scale") + 1], "0")
        self.assertIn(self.manifest["image_path"], command)
        self.assertEqual(environment["PYTHONPATH"], str(self.source))
        self.assertEqual(environment["WANDANCER_MODEL_ROOT"], str(self.models))
        self.assertEqual(report["quality_status"], "NOT_EVALUATED")
        self.assertFalse(Path(report["cwd"]).exists())
        self.assertTrue(Path(report["expected_output"]).is_relative_to(Path(report["cwd"])))

    def test_local_launcher_requires_global_input(self):
        command, _, _ = self.launch("local")
        self.assertIn("--global_video_path", command)

    def test_launcher_unknown_options_refused(self):
        with self.assertRaises(ValueError):
            self.launch(arbitrary_shell="touch /tmp/not-allowed")

    def test_launcher_temporal_change_refused(self):
        with self.assertRaises(ValueError):
            self.launch(num_frames=81)

    def test_launcher_relative_input_refused(self):
        with self.assertRaises(ValueError):
            self.launch(image_path="relative.jpg")

    def test_launcher_missing_weight_refused(self):
        (self.models / "local_model.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            self.launch("local")

    def test_launcher_safe_run_id_and_output_ownership(self):
        for bad in ("../escape", "run;sh", ".", ""):
            with self.assertRaises(ValueError):
                run_native.build_launch(None, bad, "global", None, None, None, None)
        with self.assertRaises(ValueError):
            run_native.private_output(self.source / "outputs", self.source)
        with self.assertRaises(ValueError):
            run_native.private_output(run_native.PUBLISH_ROOT / "outputs", self.source)


if __name__ == "__main__":
    unittest.main()
