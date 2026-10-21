"""Verify pinned originals, then apply only the two audited inference-file changes."""
import argparse
import hashlib
import json
from pathlib import Path
import re

PATCH_DIR = Path(__file__).resolve().parents[1] / "patches"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_metadata():
    metadata = json.loads((PATCH_DIR / "metadata.json").read_text())
    patch = (PATCH_DIR / metadata["patch_file"]).read_bytes()
    if sha256(patch) != metadata["patch_sha256"]:
        raise ValueError("Patch SHA256 does not match pinned metadata")
    return metadata, patch.decode()


def render_patch(originals, patch):
    lines = patch.splitlines(keepends=True)
    output = {}
    i = 0
    while i < len(lines):
        if not lines[i].startswith("--- a/") or not lines[i + 1].startswith("+++ b/"):
            raise ValueError("Unexpected patch header")
        name = lines[i][6:].rstrip("\n")
        if lines[i + 1][6:].rstrip("\n") != name or name not in originals or name in output:
            raise ValueError("Unexpected or duplicate patch target")
        source = originals[name].splitlines(keepends=True)
        result = []
        pos = 0
        i += 2
        while i < len(lines) and lines[i].startswith("@@ "):
            match = re.fullmatch(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@\n", lines[i])
            if not match:
                raise ValueError("Unsupported hunk header")
            old_start, old_count, new_start, new_count = map(int, match.groups())
            start = old_start - 1
            if start < pos or start > len(source):
                raise ValueError("Invalid hunk offset")
            result.extend(source[pos:start])
            if len(result) != new_start - 1:
                raise ValueError("Invalid new hunk offset")
            old_chunk, new_chunk = [], []
            i += 1
            while i < len(lines) and lines[i][:1] in (" ", "+", "-") and not lines[i].startswith("--- a/"):
                marker, text = lines[i][0], lines[i][1:]
                if marker != "+":
                    old_chunk.append(text)
                if marker != "-":
                    new_chunk.append(text)
                i += 1
            if len(old_chunk) != old_count or len(new_chunk) != new_count:
                raise ValueError("Hunk line count mismatch")
            if source[start:start + old_count] != old_chunk:
                raise ValueError("Patch context mismatch; refusing unknown changes")
            result.extend(new_chunk)
            pos = start + old_count
        result.extend(source[pos:])
        output[name] = "".join(result)
    if set(output) != set(originals):
        raise ValueError("Patch target set mismatch")
    return output


def verify_source(source_root, patched=False):
    root = Path(source_root).resolve(strict=True)
    metadata, patch = load_metadata()
    key = "patched_sha256" if patched else "original_sha256"
    for name, hashes in metadata["files"].items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or sha256(path.read_bytes()) != hashes[key]:
            raise ValueError("Unknown source contents: " + name)
    for name, expected in metadata["unchanged_files"].items():
        if sha256((root / name).read_bytes()) != expected:
            raise ValueError("Unreviewed pipeline contents: " + name)
    return metadata, patch


def apply_patch(source_root, receipt_path=None):
    root = Path(source_root).resolve(strict=True)
    metadata, patch = verify_source(root)
    originals = {name: (root / name).read_text() for name in metadata["files"]}
    modified = render_patch(originals, patch)
    for name, text in modified.items():
        compile(text, name, "exec")
        if sha256(text.encode()) != metadata["files"][name]["patched_sha256"]:
            raise ValueError("Patched hash mismatch: " + name)
    report = {"official_commit": metadata["official_commit"],
              "source_root": str(root), "patch_sha256": metadata["patch_sha256"],
              "verified": "pinned original file and unchanged pipeline SHA256",
              "applied": receipt_path is not None, "files": metadata["files"]}
    if receipt_path is not None:
        receipt = Path(receipt_path)
        if not receipt.is_absolute():
            raise ValueError("Receipt must be absolute")
        receipt = receipt.resolve()
        publish = Path(__file__).resolve().parents[3]
        if receipt.exists() or receipt.is_relative_to(root) or receipt.is_relative_to(publish):
            raise ValueError("Receipt must be a new path outside source and publish")
        receipt.parent.resolve(strict=True)
        for name, text in modified.items():
            (root / name).write_text(text)
        with receipt.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if not args.source_root.is_absolute():
        parser.error("--source-root must be absolute")
    if args.apply != (args.receipt is not None):
        parser.error("--apply requires --receipt; check-only does not write")
    print(json.dumps(apply_patch(args.source_root, args.receipt), indent=2))


if __name__ == "__main__":
    main()
