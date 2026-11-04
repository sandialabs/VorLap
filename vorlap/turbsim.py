"""TurbSim input-file generation and execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - the production target is Linux.
    fcntl = None


_STRING_SENTINELS = {
    "default",
    "true",
    "false",
}


@dataclass(frozen=True)
class TurbSimRun:
    """A prepared or generated TurbSim wind-file record."""

    wind_file: Path
    input_file: Path
    executable: Path | None
    generated: bool
    skipped_existing: bool
    command: list[str]
    settings: dict[str, Any]
    returncode: int | None
    elapsed_s: float | None
    metadata_file: Path
    provenance_status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "wind_file": str(self.wind_file),
            "input_file": str(self.input_file),
            "executable": None if self.executable is None else str(self.executable),
            "generated": self.generated,
            "skipped_existing": self.skipped_existing,
            "command": self.command,
            "settings": self.settings,
            "returncode": self.returncode,
            "elapsed_s": self.elapsed_s,
            "metadata_file": str(self.metadata_file),
            "provenance_status": self.provenance_status,
        }


def default_template_path(vorlap_root: str | Path | None = None) -> Path:
    root = Path(vorlap_root) if vorlap_root is not None else Path(__file__).resolve().parents[1]
    return root / "run_headless" / "templates" / "turbsim_openfast_iec.inp"


def render_turbsim_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def quote_turbsim_string(value: str) -> str:
    text = str(value).strip()
    if not text:
        return '""'
    if text[0] in {'"', "'"} and text[-1:] == text[0]:
        return text
    if text.lower() in _STRING_SENTINELS:
        return text
    return json.dumps(text)


def replace_turbsim_keyword_text(text: str, keyword: str, value: Any) -> tuple[str, str, int]:
    """Replace one TurbSim value-leading keyword line and preserve its comment."""
    rendered = render_turbsim_value(value)
    pattern = re.compile(
        rf"^(?P<indent>[ \t]*)(?P<old>.*?)[ \t]+(?P<keyword>{re.escape(keyword)})(?P<rest>[ \t]*(?:-.*)?)$",
        re.MULTILINE,
    )
    old_value = ""

    def replacement(match: re.Match[str]) -> str:
        nonlocal old_value
        old_value = match.group("old").strip()
        return f"{match.group('indent')}{rendered:<13}{match.group('keyword'):<16}{match.group('rest')}"

    patched, count = pattern.subn(replacement, text, count=1)
    return patched, old_value, count


def render_turbsim_input(template: str, settings: dict[str, Any]) -> str:
    text = template
    missing: list[str] = []
    for keyword, value in settings.items():
        text, _, count = replace_turbsim_keyword_text(text, keyword, value)
        if count != 1:
            missing.append(keyword)
    if missing:
        raise ValueError(f"Could not patch TurbSim keyword(s): {', '.join(missing)}")
    return text


def expected_output_flags(wind_file: str | Path) -> dict[str, bool]:
    suffix = Path(wind_file).suffix.lower()
    if suffix == ".bts":
        return {"WrADFF": True, "WrBLFF": False}
    if suffix == ".wnd":
        return {"WrADFF": False, "WrBLFF": True}
    raise ValueError(f"TurbSim wind file must end in .bts or .wnd, got: {wind_file}")


def resolve_turbsim_executable(
    *,
    explicit: str | Path | None = None,
    vorlap_root: str | Path | None = None,
) -> Path | None:
    candidates: list[str | Path] = []
    if explicit:
        candidates.append(explicit)
    for name in ("TURBSIM_EXE", "OPENFAST_TURBSIM_EXE"):
        if os.environ.get(name):
            candidates.append(os.environ[name])
    if vorlap_root is not None:
        candidates.append(Path(vorlap_root) / "run_headless" / "openfast" / "install" / "bin" / "turbsim")
    found = shutil.which("turbsim")
    if found:
        candidates.append(found)
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    return None


def estimate_full_field_bytes(
    *,
    analysis_time_s: float,
    time_step_s: float,
    num_grid_y: int,
    num_grid_z: int,
    components: int = 3,
    bytes_per_value: int = 2,
) -> int:
    if time_step_s <= 0.0:
        raise ValueError("time_step_s must be positive.")
    samples = int(round(float(analysis_time_s) / float(time_step_s))) + 1
    return samples * int(num_grid_y) * int(num_grid_z) * int(components) * int(bytes_per_value)


def write_turbsim_input(
    input_file: str | Path,
    *,
    template_path: str | Path,
    settings: dict[str, Any],
) -> Path:
    path = Path(input_file)
    template = Path(template_path).read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_turbsim_input(template, settings), encoding="utf-8")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def turbsim_metadata_path(wind_file: str | Path) -> Path:
    output = Path(wind_file)
    return output.with_suffix(f"{output.suffix}.turbsim.json")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def _wind_generation_lock(output: Path) -> Iterator[None]:
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_suffix(f"{output.suffix}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _request_payload(*, output: Path, template: Path, rendered_input: str, settings: dict[str, Any]) -> dict[str, Any]:
    template_hash = _sha256_file(template)
    input_hash = hashlib.sha256(rendered_input.encode("utf-8")).hexdigest()
    payload: dict[str, Any] = {
        "version": 1,
        "wind_file": str(output),
        "template_file": str(template.resolve()),
        "template_sha256": template_hash,
        "rendered_input_sha256": input_hash,
        "settings": settings,
    }
    payload["request_fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _validated_existing_metadata(
    *,
    output: Path,
    input_file: Path,
    metadata_file: Path,
    request: dict[str, Any],
) -> dict[str, Any]:
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read TurbSim provenance metadata {metadata_file}: {exc}") from exc
    if metadata.get("request_fingerprint") != request.get("request_fingerprint"):
        differences: list[str] = []
        existing_settings = metadata.get("settings", {})
        requested_settings = request.get("settings", {})
        if isinstance(existing_settings, dict) and isinstance(requested_settings, dict):
            for key in sorted(set(existing_settings) | set(requested_settings)):
                existing_value = existing_settings.get(key, "<missing>")
                requested_value = requested_settings.get(key, "<missing>")
                if existing_value != requested_value:
                    differences.append(
                        f"settings.{key}: existing={existing_value!r}, requested={requested_value!r}"
                    )
        for key in ("wind_file", "template_file", "template_sha256", "rendered_input_sha256"):
            existing_value = metadata.get(key, "<missing>")
            requested_value = request.get(key, "<missing>")
            if existing_value != requested_value:
                differences.append(f"{key}: existing={existing_value!r}, requested={requested_value!r}")
        detail = ""
        if differences:
            detail = " Differences: " + "; ".join(differences[:12]) + "."
        raise ValueError(
            f"Existing TurbSim wind file {output} was generated for different settings. "
            "Use --force-generate-turbsim to regenerate it, or select a different wind path."
            f"{detail}"
        )
    if not input_file.is_file() or _sha256_file(input_file) != request["rendered_input_sha256"]:
        raise ValueError(f"Existing TurbSim input {input_file} does not match its requested immutable fingerprint.")
    expected_size = int(metadata.get("wind_file_bytes", -1))
    expected_hash = str(metadata.get("wind_file_sha256", ""))
    if output.stat().st_size != expected_size or _sha256_file(output) != expected_hash:
        raise ValueError(f"Existing TurbSim wind file {output} does not match {metadata_file}.")
    return metadata


def _write_generation_metadata(
    *,
    output: Path,
    input_file: Path,
    metadata_file: Path,
    request: dict[str, Any],
    executable: Path | None,
    provenance_status: str,
) -> None:
    executable_identity = None
    if executable is not None and executable.is_file():
        executable_identity = {
            "path": str(executable),
            "sha256": _sha256_file(executable),
            "size_bytes": executable.stat().st_size,
        }
    _atomic_write_json(
        metadata_file,
        {
            **request,
            "created_unix": time.time(),
            "provenance_status": provenance_status,
            "input_file": str(input_file),
            "input_file_sha256": _sha256_file(input_file),
            "wind_file_sha256": _sha256_file(output),
            "wind_file_bytes": output.stat().st_size,
            "executable": executable_identity,
        },
    )


def generate_turbsim_wind_file(
    *,
    wind_file: str | Path,
    settings: dict[str, Any],
    executable: str | Path | None = None,
    template_path: str | Path | None = None,
    vorlap_root: str | Path | None = None,
    force: bool = False,
    adopt_existing: bool = False,
) -> TurbSimRun:
    """Generate or safely reuse a fingerprinted TurbSim wind file."""
    output = Path(wind_file).expanduser().resolve()
    input_file = output.with_suffix(".inp")
    executable_path = resolve_turbsim_executable(explicit=executable, vorlap_root=vorlap_root)
    template = Path(template_path) if template_path is not None else default_template_path(vorlap_root)
    final_settings = dict(settings)
    final_settings.update(expected_output_flags(output))
    template = template.expanduser().resolve()
    rendered_input = render_turbsim_input(template.read_text(encoding="utf-8"), final_settings)
    metadata_file = turbsim_metadata_path(output)
    request = _request_payload(
        output=output,
        template=template,
        rendered_input=rendered_input,
        settings=final_settings,
    )

    with _wind_generation_lock(output):
        if output.is_file() and not force:
            if metadata_file.is_file():
                metadata = _validated_existing_metadata(
                    output=output,
                    input_file=input_file,
                    metadata_file=metadata_file,
                    request=request,
                )
                provenance_status = str(metadata.get("provenance_status", "generated"))
            elif adopt_existing:
                if not input_file.is_file() or _sha256_file(input_file) != request["rendered_input_sha256"]:
                    raise ValueError(
                        f"Cannot adopt legacy TurbSim output {output}: its existing .inp is missing or "
                        "does not exactly match the requested settings. Regenerate with --force-generate-turbsim."
                    )
                provenance_status = "adopted_existing_exact_input"
                _write_generation_metadata(
                    output=output,
                    input_file=input_file,
                    metadata_file=metadata_file,
                    request=request,
                    executable=executable_path,
                    provenance_status=provenance_status,
                )
            else:
                raise ValueError(
                    f"Refusing to reuse untracked TurbSim output {output}. Regenerate it with "
                    "--force-generate-turbsim, or explicitly adopt it with --adopt-existing-turbsim "
                    "when its existing .inp is known to be authoritative."
                )
            return TurbSimRun(
                wind_file=output,
                input_file=input_file,
                executable=executable_path,
                generated=False,
                skipped_existing=True,
                command=[],
                settings=final_settings,
                returncode=None,
                elapsed_s=None,
                metadata_file=metadata_file,
                provenance_status=provenance_status,
            )
        if executable_path is None:
            raise FileNotFoundError(
                "TurbSim executable not found. Run scripts/install_openfast_turbsim.sh, "
                "set TURBSIM_EXE, or put turbsim on PATH."
            )

        output.parent.mkdir(parents=True, exist_ok=True)
        command = [str(executable_path), input_file.name]
        start = time.time()
        with tempfile.TemporaryDirectory(prefix=f".{output.stem}.turbsim-", dir=output.parent) as temporary:
            temporary_dir = Path(temporary)
            temporary_input = temporary_dir / input_file.name
            temporary_input.write_text(rendered_input, encoding="utf-8")
            completed = subprocess.run(
                [str(executable_path), temporary_input.name],
                cwd=temporary_dir,
                check=True,
            )
            temporary_output = temporary_dir / output.name
            if not temporary_output.is_file() or temporary_output.stat().st_size <= 0:
                raise FileNotFoundError(
                    f"TurbSim completed but expected nonempty temporary wind file was not created: {temporary_output}"
                )
            os.replace(temporary_output, output)
            os.replace(temporary_input, input_file)
            for suffix in (".sum", ".ech"):
                auxiliary = temporary_dir / output.with_suffix(suffix).name
                if auxiliary.is_file():
                    os.replace(auxiliary, output.with_suffix(suffix))
        elapsed = time.time() - start
        provenance_status = "generated"
        _write_generation_metadata(
            output=output,
            input_file=input_file,
            metadata_file=metadata_file,
            request=request,
            executable=executable_path,
            provenance_status=provenance_status,
        )
        return TurbSimRun(
            wind_file=output,
            input_file=input_file,
            executable=executable_path,
            generated=True,
            skipped_existing=False,
            command=command,
            settings=final_settings,
            returncode=completed.returncode,
            elapsed_s=elapsed,
            metadata_file=metadata_file,
            provenance_status=provenance_status,
        )
