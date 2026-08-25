"""Privacy-safe, bounded target-environment profiling.

The collector deliberately exposes only facts accepted by the existing
``EnvironmentProfile`` interface contract.  Native optional runtimes are
loaded in bounded child processes so a broken import cannot hang the caller.
"""

from __future__ import annotations

import csv
import io
import json
import os
import platform
import re
import selectors
import signal
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import (
    EnvironmentProfile,
    compute_environment_fingerprint,
    validate_environment_profile,
)

__all__ = ["build_environment_profile"]


COLLECTOR_ID = "yolozu_environment_profiler"
COLLECTOR_VERSION = "1"
MAX_PROBES = 32
PROBE_TIMEOUT_SECONDS = 5.0
PROFILE_TIMEOUT_SECONDS = 30.0
MAX_STDOUT_BYTES = 65_536
MAX_STDERR_BYTES = 65_536

_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_IP_RE = re.compile(r"(?:\A|[^0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?:\Z|[^0-9])")


@dataclass(frozen=True)
class _ProbeSpec:
    probe_id: str
    argv_candidates: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class _ProbeRun:
    status: str
    stdout: bytes = b""
    code: str | None = None


_ProbeRunner = Callable[[_ProbeSpec, float], _ProbeRun]


def _runtime_script(module_name: str) -> str:
    """Return a code-owned isolated runtime probe for one allowlisted module."""

    if module_name not in {"torch", "onnxruntime", "cv2", "tensorrt"}:
        raise ValueError("runtime module is not allowlisted")
    common = (
        "import json\n"
        "def emit(value):\n"
        " print(json.dumps(value, ensure_ascii=True, separators=(',', ':')))\n"
    )
    if module_name == "torch":
        body = r'''
try:
 import torch
except ModuleNotFoundError:
 emit({'status':'absent'})
except BaseException:
 emit({'status':'failed','code':'import_failed'})
else:
 try:
  providers=['cpu']
  accelerators=[]
  cuda_available=bool(torch.cuda.is_available())
  count=0
  if cuda_available:
   providers.append('cuda')
   count=int(torch.cuda.device_count())
   cuda_vendor='AMD' if getattr(getattr(torch,'version',None),'hip',None) is not None else 'NVIDIA'
   for index in range(max(0, min(count, 256))):
    props=torch.cuda.get_device_properties(index)
    accelerators.append({'kind':'gpu','vendor':cuda_vendor,'model':str(torch.cuda.get_device_name(index)),'memory_bytes':int(getattr(props,'total_memory',0) or 0)})
  mps=getattr(getattr(torch,'backends',None),'mps',None)
  if mps is not None and bool(mps.is_available()):
   providers.append('mps')
   accelerators.append({'kind':'gpu','vendor':'Apple','model':'Apple GPU','memory_bytes':None})
  cuda_version=getattr(getattr(torch,'version',None),'cuda',None)
  cudnn=getattr(getattr(torch,'backends',None),'cudnn',None)
  cudnn_version=int(cudnn.version()) if cudnn is not None and bool(cudnn.is_available()) else None
  emit({'status':'present','version':str(torch.__version__),'providers':providers,'accelerators':accelerators,'diagnostics':{'cuda_available':cuda_available,'cuda_version':str(cuda_version) if cuda_version is not None else None,'cudnn_version':cudnn_version,'device_count':count,'mps_built':bool(mps is not None and mps.is_built()),'mps_available':bool(mps is not None and mps.is_available())}})
 except BaseException:
  emit({'status':'failed','code':'probe_failed'})
'''
    elif module_name == "onnxruntime":
        body = r'''
try:
 import onnxruntime as runtime
except ModuleNotFoundError:
 emit({'status':'absent'})
except BaseException:
 emit({'status':'failed','code':'import_failed'})
else:
 try:
  mapping={'CPUExecutionProvider':'cpu','CUDAExecutionProvider':'cuda','TensorrtExecutionProvider':'tensorrt','CoreMLExecutionProvider':'coreml'}
  providers=[mapping[item] for item in runtime.get_available_providers() if item in mapping]
  emit({'status':'present','version':str(runtime.__version__),'providers':providers,'accelerators':[],'diagnostics':{'providers':list(runtime.get_available_providers())}})
 except BaseException:
  emit({'status':'failed','code':'probe_failed'})
'''
    elif module_name == "cv2":
        body = r'''
try:
 import cv2
except ModuleNotFoundError:
 emit({'status':'absent'})
except BaseException:
 emit({'status':'failed','code':'import_failed'})
else:
 try:
  providers=['cpu']
  cuda=getattr(cv2,'cuda',None)
  if cuda is not None and int(cuda.getCudaEnabledDeviceCount()) > 0:
   providers.append('cuda')
  emit({'status':'present','version':str(cv2.__version__),'providers':providers,'accelerators':[],'diagnostics':{'cuda_enabled_device_count':int(cuda.getCudaEnabledDeviceCount()) if cuda is not None else None}})
 except BaseException:
  emit({'status':'failed','code':'probe_failed'})
'''
    else:
        body = r'''
try:
 import tensorrt
except ModuleNotFoundError:
 emit({'status':'absent'})
except BaseException:
 emit({'status':'failed','code':'import_failed'})
else:
 try:
  emit({'status':'present','version':str(tensorrt.__version__),'providers':['tensorrt'],'accelerators':[],'diagnostics':{}})
 except BaseException:
  emit({'status':'failed','code':'probe_failed'})
'''
    return common + body


_PYTHON = str(Path(sys.executable).resolve())
_RUNTIME_SPECS = tuple(
    _ProbeSpec(
        probe_id=f"runtime_{module_name}",
        argv_candidates=((_PYTHON, "-I", "-c", _runtime_script(module_name)),),
    )
    for module_name in ("torch", "onnxruntime", "cv2", "tensorrt")
)

_RUNTIME_COMMAND_SPECS = (
    _ProbeSpec(
        "runtime_trtexec",
        (("/usr/bin/trtexec", "--version"), ("/usr/local/bin/trtexec", "--version")),
    ),
)

_DARWIN_SPECS = (
    _ProbeSpec("darwin_cpu_model", (("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"),)),
    _ProbeSpec("darwin_physical_cores", (("/usr/sbin/sysctl", "-n", "hw.physicalcpu"),)),
    _ProbeSpec("darwin_total_memory", (("/usr/sbin/sysctl", "-n", "hw.memsize"),)),
    _ProbeSpec(
        "darwin_accelerators",
        (("/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"),),
    ),
    _ProbeSpec("darwin_power_mode", (("/usr/bin/pmset", "-g"),)),
)

_LINUX_SPECS = (
    _ProbeSpec("linux_cpu", (("/usr/bin/lscpu", "--json"), ("/bin/lscpu", "--json"))),
    _ProbeSpec(
        "linux_accelerators",
        (("/usr/bin/lshw", "-json", "-class", "display"), ("/usr/sbin/lshw", "-json", "-class", "display")),
    ),
    _ProbeSpec(
        "nvidia_accelerators",
        (
            (
                "/usr/bin/nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ),
            (
                "/usr/local/bin/nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ),
        ),
    ),
    _ProbeSpec(
        "linux_power_mode",
        (("/usr/bin/powerprofilesctl", "get"), ("/bin/powerprofilesctl", "get")),
    ),
)

_LIMIT_TEST_SPECS = (
    _ProbeSpec(
        "limit_test_timeout",
        ((_PYTHON, "-I", "-c", "import time; time.sleep(60)"),),
    ),
    _ProbeSpec(
        "limit_test_output",
        ((_PYTHON, "-I", "-c", "import sys; sys.stdout.write('x'*70000)"),),
    ),
    _ProbeSpec(
        "limit_test_process_tree",
        (
            (
                _PYTHON,
                "-I",
                "-c",
                "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); time.sleep(60)",
            ),
        ),
    ),
)

_ALL_CODE_OWNED_SPECS = (
    _DARWIN_SPECS
    + _LINUX_SPECS
    + _RUNTIME_SPECS
    + _RUNTIME_COMMAND_SPECS
    + _LIMIT_TEST_SPECS
)
if len(_ALL_CODE_OWNED_SPECS) > MAX_PROBES:
    raise RuntimeError("code-owned probe catalog exceeds limit")
_CODE_OWNED_ARGV = {
    spec.probe_id: frozenset(spec.argv_candidates) for spec in _ALL_CODE_OWNED_SPECS
}


def _specs_for_system(system: str) -> tuple[_ProbeSpec, ...]:
    platform_specs: tuple[_ProbeSpec, ...]
    if system == "Darwin":
        platform_specs = _DARWIN_SPECS
    elif system == "Linux":
        platform_specs = _LINUX_SPECS
    else:
        platform_specs = ()
    specs = platform_specs + _RUNTIME_SPECS + _RUNTIME_COMMAND_SPECS
    if len(specs) > MAX_PROBES:
        raise RuntimeError("code-owned probe catalog exceeds limit")
    return specs


def _resolve_argv(spec: _ProbeSpec) -> tuple[str, ...] | None:
    if not _SAFE_ID_RE.fullmatch(spec.probe_id):
        raise ValueError("invalid code-owned probe id")
    allowed = _CODE_OWNED_ARGV.get(spec.probe_id)
    if allowed is None or frozenset(spec.argv_candidates) != allowed:
        raise ValueError("probe command or arguments are not code-owned")
    for candidate in spec.argv_candidates:
        if not candidate or not Path(candidate[0]).is_absolute():
            raise ValueError("probe executable must be an absolute code-owned path")
        try:
            if Path(candidate[0]).is_file() and os.access(candidate[0], os.X_OK):
                return candidate
        except OSError:
            continue
    return None


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            # The child may already have exited; termination is best effort.
            pass
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - Windows CI is not currently supported.
            process.terminate()
        process.wait(timeout=0.2)
        return
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
        # Fall through to the stronger termination attempt below.
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover
            process.kill()
    except (OSError, ProcessLookupError):
        # The process group may have exited between poll and kill.
        pass
    try:
        process.wait(timeout=0.2)
    except (OSError, subprocess.TimeoutExpired):
        # No further safe cleanup is available after SIGKILL.
        pass


def _run_bounded_probe(spec: _ProbeSpec, timeout_seconds: float) -> _ProbeRun:
    """Run one exact allowlisted probe without a shell or inherited environment."""

    argv = _resolve_argv(spec)
    if argv is None:
        return _ProbeRun("unsupported", code="executable_unavailable")
    timeout_seconds = max(0.0, min(float(timeout_seconds), PROBE_TIMEOUT_SECONDS))
    if timeout_seconds <= 0.0:
        return _ProbeRun("failed", code="total_deadline")

    env = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    creationflags = 0
    if os.name == "nt":  # pragma: no cover
        creationflags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    try:
        process = subprocess.Popen(
            list(argv),
            cwd=str(Path(os.path.abspath(os.sep))),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=(os.name == "posix"),
            creationflags=creationflags,
        )
    except (OSError, ValueError):
        return _ProbeRun("failed", code="spawn_failed")

    stdout = bytearray()
    stderr = bytearray()
    selector = selectors.DefaultSelector()
    assert process.stdout is not None
    assert process.stderr is not None
    for stream, label in ((process.stdout, "stdout"), (process.stderr, "stderr")):
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, label)
    deadline = time.monotonic() + timeout_seconds
    failure_code: str | None = None
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                failure_code = "timeout"
                break
            events = selector.select(min(remaining, 0.05))
            if not events and process.poll() is not None:
                events = [(key, selectors.EVENT_READ) for key in selector.get_map().values()]
            for key, _ in events:
                try:
                    chunk = os.read(key.fileobj.fileno(), 8192)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                target = stdout if key.data == "stdout" else stderr
                limit = MAX_STDOUT_BYTES if key.data == "stdout" else MAX_STDERR_BYTES
                if len(target) + len(chunk) > limit:
                    failure_code = f"{key.data}_limit"
                    break
                target.extend(chunk)
            if failure_code is not None:
                break
        if failure_code is not None:
            _terminate_process_group(process)
            return _ProbeRun("failed", code=failure_code)
        try:
            return_code = process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            return _ProbeRun("failed", code="timeout")
        if return_code != 0:
            return _ProbeRun("failed", code="subprocess_exit")
        return _ProbeRun("ok", stdout=bytes(stdout))
    finally:
        selector.close()
        for stream in (process.stdout, process.stderr):
            try:
                stream.close()
            except OSError:
                # Streams are local cleanup handles and may already be closed.
                pass


def _safe_text(value: Any, *, maximum_bytes: int = 256) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text.encode("utf-8")) > maximum_bytes:
        return None
    if any(unicodedata.category(character).startswith("C") for character in text):
        return None
    if "/" in text or "\\" in text or _UUID_RE.search(text) or _IP_RE.search(text):
        return None
    return text


def _positive_int(value: Any, *, maximum: int = 9_223_372_036_854_775_807) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if 1 <= parsed <= maximum else None


def _json_payload(run: _ProbeRun) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    if run.status != "ok":
        return None, run.code or run.status
    try:
        text = run.stdout.decode("utf-8", errors="strict")
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        return None, "malformed_output"
    if not isinstance(value, (dict, list)):
        return None, "malformed_output"
    return value, None


def _text_payload(run: _ProbeRun) -> tuple[str | None, str | None]:
    if run.status != "ok":
        return None, run.code or run.status
    try:
        return run.stdout.decode("utf-8", errors="strict"), None
    except UnicodeDecodeError:
        return None, "malformed_output"


def _issue(probe_id: str, status: str, code: str) -> dict[str, str]:
    normalized_status = "unsupported" if status == "unsupported" else "failed"
    normalized_code = code if _SAFE_ID_RE.fullmatch(code) else "probe_failed"
    return {"probe_id": probe_id, "status": normalized_status, "code": normalized_code}


def _failed_fact(run: _ProbeRun) -> dict[str, str]:
    return {"probe_status": "unsupported" if run.status == "unsupported" else "failed"}


def _collect_runs(
    specs: Sequence[_ProbeSpec],
    *,
    runner: _ProbeRunner,
    monotonic: Callable[[], float],
) -> dict[str, _ProbeRun]:
    started = monotonic()
    runs: dict[str, _ProbeRun] = {}
    for spec in specs:
        remaining = PROFILE_TIMEOUT_SECONDS - (monotonic() - started)
        if remaining <= 0.0:
            runs[spec.probe_id] = _ProbeRun("failed", code="total_deadline")
            continue
        try:
            result = runner(spec, min(PROBE_TIMEOUT_SECONDS, remaining))
        except Exception:
            result = _ProbeRun("failed", code="probe_exception")
        if not isinstance(result, _ProbeRun) or result.status not in {
            "ok",
            "unsupported",
            "failed",
        }:
            result = _ProbeRun("failed", code="invalid_probe_result")
        runs[spec.probe_id] = result
    return runs


def _platform_record(platform_facts: Mapping[str, Any]) -> dict[str, Any]:
    name = _safe_text(platform_facts.get("system"), maximum_bytes=128)
    version = _safe_text(platform_facts.get("release"), maximum_bytes=128)
    architecture = _safe_text(platform_facts.get("machine"), maximum_bytes=128)
    if name is None or version is None or architecture is None:
        return {"probe_status": "failed"}
    return {
        "probe_status": "present",
        "name": name,
        "version": version,
        "architecture": architecture,
    }


def _parse_lscpu(run: _ProbeRun) -> tuple[dict[str, str], str | None]:
    payload, error = _json_payload(run)
    if error is not None or not isinstance(payload, dict):
        return {}, error or "malformed_output"
    items = payload.get("lscpu")
    if not isinstance(items, list) or len(items) > 256:
        return {}, "malformed_output"
    facts: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict) or set(item) - {"field", "data"}:
            continue
        field = item.get("field")
        data = item.get("data")
        if isinstance(field, str) and isinstance(data, (str, int)):
            facts[field.rstrip(":").strip()] = str(data).strip()
    return facts, None


def _cpu_record(
    *,
    system: str,
    platform_facts: Mapping[str, Any],
    runs: Mapping[str, _ProbeRun],
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    fallback_model = _safe_text(platform_facts.get("processor"))
    model = fallback_model
    logical = _positive_int(platform_facts.get("logical_cores"), maximum=1_048_576)
    logical_fact: dict[str, Any] = (
        {"probe_status": "present", "value": logical}
        if logical is not None
        else {"probe_status": "failed"}
    )
    physical_fact: dict[str, Any] = {"probe_status": "unsupported"}

    if system == "Darwin":
        model_run = runs["darwin_cpu_model"]
        model_text, model_error = _text_payload(model_run)
        parsed_model = _safe_text(model_text) if model_text is not None else None
        if parsed_model is not None:
            model = parsed_model
        else:
            issues.append(
                _issue(
                    "darwin_cpu_model",
                    model_run.status,
                    model_error or "unsafe_output",
                )
            )
        core_run = runs["darwin_physical_cores"]
        core_text, core_error = _text_payload(core_run)
        physical = _positive_int(core_text.strip() if core_text is not None else None, maximum=1_048_576)
        if physical is not None:
            physical_fact = {"probe_status": "present", "value": physical}
        else:
            physical_fact = _failed_fact(core_run)
            issues.append(
                _issue(
                    "darwin_physical_cores",
                    core_run.status,
                    core_error or "malformed_output",
                )
            )
    elif system == "Linux":
        cpu_run = runs["linux_cpu"]
        facts, cpu_error = _parse_lscpu(cpu_run)
        parsed_model = _safe_text(facts.get("Model name"))
        if parsed_model is not None:
            model = parsed_model
        elif "Model name" in facts:
            issues.append(_issue("linux_cpu", "failed", "unsafe_output"))
        lscpu_logical = _positive_int(facts.get("CPU(s)"), maximum=1_048_576)
        if lscpu_logical is not None:
            logical_fact = {"probe_status": "present", "value": lscpu_logical}
        cores_per_socket = _positive_int(facts.get("Core(s) per socket"), maximum=1_048_576)
        sockets = _positive_int(facts.get("Socket(s)"), maximum=1_048_576)
        if cores_per_socket is not None and sockets is not None:
            physical = cores_per_socket * sockets
            if physical <= 1_048_576:
                physical_fact = {"probe_status": "present", "value": physical}
        if cpu_error is not None:
            issues.append(_issue("linux_cpu", cpu_run.status, cpu_error))

    if model is None:
        issues.append(_issue("cpu", "failed", "model_unavailable"))
        return {"probe_status": "failed"}
    return {
        "probe_status": "present",
        "model": model,
        "logical_cores": logical_fact,
        "physical_cores": physical_fact,
    }


def _total_memory_record(
    *,
    system: str,
    platform_facts: Mapping[str, Any],
    runs: Mapping[str, _ProbeRun],
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    injected = _positive_int(platform_facts.get("total_memory_bytes"))
    if injected is not None:
        return {"probe_status": "present", "value_bytes": injected}
    if system == "Darwin":
        run = runs["darwin_total_memory"]
        text, error = _text_payload(run)
        value = _positive_int(text.strip() if text is not None else None)
        if value is not None:
            return {"probe_status": "present", "value_bytes": value}
        issues.append(_issue("darwin_total_memory", run.status, error or "malformed_output"))
        return _failed_fact(run)
    if system == "Linux":
        try:
            pages = _positive_int(os.sysconf("SC_PHYS_PAGES"))
            page_size = _positive_int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError):
            issues.append(_issue("linux_total_memory", "failed", "sysconf_failed"))
            return {"probe_status": "failed"}
        if pages is not None and page_size is not None:
            total = pages * page_size
            if total <= 9_223_372_036_854_775_807:
                return {"probe_status": "present", "value_bytes": total}
        issues.append(_issue("linux_total_memory", "failed", "sysconf_failed"))
        return {"probe_status": "failed"}
    return {"probe_status": "unsupported"}


def _memory_from_display_text(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        return {"probe_status": "unsupported"}
    match = re.fullmatch(r"\s*([0-9]+)\s*(MB|GB)\s*", value, flags=re.IGNORECASE)
    if match is None:
        return {"probe_status": "unsupported"}
    amount = _positive_int(match.group(1))
    if amount is None:
        return {"probe_status": "failed"}
    multiplier = 1_048_576 if match.group(2).upper() == "MB" else 1_073_741_824
    value_bytes = amount * multiplier
    if value_bytes > 9_223_372_036_854_775_807:
        return {"probe_status": "failed"}
    return {"probe_status": "present", "value_bytes": value_bytes}


def _parse_darwin_accelerators(run: _ProbeRun) -> tuple[list[dict[str, Any]], str | None]:
    payload, error = _json_payload(run)
    if error is not None or not isinstance(payload, dict):
        return [], error or "malformed_output"
    displays = payload.get("SPDisplaysDataType")
    if not isinstance(displays, list) or len(displays) > 32:
        return [], "malformed_output"
    records: list[dict[str, Any]] = []
    for index, item in enumerate(displays):
        if not isinstance(item, dict):
            return [], "malformed_output"
        model = _safe_text(item.get("sppci_model") or item.get("_name"))
        if model is None:
            return [], "unsafe_output"
        lowered_model = model.lower()
        if "apple" in lowered_model:
            vendor = "Apple"
        elif "nvidia" in lowered_model or "geforce" in lowered_model:
            vendor = "NVIDIA"
        elif "amd" in lowered_model or "radeon" in lowered_model:
            vendor = "AMD"
        elif "intel" in lowered_model:
            vendor = "Intel"
        else:
            return [], "vendor_unknown"
        memory = _memory_from_display_text(
            item.get("spdisplays_vram") or item.get("spdisplays_vram_shared")
        )
        records.append(
            {
                "accelerator_id": f"darwin_display_{index}",
                "probe_status": "present",
                "kind": "gpu",
                "vendor": vendor,
                "model": model,
                "device_count": 1,
                "memory": memory,
            }
        )
    if not records:
        return [{"accelerator_id": "host_accelerator_inventory", "probe_status": "absent"}], None
    return records, None


def _parse_linux_accelerators(run: _ProbeRun) -> tuple[list[dict[str, Any]], str | None]:
    payload, error = _json_payload(run)
    if error is not None:
        return [], error
    items = payload if isinstance(payload, list) else [payload]
    if len(items) > 32:
        return [], "malformed_output"
    records: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            return [], "malformed_output"
        if item.get("class") != "display":
            continue
        model = _safe_text(item.get("product") or item.get("description"))
        vendor = _safe_text(item.get("vendor"), maximum_bytes=128)
        if model is None or vendor is None:
            return [], "unsafe_output"
        memory_value = _positive_int(item.get("size"))
        memory = (
            {"probe_status": "present", "value_bytes": memory_value}
            if memory_value is not None
            else {"probe_status": "unsupported"}
        )
        records.append(
            {
                "accelerator_id": f"linux_display_{len(records)}",
                "probe_status": "present",
                "kind": "gpu",
                "vendor": vendor,
                "model": model,
                "device_count": 1,
                "memory": memory,
            }
        )
    if not records:
        return [{"accelerator_id": "host_accelerator_inventory", "probe_status": "absent"}], None
    return records, None


def _parse_nvidia_accelerators(run: _ProbeRun) -> tuple[list[dict[str, Any]], str | None]:
    text, error = _text_payload(run)
    if error is not None or text is None:
        return [], error or "malformed_output"
    try:
        rows = list(csv.reader(io.StringIO(text), strict=True))
    except csv.Error:
        return [], "malformed_output"
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if len(rows) > 32:
        return [], "malformed_output"
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if len(row) != 2:
            return [], "malformed_output"
        model = _safe_text(row[0])
        memory_mib = _positive_int(row[1].strip())
        if model is None or memory_mib is None:
            return [], "unsafe_output"
        memory_bytes = memory_mib * 1_048_576
        if memory_bytes > 9_223_372_036_854_775_807:
            return [], "malformed_output"
        records.append(
            {
                "accelerator_id": f"nvidia_{index}",
                "probe_status": "present",
                "kind": "gpu",
                "vendor": "NVIDIA",
                "model": model,
                "device_count": 1,
                "memory": {
                    "probe_status": "present",
                    "value_bytes": memory_bytes,
                },
            }
        )
    if not records:
        return [{"accelerator_id": "nvidia_inventory", "probe_status": "absent"}], None
    return records, None


def _sanitize_runtime_diagnostics(module_name: str, value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if module_name == "torch":
        allowed = {
            "cuda_available",
            "cuda_version",
            "cudnn_version",
            "device_count",
            "mps_built",
            "mps_available",
        }
        if set(value) - allowed:
            return None
        for key in ("cuda_available", "mps_built", "mps_available"):
            if key in value and not isinstance(value[key], bool):
                return None
        cuda_version = value.get("cuda_version")
        if cuda_version is not None and _safe_text(cuda_version, maximum_bytes=128) is None:
            return None
        cudnn_version = value.get("cudnn_version")
        if cudnn_version is not None and _positive_int(cudnn_version) is None:
            return None
        device_count = value.get("device_count")
        if device_count is not None and (
            isinstance(device_count, bool)
            or not isinstance(device_count, int)
            or not 0 <= device_count <= 256
        ):
            return None
        return {
            "cuda_available": bool(value.get("cuda_available", False)),
            "cuda_version": cuda_version,
            "cudnn_version": cudnn_version,
            "device_count": int(device_count or 0),
            "mps_built": bool(value.get("mps_built", False)),
            "mps_available": bool(value.get("mps_available", False)),
        }
    if module_name == "onnxruntime":
        if set(value) - {"providers"}:
            return None
        providers = value.get("providers", [])
        allowed = {
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "CoreMLExecutionProvider",
        }
        if (
            not isinstance(providers, list)
            or len(providers) > 32
            or any(provider not in allowed for provider in providers)
        ):
            return None
        return {"providers": list(dict.fromkeys(providers))}
    if module_name == "cv2":
        if set(value) - {"cuda_enabled_device_count"}:
            return None
        count = value.get("cuda_enabled_device_count")
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 256
        ):
            return None
        return {"cuda_enabled_device_count": count}
    return {} if not value else None


def _runtime_records(
    runs: Mapping[str, _ProbeRun],
    *,
    issues: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    runtimes: list[dict[str, Any]] = [
        {
            "runtime_id": "python",
            "probe_status": "present",
            "version": platform.python_version(),
            "provider_ids": ["cpu"],
        }
    ]
    accelerators: list[dict[str, Any]] = []
    diagnostics: dict[str, dict[str, Any]] = {}
    allowed_providers = {"cpu", "cuda", "mps", "coreml", "tensorrt"}
    for module_name in ("torch", "onnxruntime", "cv2", "tensorrt"):
        probe_id = f"runtime_{module_name}"
        run = runs[probe_id]
        payload, error = _json_payload(run)
        if error is not None or not isinstance(payload, dict):
            runtimes.append({"runtime_id": module_name, **_failed_fact(run)})
            issues.append(_issue(probe_id, run.status, error or "malformed_output"))
            continue
        if set(payload) - {
            "status",
            "version",
            "providers",
            "accelerators",
            "diagnostics",
            "code",
        }:
            runtimes.append({"runtime_id": module_name, "probe_status": "failed"})
            issues.append(_issue(probe_id, "failed", "unexpected_output"))
            continue
        status = payload.get("status")
        if status == "absent":
            runtimes.append({"runtime_id": module_name, "probe_status": "absent"})
            continue
        if status == "failed":
            code = payload.get("code")
            runtimes.append({"runtime_id": module_name, "probe_status": "failed"})
            issues.append(_issue(probe_id, "failed", code if isinstance(code, str) else "probe_failed"))
            continue
        version = _safe_text(payload.get("version"), maximum_bytes=128)
        providers = payload.get("providers")
        if (
            status != "present"
            or version is None
            or not isinstance(providers, list)
            or len(providers) > 32
            or any(provider not in allowed_providers for provider in providers)
            or len(set(providers)) != len(providers)
        ):
            runtimes.append({"runtime_id": module_name, "probe_status": "failed"})
            issues.append(_issue(probe_id, "failed", "malformed_output"))
            continue
        runtimes.append(
            {
                "runtime_id": module_name,
                "probe_status": "present",
                "version": version,
                "provider_ids": sorted(providers, key=lambda item: item.encode("utf-8")),
            }
        )
        runtime_diagnostics = _sanitize_runtime_diagnostics(
            module_name, payload.get("diagnostics", {})
        )
        if runtime_diagnostics is None:
            issues.append(_issue(probe_id, "failed", "malformed_diagnostics"))
            runtime_diagnostics = {}
        diagnostics[module_name] = {
            "version": version,
            "provider_ids": sorted(providers, key=lambda item: item.encode("utf-8")),
            "details": runtime_diagnostics,
        }
        runtime_accelerators = payload.get("accelerators")
        if not isinstance(runtime_accelerators, list) or len(runtime_accelerators) > 32:
            issues.append(_issue(probe_id, "failed", "malformed_accelerators"))
            continue
        for item in runtime_accelerators:
            if not isinstance(item, dict) or set(item) != {"kind", "vendor", "model", "memory_bytes"}:
                issues.append(_issue(probe_id, "failed", "malformed_accelerators"))
                break
            kind = item.get("kind")
            vendor = _safe_text(item.get("vendor"), maximum_bytes=128)
            model = _safe_text(item.get("model"))
            if kind != "gpu" or vendor is None or model is None:
                issues.append(_issue(probe_id, "failed", "unsafe_accelerator_output"))
                break
            memory_bytes = _positive_int(item.get("memory_bytes"))
            accelerators.append(
                {
                    "accelerator_id": f"{module_name}_accelerator_{len(accelerators)}",
                    "probe_status": "present",
                    "kind": "gpu",
                    "vendor": vendor,
                    "model": model,
                    "device_count": 1,
                    "memory": (
                        {"probe_status": "present", "value_bytes": memory_bytes}
                        if memory_bytes is not None
                        else {"probe_status": "unsupported"}
                    ),
                }
            )
    trtexec_run = runs["runtime_trtexec"]
    trtexec_text, trtexec_error = _text_payload(trtexec_run)
    trtexec_lines = (
        [line.strip() for line in trtexec_text.splitlines() if line.strip()]
        if trtexec_text is not None
        else []
    )
    trtexec_version = _safe_text(trtexec_lines[0], maximum_bytes=128) if trtexec_lines else None
    if trtexec_version is not None:
        runtimes.append(
            {
                "runtime_id": "trtexec",
                "probe_status": "present",
                "version": trtexec_version,
                "provider_ids": ["tensorrt"],
            }
        )
        diagnostics["trtexec"] = {"version": trtexec_version, "details": {}}
    else:
        runtimes.append({"runtime_id": "trtexec", **_failed_fact(trtexec_run)})
        issues.append(
            _issue(
                "runtime_trtexec",
                trtexec_run.status,
                trtexec_error or "malformed_output",
            )
        )
    return runtimes, accelerators, diagnostics


def _accelerator_records(
    *,
    system: str,
    runs: Mapping[str, _ProbeRun],
    runtime_accelerators: Sequence[dict[str, Any]],
    issues: list[dict[str, str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if system == "Darwin":
        run = runs["darwin_accelerators"]
        parsed, error = _parse_darwin_accelerators(run)
        if error is None:
            records.extend(parsed)
        else:
            records.append({"accelerator_id": "host_accelerator_inventory", **_failed_fact(run)})
            issues.append(_issue("darwin_accelerators", run.status, error))
    elif system == "Linux":
        host_run = runs["linux_accelerators"]
        host_records, host_error = _parse_linux_accelerators(host_run)
        if host_error is None:
            records.extend(host_records)
        else:
            records.append({"accelerator_id": "host_accelerator_inventory", **_failed_fact(host_run)})
            issues.append(_issue("linux_accelerators", host_run.status, host_error))
        nvidia_run = runs["nvidia_accelerators"]
        nvidia_records, nvidia_error = _parse_nvidia_accelerators(nvidia_run)
        if nvidia_error is None:
            records.extend(nvidia_records)
        else:
            records.append({"accelerator_id": "nvidia_inventory", **_failed_fact(nvidia_run)})
            issues.append(_issue("nvidia_accelerators", nvidia_run.status, nvidia_error))
    else:
        records.append(
            {"accelerator_id": "host_accelerator_inventory", "probe_status": "unsupported"}
        )
        issues.append(_issue("host_accelerator_inventory", "unsupported", "unsupported_os"))

    records.extend(runtime_accelerators)
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        candidate_id = str(record["accelerator_id"])
        if candidate_id not in by_id:
            by_id[candidate_id] = record
    if len(by_id) > 32:
        issues.append(_issue("accelerator_inventory", "failed", "result_limit"))
        return [{"accelerator_id": "host_accelerator_inventory", "probe_status": "failed"}]
    return list(by_id.values())


def _power_record(
    *,
    system: str,
    runs: Mapping[str, _ProbeRun],
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    if system == "Darwin":
        run = runs["darwin_power_mode"]
        text, error = _text_payload(run)
        if text is not None and re.search(r"(?m)^\s*lowpowermode\s+1\s*$", text):
            return {"probe_status": "present", "mode": "low_power"}
        issues.append(_issue("darwin_power_mode", run.status, error or "mode_unknown"))
        return _failed_fact(run) if run.status != "ok" else {"probe_status": "failed"}
    if system == "Linux":
        run = runs["linux_power_mode"]
        text, error = _text_payload(run)
        mapping = {"balanced": "balanced", "performance": "performance", "power-saver": "power_saver"}
        token = text.strip() if text is not None else None
        if token in mapping:
            return {"probe_status": "present", "mode": mapping[token]}
        issues.append(_issue("linux_power_mode", run.status, error or "mode_unknown"))
        return _failed_fact(run) if run.status != "ok" else {"probe_status": "failed"}
    return {"probe_status": "unsupported"}


def _default_platform_facts() -> dict[str, Any]:
    try:
        logical_cores = os.cpu_count()
    except OSError:
        logical_cores = None
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cores": logical_cores,
    }


def _build_environment_observation(
    *,
    probe_runner: _ProbeRunner | None = None,
    platform_facts: Mapping[str, Any] | None = None,
    collected_at: str | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> tuple[EnvironmentProfile, dict[str, dict[str, Any]]]:

    facts = dict(_default_platform_facts() if platform_facts is None else platform_facts)
    system = facts.get("system") if isinstance(facts.get("system"), str) else "Unsupported"
    specs = _specs_for_system(system)
    runs = _collect_runs(specs, runner=probe_runner or _run_bounded_probe, monotonic=monotonic)
    issues: list[dict[str, str]] = []

    os_record = _platform_record(facts)
    if os_record["probe_status"] != "present":
        issues.append(_issue("os", "failed", "unsafe_or_unavailable"))
    cpu = _cpu_record(system=system, platform_facts=facts, runs=runs, issues=issues)
    total_memory = _total_memory_record(
        system=system,
        platform_facts=facts,
        runs=runs,
        issues=issues,
    )
    runtimes, runtime_accelerators, runtime_diagnostics = _runtime_records(
        runs, issues=issues
    )
    accelerators = _accelerator_records(
        system=system,
        runs=runs,
        runtime_accelerators=runtime_accelerators,
        issues=issues,
    )
    power = _power_record(system=system, runs=runs, issues=issues)

    unique_issues: dict[tuple[str, str], dict[str, str]] = {}
    for item in issues:
        unique_issues.setdefault((item["probe_id"], item["code"]), item)
    issue_records = list(unique_issues.values())[:32]
    timestamp = collected_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record: dict[str, Any] = {
        "schema_version": 1,
        "collector_id": COLLECTOR_ID,
        "collector_version": COLLECTOR_VERSION,
        "collected_at": timestamp,
        "os": os_record,
        "cpu": cpu,
        "total_memory": total_memory,
        "accelerators": accelerators,
        "runtimes": runtimes,
        "power_performance_mode": power,
        "probe_issues": issue_records,
        "environment_fingerprint": "0" * 64,
    }
    record["environment_fingerprint"] = compute_environment_fingerprint(record)
    return validate_environment_profile(record), runtime_diagnostics


def build_environment_profile(
    *,
    probe_runner: _ProbeRunner | None = None,
    platform_facts: Mapping[str, Any] | None = None,
    collected_at: str | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> EnvironmentProfile:
    """Collect and validate one bounded privacy-safe environment profile.

    ``probe_runner`` and ``platform_facts`` exist for deterministic tests. The
    caller cannot add a command: every runner receives only the fixed catalog.
    """

    profile, _ = _build_environment_observation(
        probe_runner=probe_runner,
        platform_facts=platform_facts,
        collected_at=collected_at,
        monotonic=monotonic,
    )
    return profile
