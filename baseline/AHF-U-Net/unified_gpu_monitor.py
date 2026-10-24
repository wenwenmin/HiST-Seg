from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Set

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def _parse_memory_value(text: str) -> float:
    value = text.strip().replace("MiB", "").replace("MB", "").strip()
    if not value or value.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return 0.0
    return float(value)


def query_gpu_process_memory(gpu_index: int = 0) -> Dict[int, float]:
    """Return {pid: used_gpu_memory_MiB} for one NVIDIA GPU."""
    field_variants = ("used_gpu_memory", "used_memory")
    for field in field_variants:
        command = [
            "nvidia-smi",
            "-i",
            str(int(gpu_index)),
            f"--query-compute-apps=pid,{field}",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(
                command,
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            continue

        result: Dict[int, float] = {}
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                memory = _parse_memory_value(parts[1])
            except (TypeError, ValueError):
                continue
            result[pid] = result.get(pid, 0.0) + memory
        return result

    raise RuntimeError(
        "Unable to query NVIDIA process memory. Confirm that nvidia-smi is "
        "available and that the selected GPU supports compute-process queries."
    )


def process_family(root_pid: int) -> Set[int]:
    pids: Set[int] = {int(root_pid)}
    if psutil is None:
        return pids
    try:
        process = psutil.Process(int(root_pid))
        for child in process.children(recursive=True):
            pids.add(int(child.pid))
    except (psutil.Error, OSError):
        pass
    return pids


@dataclass
class GPUMemorySample:
    peak_family_memory_mb: float
    peak_root_memory_mb: float
    samples: int


class ProcessTreeGPUMonitor:
    """Poll absolute NVIDIA memory used by a process and its descendants.

    This avoids mixing torch.cuda.max_memory_allocated(), reserved-memory
    values, and whole-GPU deltas. Each profiled phase should run in a clean
    process for the most reproducible result.
    """

    def __init__(
        self,
        root_pid: Optional[int] = None,
        gpu_index: int = 0,
        interval_s: float = 0.05,
    ) -> None:
        self.root_pid = int(root_pid if root_pid is not None else os.getpid())
        self.gpu_index = int(gpu_index)
        self.interval_s = float(interval_s)
        self.peak_family_memory_mb = 0.0
        self.peak_root_memory_mb = 0.0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None

    def _sample_once(self) -> None:
        memory_by_pid = query_gpu_process_memory(self.gpu_index)
        family = process_family(self.root_pid)
        family_memory = sum(memory_by_pid.get(pid, 0.0) for pid in family)
        root_memory = memory_by_pid.get(self.root_pid, 0.0)
        self.peak_family_memory_mb = max(self.peak_family_memory_mb, family_memory)
        self.peak_root_memory_mb = max(self.peak_root_memory_mb, root_memory)
        self.samples += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample_once()
            except BaseException as error:  # store and report on stop
                self._error = error
                return
            self._stop.wait(self.interval_s)

    def start(self) -> "ProcessTreeGPUMonitor":
        if self._thread is not None:
            raise RuntimeError("GPU monitor has already been started.")
        # nvidia-smi may not show a newly launched process immediately.
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> GPUMemorySample:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        try:
            self._sample_once()
        except BaseException as error:
            if self._error is None:
                self._error = error
        if self._error is not None and self.samples == 0:
            raise RuntimeError(f"GPU memory monitoring failed: {self._error}")
        return GPUMemorySample(
            peak_family_memory_mb=float(self.peak_family_memory_mb),
            peak_root_memory_mb=float(self.peak_root_memory_mb),
            samples=int(self.samples),
        )

    def __enter__(self) -> "ProcessTreeGPUMonitor":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
