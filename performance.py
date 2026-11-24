"""
Performance monitoring utilities for the On-Device Vision AI application.

Tracks inference times, session statistics, and hardware resource usage
(CPU, system RAM, and optional CUDA GPU).
"""

from __future__ import annotations

import logging
import platform
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Lightweight tracker for inference latency and hardware metrics.

    Usage::

        monitor = PerformanceMonitor()
        monitor.record_inference(0.35)
        stats = monitor.get_stats()
    """

    def __init__(self) -> None:
        self.model_load_time: float = 0.0
        self.inference_times: list[float] = []
        self.total_images_processed: int = 0
        self.session_start_time: float = time.time()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_inference(self, time_seconds: float) -> None:
        """Record a single inference duration.

        Args:
            time_seconds: Wall-clock inference time in seconds.
        """
        self.inference_times.append(time_seconds)
        self.total_images_processed += 1

    # ------------------------------------------------------------------
    # Aggregated statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return aggregated session statistics.

        Returns:
            dict with ``avg_inference_time``, ``min_inference_time``,
            ``max_inference_time``, ``total_images_processed``, and
            ``session_duration`` (seconds).
        """
        n = len(self.inference_times)
        if n > 0:
            avg_time = sum(self.inference_times) / n
            min_time = min(self.inference_times)
            max_time = max(self.inference_times)
        else:
            avg_time = min_time = max_time = 0.0

        return {
            "avg_inference_time": round(avg_time, 4),
            "min_inference_time": round(min_time, 4),
            "max_inference_time": round(max_time, 4),
            "total_images_processed": self.total_images_processed,
            "session_duration": round(time.time() - self.session_start_time, 2),
            "model_load_time": round(self.model_load_time, 4),
        }

    def get_metrics(self) -> dict[str, Any]:
        """Return combined metrics dictionary for the UI dashboard."""
        stats = self.get_stats()
        dev_info = self.get_device_info()
        mem_info = self.get_memory_usage()

        device_name = "CPU"
        if dev_info.get("device_type") == "cuda":
            gpu_info = dev_info.get("gpu", {})
            device_name = gpu_info.get("name", "CUDA GPU")

        memory_mb = None
        if "system" in mem_info:
            used_gb = mem_info["system"].get("used_gb")
            if used_gb is not None:
                memory_mb = used_gb * 1024

        return {
            "model_load_time": self.model_load_time if self.model_load_time > 0 else None,
            "avg_inference_time": stats["avg_inference_time"] if stats["total_images_processed"] > 0 else None,
            "total_images_processed": self.total_images_processed,
            "device": device_name,
            "memory_usage_mb": memory_mb,
            "inference_history": list(self.inference_times),
            **stats,
        }

    # ------------------------------------------------------------------
    # Device information
    # ------------------------------------------------------------------

    @staticmethod
    def get_device_info() -> dict[str, Any]:
        """Return information about the compute device.

        Includes CPU details and CUDA GPU details when available.

        Returns:
            dict with ``device_type``, ``cpu`` sub-dict, and optionally
            ``gpu`` sub-dict.
        """
        info: dict[str, Any] = {
            "device_type": "cuda" if torch.cuda.is_available() else "cpu",
            "cpu": {
                "processor": platform.processor() or "unknown",
                "architecture": platform.machine(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            },
        }

        # Enrich CPU info with psutil if available
        try:
            import psutil  # type: ignore[import-untyped]

            info["cpu"]["physical_cores"] = psutil.cpu_count(logical=False)
            info["cpu"]["logical_cores"] = psutil.cpu_count(logical=True)
            freq = psutil.cpu_freq()
            if freq is not None:
                info["cpu"]["max_freq_mhz"] = round(freq.max, 1) if freq.max else None
                info["cpu"]["current_freq_mhz"] = round(freq.current, 1) if freq.current else None
        except ImportError:
            logger.debug("psutil not installed – CPU core/freq info unavailable")

        # GPU info
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(props.total_mem / (1024 ** 3), 2),
                "multi_processor_count": props.multi_processor_count,
                "cuda_version": torch.version.cuda or "unknown",
            }

        info["torch_version"] = torch.__version__

        return info

    # ------------------------------------------------------------------
    # Memory usage
    # ------------------------------------------------------------------

    @staticmethod
    def get_memory_usage() -> dict[str, Any]:
        """Return current memory usage for system RAM and (optionally) GPU.

        Returns:
            dict with ``system`` and optionally ``gpu`` sub-dicts, each
            containing ``used_gb``, ``total_gb``, and ``percent``.
        """
        mem: dict[str, Any] = {}

        # System RAM via psutil
        try:
            import psutil  # type: ignore[import-untyped]

            vm = psutil.virtual_memory()
            mem["system"] = {
                "total_gb": round(vm.total / (1024 ** 3), 2),
                "used_gb": round(vm.used / (1024 ** 3), 2),
                "available_gb": round(vm.available / (1024 ** 3), 2),
                "percent": vm.percent,
            }
        except ImportError:
            mem["system"] = {"note": "Install psutil for detailed memory info"}

        # GPU memory via torch.cuda
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                total = torch.cuda.get_device_properties(0).total_mem

                mem["gpu"] = {
                    "total_gb": round(total / (1024 ** 3), 2),
                    "allocated_gb": round(allocated / (1024 ** 3), 2),
                    "reserved_gb": round(reserved / (1024 ** 3), 2),
                    "free_gb": round((total - allocated) / (1024 ** 3), 2),
                    "percent": round((allocated / total) * 100, 1) if total else 0.0,
                }
            except Exception as exc:
                logger.warning("Could not read GPU memory stats: %s", exc)
                mem["gpu"] = {"error": str(exc)}

        return mem

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all counters and timers."""
        self.model_load_time = 0.0
        self.inference_times.clear()
        self.total_images_processed = 0
        self.session_start_time = time.time()
        logger.info("Performance monitor reset")
