"""
Model manager for Florence-2 vision AI model.

Provides a singleton-based manager that handles model loading, inference,
and lifecycle management for the Florence-2 model from Hugging Face.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

import config

logger = logging.getLogger(__name__)


class VisionModelManager:
    """Thread-safe singleton manager for the Florence-2 vision model.

    Usage::

        manager = VisionModelManager.get_instance()
        manager.load_model()
        result = manager.run_inference(pil_image, "<OD>")
    """

    _instance: VisionModelManager | None = None
    _lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Construction / singleton
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self.device: torch.device | None = None
        self.model: AutoModelForCausalLM | None = None
        self.processor: AutoProcessor | None = None
        self._loaded: bool = False
        self._load_time: float = 0.0

    @classmethod
    def get_instance(cls) -> VisionModelManager:
        """Return the global singleton instance (created on first call)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------

    def detect_device(self) -> torch.device:
        """Detect the best available compute device (CUDA → CPU).

        Returns:
            torch.device: The selected device.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            logger.info(
                "CUDA device detected: %s (%.1f GB VRAM)", gpu_name, gpu_mem
            )
        else:
            device = torch.device("cpu")
            logger.info("No CUDA device found – using CPU")

        self.device = device
        return device

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Download (if necessary) and load the Florence-2 model & processor.

        The model is loaded in FP16 on CUDA and FP32 on CPU. Once loaded the
        model is set to evaluation mode and cached so subsequent calls are
        no-ops.

        Raises:
            RuntimeError: If model download, loading, or weight initialisation
                fails.
            torch.cuda.OutOfMemoryError: If the GPU runs out of memory.
        """
        if self._loaded:
            logger.info("Model already loaded – skipping reload")
            return

        if self.device is None:
            self.detect_device()

        assert self.device is not None  # satisfy type checker

        logger.info("Loading model '%s' …", config.MODEL_ID)
        t0 = time.perf_counter()

        try:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32

            # Try AutoModelForImageTextToText (native in newer transformers),
            # Florence2ForConditionalGeneration, or AutoModelForCausalLM (legacy)
            loaded_model = None
            load_errors = []

            for cls_getter in (
                lambda: getattr(__import__("transformers", fromlist=["AutoModelForImageTextToText"]), "AutoModelForImageTextToText"),
                lambda: getattr(__import__("transformers", fromlist=["Florence2ForConditionalGeneration"]), "Florence2ForConditionalGeneration"),
                lambda: AutoModelForCausalLM,
            ):
                try:
                    cls = cls_getter()
                    loaded_model = cls.from_pretrained(
                        config.MODEL_ID,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                    ).to(self.device)
                    break
                except Exception as e:
                    load_errors.append(f"{cls_getter.__name__ if hasattr(cls_getter, '__name__') else 'class'}: {e}")
                    continue

            if loaded_model is None:
                raise RuntimeError(f"Could not load Florence-2 with any model class. Errors: {'; '.join(load_errors)}")

            self.model = loaded_model

            self.processor = AutoProcessor.from_pretrained(
                config.MODEL_ID,
                trust_remote_code=True,
            )

            self.model.eval()
            self._loaded = True
            self._load_time = time.perf_counter() - t0

            logger.info(
                "Model loaded in %.2f s on %s (dtype=%s)",
                self._load_time,
                self.device,
                dtype,
            )

        except torch.cuda.OutOfMemoryError:
            logger.error(
                "GPU out of memory while loading model. "
                "Try closing other GPU applications or use CPU."
            )
            self._cleanup_partial_load()
            raise

        except OSError as exc:
            logger.error("Failed to download / locate model files: %s", exc)
            self._cleanup_partial_load()
            raise RuntimeError(
                f"Model download failed for '{config.MODEL_ID}'. "
                f"Check your internet connection and model ID."
            ) from exc

        except Exception as exc:
            logger.error("Unexpected error loading model: %s", exc)
            self._cleanup_partial_load()
            raise RuntimeError(
                f"Failed to load model '{config.MODEL_ID}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def run_inference(
        self,
        image: Image.Image,
        task_prompt: str,
        text_input: str = "",
    ) -> dict[str, Any]:
        """Run a single inference pass on *image* using *task_prompt*.

        Args:
            image: A PIL RGB image.
            task_prompt: A Florence-2 task token (e.g. ``"<OD>"``).
            text_input: Optional free-text input appended to the task prompt
                (used by grounding / region-description tasks).

        Returns:
            dict: Parsed result from Florence-2's post-processing. The
            structure varies by task. An ``inference_time`` key is always
            added.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self._loaded or self.model is None or self.processor is None:
            raise RuntimeError(
                "Model is not loaded. Call load_model() before run_inference()."
            )

        assert self.device is not None

        prompt = task_prompt if not text_input else task_prompt + text_input

        t0 = time.perf_counter()

        try:
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(self.device)

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                early_stopping=False,
                do_sample=False,
            )

            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            parsed = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height),
            )

            inference_time = time.perf_counter() - t0

            result: dict[str, Any] = (
                parsed if isinstance(parsed, dict) else {task_prompt: parsed}
            )
            result["inference_time"] = inference_time

            logger.info(
                "Inference completed for task '%s' in %.3f s",
                task_prompt,
                inference_time,
            )
            return result

        except torch.cuda.OutOfMemoryError:
            logger.error("GPU OOM during inference for task '%s'", task_prompt)
            raise

        except Exception as exc:
            inference_time = time.perf_counter() - t0
            logger.error(
                "Inference failed for task '%s': %s", task_prompt, exc
            )
            return {
                "error": str(exc),
                "task": task_prompt,
                "inference_time": inference_time,
            }

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict[str, Any]:
        """Return metadata about the currently loaded (or unloaded) model."""
        info: dict[str, Any] = {
            "model_id": config.MODEL_ID,
            "device": str(self.device) if self.device else "not set",
            "dtype": (
                str(next(self.model.parameters()).dtype)
                if self.model is not None
                else "n/a"
            ),
            "loaded": self._loaded,
            "load_time": round(self._load_time, 3),
        }
        return info

    def is_loaded(self) -> bool:
        """Return ``True`` if the model is ready for inference."""
        return self._loaded

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def unload_model(self) -> None:
        """Release model and processor, freeing GPU/CPU memory."""
        logger.info("Unloading model …")
        self._cleanup_partial_load()
        self._loaded = False
        self._load_time = 0.0
        logger.info("Model unloaded successfully")

    def _cleanup_partial_load(self) -> None:
        """Internal helper to free partially loaded resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
