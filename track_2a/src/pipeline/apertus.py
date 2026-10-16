"""Apertus through a local vLLM server: start it, talk to it, stop it."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gpu_memory_gb() -> tuple[float, float] | None:
    """(free, total) of the first GPU in GB, or None when nvidia-smi is not available."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True, timeout=20).stdout
        free, total = (float(x) / 1024 for x in out.splitlines()[0].split(","))
        return free, total
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _compute_capability() -> float | None:
    """CUDA compute capability of the first GPU (8.0 = Ampere, 7.5 = Turing), or None when unknown."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True, timeout=20).stdout
        return float(out.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def choose_dtype(requested: str = "auto") -> str:
    """The dtype the weights are served in. Apertus is trained in bfloat16, which vLLM supports from compute
    capability 8.0 on. Older GPUs (Turing, e.g. Quadro RTX 8000) get float32: in float16 the activations
    overflow and every request fails. float32 needs about 34 GB for the weights instead of 17.2 GB."""
    if requested != "auto":
        return requested
    capability = _compute_capability()
    return "float32" if capability is not None and capability < 8.0 else "bfloat16"


class ApertusServer:
    """Context manager that owns a `vllm serve` process.

        with ApertusServer() as server:
            chat(server.base_url, server.model, messages)
    """

    def __init__(self, model: str = config.APERTUS_MODEL, log_path: str | Path = "apertus_server.log",
                 dtype: str = "auto", max_num_seqs: int = 1):
        self.model = model
        self.dtype = choose_dtype(dtype)
        self.max_num_seqs = max_num_seqs
        self.log_path = Path(log_path)
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self.process = None

    def _utilization(self) -> float:
        mem = _gpu_memory_gb()
        if mem is None:
            return config.GPU_MEMORY_UTILIZATION_MAX
        free, total = mem
        return round(min(config.GPU_MEMORY_UTILIZATION_MAX, (free - config.GPU_HEADROOM_GB) / total), 2)

    def __enter__(self) -> "ApertusServer":
        vllm = Path(sys.executable).parent / "vllm"
        cmd = [str(vllm) if vllm.exists() else "vllm", "serve", self.model,
               "--port", str(self.port), "--host", "127.0.0.1",
               "--chat-template-content-format", "string",
               "--dtype", self.dtype,
               "--gpu-memory-utilization", str(self._utilization()),
               "--max-model-len", str(config.MAX_MODEL_LEN),
               "--max-num-seqs", str(self.max_num_seqs),    # requests answered together (run.py --workers)
               "--seed", str(config.SEED),
               "--no-enable-flashinfer-autotune"]
        env = dict(os.environ,
                   VLLM_USE_FLASHINFER_SAMPLER="0",         # the sampler needs nvcc at run time
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        print(f"starting Apertus: {' '.join(cmd)}\n  log: {self.log_path}", flush=True)
        self.process = subprocess.Popen(cmd, env=env, stdout=self.log_path.open("w"), stderr=subprocess.STDOUT,
                                        start_new_session=True)
        try:
            self._wait_until_ready()
            self._check_replies()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _wait_until_ready(self) -> None:
        start = time.time()
        health = f"http://127.0.0.1:{self.port}/health"
        while time.time() - start < config.SERVER_START_TIMEOUT_S:
            if self.process.poll() is not None:
                tail = "".join(self.log_path.read_text(errors="replace").splitlines(keepends=True)[-15:])
                raise RuntimeError(f"the Apertus server stopped during start-up (see {self.log_path}):\n{tail}")
            try:
                with urllib.request.urlopen(health, timeout=5) as r:
                    if r.status == 200:
                        print(f"Apertus is ready after {time.time() - start:.0f} s", flush=True)
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        raise TimeoutError(f"the Apertus server was not ready after {config.SERVER_START_TIMEOUT_S} s (see {self.log_path})")

    def _check_replies(self) -> None:
        """A server can start and still answer nothing but errors (float16 overflow: HTTP 500 on every request).
        One constrained request before the real ones turns that into a stop instead of a file of fallbacks."""
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
        try:
            reply = json.loads(chat(self.base_url, self.model, [{"role": "user", "content": "Is 2 + 2 = 4? Answer as JSON."}],
                                    schema))
            if not isinstance(reply.get("ok"), bool):
                raise ValueError(f"unexpected reply {reply!r}")
        except (OSError, ValueError, KeyError) as e:
            raise RuntimeError(f"the Apertus server started in {self.dtype} but cannot answer a test request ({e}). "
                               f"See {self.log_path}; on a GPU without bfloat16 use --dtype float32.") from e
        print(f"Apertus answers ({self.dtype})", flush=True)

    def __exit__(self, *exc) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()


def chat(base_url: str, model: str, messages: list, schema: dict | None = None) -> str:
    """One chat completion, temperature 0. With a schema the reply is constrained to that JSON schema."""
    body = {"model": model, "messages": messages, "temperature": config.TEMPERATURE, "seed": config.SEED,
            "max_tokens": config.MAX_TOKENS}
    if schema is not None:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    try:
        with urllib.request.urlopen(request, timeout=config.REQUEST_TIMEOUT_S) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        raise OSError(f"HTTP {e.code} from the Apertus server: {e.read().decode(errors='replace')[:500]}") from e
