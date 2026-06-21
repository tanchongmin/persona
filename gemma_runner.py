import json
import os
import sys
import threading
import time
from collections import deque


ROOT = os.getcwd()
DEFAULT_MODEL_PATH = os.path.join(
    ROOT,
    "models",
    "gemma-4-E2B-it-q4",
    "gemma-4-E2B_q4_0-it.gguf",
)
DEFAULT_MM_PROJECTOR_PATH = os.path.join(
    ROOT,
    "models",
    "gemma-4-E2B-it-q4",
    "mmproj-google_gemma-4-E2B-it-f16.gguf",
)


def load_env():
    env_path = os.path.join(ROOT, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env()

MODEL_PATH = os.environ.get("GEMMA_MODEL_PATH", DEFAULT_MODEL_PATH)
MM_PROJECTOR_PATH = (
    os.environ.get("GEMMA_MM_PROJECTOR_PATH")
    or os.environ.get("GEMMA_CLIP_MODEL_PATH")
    or (DEFAULT_MM_PROJECTOR_PATH if os.path.exists(DEFAULT_MM_PROJECTOR_PATH) else "")
)
if os.environ.get("GEMMA_DISABLE_VISION", "false").lower() == "true":
    MM_PROJECTOR_PATH = ""
VISION_REQUIRED_FOR_IMAGES = os.environ.get("GEMMA_REQUIRE_VISION_FOR_IMAGES", "true").lower() != "false"
N_CTX = int(os.environ.get("GEMMA_N_CTX", "131072"))
N_THREADS = int(os.environ.get("GEMMA_THREADS", "2"))
N_BATCH = int(os.environ.get("GEMMA_N_BATCH", "64"))
N_GPU_LAYERS = int(os.environ.get("GEMMA_N_GPU_LAYERS", "0"))
MAX_CONCURRENT_STREAMS = int(os.environ.get("MAX_CONCURRENT_STREAMS", "100"))
PER_IP_MIN_TURN_INTERVAL_MS = int(os.environ.get("PER_IP_MIN_TURN_INTERVAL_MS", "3000"))
PER_IP_MAX_QUEUED_TURNS = int(os.environ.get("PER_IP_MAX_QUEUED_TURNS", "5"))

emit_lock = threading.Lock()
model_lock = threading.Lock()
vision_enabled = False


def emit(payload):
    with emit_lock:
        print(json.dumps(payload, ensure_ascii=False), flush=True)


def emit_error(request, message, code="error"):
    emit({
        "id": request.get("id") if isinstance(request, dict) else None,
        "type": "error",
        "code": code,
        "error": message,
    })


def load_model():
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Gemma4ChatHandler

    chat_handler = None
    if MM_PROJECTOR_PATH:
        chat_handler = Gemma4ChatHandler(
            clip_model_path=MM_PROJECTOR_PATH,
            verbose=False,
            use_gpu=os.environ.get("GEMMA_MM_PROJECTOR_USE_GPU", "false").lower() == "true",
        )

    return Llama(
        model_path=MODEL_PATH,
        chat_handler=chat_handler,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_batch=N_BATCH,
        n_gpu_layers=N_GPU_LAYERS,
        offload_kqv=os.environ.get("GEMMA_OFFLOAD_KQV") == "true",
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )


try:
    model = load_model()
    vision_enabled = bool(MM_PROJECTOR_PATH)
    emit({
        "type": "ready",
        "model": MODEL_PATH,
        "mm_projector": MM_PROJECTOR_PATH or None,
        "vision_enabled": vision_enabled,
        "backend": "llama_cpp",
        "n_ctx": N_CTX,
        "n_threads": N_THREADS,
        "max_concurrent_streams": MAX_CONCURRENT_STREAMS,
        "per_ip_min_turn_interval_ms": PER_IP_MIN_TURN_INTERVAL_MS,
        "per_ip_max_queued_turns": PER_IP_MAX_QUEUED_TURNS,
    })
except Exception as exc:
    emit({"type": "load_error", "error": str(exc), "model": MODEL_PATH})
    raise


def parse_response(text):
    text = text.strip()
    marker = "<|channel>final\n"
    if marker in text:
        text = text.split(marker, 1)[1]
    for token in ("<|return|>", "<|end|>", "<|eot_id|>"):
        text = text.replace(token, "")
    return text.strip()


def image_urls(request):
    urls = request.get("image_urls")
    if not isinstance(urls, list):
        return []
    return [
        url for url in urls
        if isinstance(url, str) and url.startswith("data:image/")
    ]


def build_messages(prompt, request=None):
    images = image_urls(request or {})
    user_content = prompt
    if images:
        user_content = [
            {"type": "text", "text": prompt},
            *[
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
                for image_url in images
            ],
        ]

    return [
        {
            "role": "system",
            "content": "Follow the user's formatting instructions exactly. Return only the requested content.",
        },
        {"role": "user", "content": user_content},
    ]


def generation_options(request):
    return {
        "max_tokens": int(request.get("max_new_tokens") or 500),
        "temperature": float(request.get("temperature") or 1.0),
        "top_p": float(request.get("top_p") or 0.95),
        "top_k": int(request.get("top_k") or 64),
        "repeat_penalty": float(os.environ.get("GEMMA_REPEAT_PENALTY", "1.05")),
    }


def generate_text(request):
    validate_vision_request(request)
    response = model.create_chat_completion(
        messages=build_messages(request["input"], request),
        stream=False,
        **generation_options(request),
    )
    return parse_response(response["choices"][0]["message"]["content"])


def stream_text(request):
    validate_vision_request(request)
    chunks = []
    response = model.create_chat_completion(
        messages=build_messages(request["input"], request),
        stream=True,
        **generation_options(request),
    )
    for event in response:
        delta = event["choices"][0].get("delta", {}).get("content", "")
        if not delta:
            continue
        chunks.append(delta)
        emit({"id": request["id"], "type": "delta", "text": delta})
    return parse_response("".join(chunks))


def validate_vision_request(request):
    if vision_enabled or not image_urls(request):
        return
    if VISION_REQUIRED_FOR_IMAGES:
        raise ValueError(
            "Gemma image input requires a multimodal projector. "
            "Run npm run download:model:vps or set GEMMA_MM_PROJECTOR_PATH to mmproj-google_gemma-4-E2B-it-f16.gguf."
        )


def handle_request(request):
    with model_lock:
        text = stream_text(request) if request.get("stream") else generate_text(request)
    emit({"id": request["id"], "type": "done", "text": text})


class RequestScheduler:
    def __init__(self):
        self.condition = threading.Condition()
        self.ip_states = {}
        self.active_count = 0

    def submit(self, request):
        ip = self.client_ip(request)
        with self.condition:
            state = self.ip_states.setdefault(ip, {
                "active": False,
                "last_started_at": 0.0,
                "queue": deque(),
            })
            if len(state["queue"]) >= PER_IP_MAX_QUEUED_TURNS:
                return False
            state["queue"].append(request)
            self.condition.notify()
            return True

    def client_ip(self, request):
        value = str(request.get("client_ip") or "unknown").strip()
        return value or "unknown"

    def run(self):
        while True:
            with self.condition:
                wait_seconds = self.next_wait_seconds()
                request_item = self.next_request()
                if request_item is None:
                    self.condition.wait(wait_seconds)
                    continue

            ip, request = request_item
            thread = threading.Thread(target=self.execute, args=(ip, request), daemon=True)
            thread.start()

    def next_wait_seconds(self):
        if self.active_count >= MAX_CONCURRENT_STREAMS:
            return None

        now = time.monotonic()
        wait_seconds = None
        for state in self.ip_states.values():
            if state["active"] or not state["queue"]:
                continue
            remaining = state["last_started_at"] + (PER_IP_MIN_TURN_INTERVAL_MS / 1000) - now
            if remaining <= 0:
                return 0
            wait_seconds = remaining if wait_seconds is None else min(wait_seconds, remaining)
        return wait_seconds

    def next_request(self):
        if self.active_count >= MAX_CONCURRENT_STREAMS:
            return None

        now = time.monotonic()
        for ip, state in list(self.ip_states.items()):
            if state["active"] or not state["queue"]:
                self.cleanup_ip(ip, state)
                continue
            earliest_start = state["last_started_at"] + (PER_IP_MIN_TURN_INTERVAL_MS / 1000)
            if earliest_start > now:
                continue

            request = state["queue"].popleft()
            state["active"] = True
            state["last_started_at"] = now
            self.active_count += 1
            return ip, request
        return None

    def execute(self, ip, request):
        try:
            handle_request(request)
        except Exception as exc:
            emit_error(request, str(exc))
        finally:
            with self.condition:
                state = self.ip_states.get(ip)
                if state:
                    state["active"] = False
                    self.cleanup_ip(ip, state)
                self.active_count = max(0, self.active_count - 1)
                self.condition.notify_all()

    def cleanup_ip(self, ip, state):
        if not state["active"] and not state["queue"]:
            self.ip_states.pop(ip, None)


scheduler = RequestScheduler()
threading.Thread(target=scheduler.run, daemon=True).start()

for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception as exc:
        emit({"id": None, "type": "error", "error": str(exc)})
        continue

    if not scheduler.submit(request):
        emit_error(
            request,
            "Too many queued turns from this IP address. Try again after your pending turns finish.",
            "rate_limited",
        )
