"""inkserve: Inkling 2 Mini at long context on a 16 GB Mac.

A single-stream HTTP server over oMLX's Inkling 2 model code (MLX), with the
cache layout a 16 GB Mac needs:

- the 40 sliding-window layer visits keep only their last 1,024 keys
  (oMLX keeps every key on all 48 visits, ~192 KB/token);
- the 8 global visits keep full history as 8-bit affine K/V (group 64),
  allocated once at --ctx, ~17 KB/token;
- decode attention on the global visits runs a split-K Metal kernel straight
  from the 8-bit codes (q8attn.py) once the context passes FLASH_MIN tokens.

API: llama-server style /health and /completion (with "timings"), and OpenAI
style /v1/models and /v1/chat/completions (streaming, reasoning_content,
tool calls via oMLX's Inkling output parser). One request runs at a time.
"""
import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

__version__ = "0.2.0"
log = logging.getLogger("inkserve")

# Below this many cached keys the plain MLX-ops path decodes faster end to end:
# the custom kernel's own time is lower at every depth, but on a base M4 it cost
# ~1.4 ms per token at short context (measured A/B), more than it saves there.
FLASH_MIN = 12288

class WindowKVCache(KVCache):
    """Sliding-window visit: keeps the last window-1 keys plus the new ones,
    contiguous and in order (what oMLX's banded kernels expect)."""

    def __init__(self, window, slack=256):
        super().__init__()
        self.window = window
        self.slack = slack
        self.n = 0

    def update_and_fetch(self, keys, values):
        L = keys.shape[2]
        keep = self.window - 1
        if self.keys is None or self.n + L > self.keys.shape[2]:
            B, H, _, D = keys.shape
            n_old = min(self.n, keep)
            cap = n_old + L + self.slack
            nk = mx.zeros((B, H, cap, D), keys.dtype)
            nv = mx.zeros((B, H, cap, values.shape[3]), values.dtype)
            if self.keys is not None and n_old:
                nk[..., :n_old, :] = self.keys[..., self.n - n_old : self.n, :]
                nv[..., :n_old, :] = self.values[..., self.n - n_old : self.n, :]
            self.keys, self.values, self.n = nk, nv, n_old
        self.keys[..., self.n : self.n + L, :] = keys
        self.values[..., self.n : self.n + L, :] = values
        self.n += L
        self.offset += L
        return self.keys[..., : self.n, :], self.values[..., : self.n, :]

    def snapshot(self):
        return (self.keys, self.values, self.n, self.offset)

    def restore(self, snap):
        self.keys, self.values, self.n, self.offset = snap


class Q8GlobalKVCache(KVCache):
    """Global visit: full history as 8-bit affine codes, allocated once.
    Prefill fetches a bf16 copy for the flash kernel; decode reads the codes."""

    bits = 8
    group = 64

    def __init__(self, max_len):
        super().__init__()
        self.max_len = max_len
        self.kq = None

    def _alloc(self, keys):
        B, H, _, D = keys.shape
        w = D * self.bits // 32
        g = D // self.group
        z = lambda last, dt: mx.zeros((B, H, self.max_len, last), dt)  # noqa: E731
        self.kq, self.vq = z(w, mx.uint32), z(w, mx.uint32)
        self.ks, self.kb = z(g, keys.dtype), z(g, keys.dtype)
        self.vs, self.vb = z(g, keys.dtype), z(g, keys.dtype)
        self.dtype = keys.dtype

    def append(self, keys, values):
        if self.kq is None:
            self._alloc(keys)
        a, b = self.offset, self.offset + keys.shape[2]
        if b > self.max_len:
            raise ContextFull(f"prompt + reply need {b} tokens; context is {self.max_len}")
        kq, ks, kb = mx.quantize(keys, group_size=self.group, bits=self.bits)
        vq, vs, vb = mx.quantize(values, group_size=self.group, bits=self.bits)
        self.kq[..., a:b, :] = kq
        self.ks[..., a:b, :] = ks
        self.kb[..., a:b, :] = kb
        self.vq[..., a:b, :] = vq
        self.vs[..., a:b, :] = vs
        self.vb[..., a:b, :] = vb
        self.offset = b

    def codes(self):
        n = self.offset
        return ((self.kq[..., :n, :], self.ks[..., :n, :], self.kb[..., :n, :]),
                (self.vq[..., :n, :], self.vs[..., :n, :], self.vb[..., :n, :]))

    def update_and_fetch(self, keys, values):
        self.append(keys, values)
        (kq, ks, kb), (vq, vs, vb) = self.codes()
        k = mx.dequantize(kq, ks, kb, group_size=self.group, bits=self.bits)
        v = mx.dequantize(vq, vs, vb, group_size=self.group, bits=self.bits)
        return k.astype(self.dtype), v.astype(self.dtype)

    def snapshot(self):
        # Codes below offset are never rewritten, so the offset is enough.
        return self.offset

    def restore(self, snap):
        self.offset = snap


class ContextFull(Exception):
    pass


def q8_decode_attention(q, kc, vc, rel, sink, scale, group, bits):
    """One query over 8-bit K/V via quantized_matmul (short contexts)."""
    B, hq, L, D = q.shape
    hkv = kc[0].shape[1]
    s = kc[0].shape[2]
    rep = hq // hkv
    n = min(rel.shape[-1], s)
    tail = rel[0, 0, :, :n][:, ::-1]
    if n < s:
        tail = mx.concatenate([mx.zeros((hq, s - n), dtype=tail.dtype), tail], -1)
    qg = (q * scale).reshape(B, hkv, rep, L, D)
    kc = tuple(mx.expand_dims(x, -3) for x in kc)
    vc = tuple(mx.expand_dims(x, -3) for x in vc)
    scores = mx.quantized_matmul(qg, *kc, transpose=True, group_size=group, bits=bits)
    scores = scores.astype(mx.float32) + tail.reshape(B, hkv, rep, 1, s).astype(mx.float32)
    sk = sink[0, 0].astype(mx.float32).reshape(B, hkv, rep, 1, 1)
    p = mx.softmax(mx.concatenate([scores, sk], axis=-1), axis=-1, precise=True)
    out = mx.quantized_matmul(p[..., :s].astype(q.dtype), *vc, transpose=False,
                              group_size=group, bits=bits)
    return out.reshape(B, hq, L, D)


def check_omlx(expect):
    """inkserve leans on oMLX internals: fail early and clearly if they moved."""
    import importlib.metadata as md
    import inspect

    have = md.version("omlx")
    from omlx.patches.inkling2 import model as ink

    sig = list(inspect.signature(ink.Inkling2Attention.attend).parameters)
    if sig != ["self", "q", "k", "v", "rel", "sink", "kv"] or not hasattr(ink, "kernels"):
        raise SystemExit(f"oMLX {have}: Inkling2Attention.attend changed ({sig}); "
                         f"this inkserve was written for oMLX {expect or '0.8.2'}")
    if expect and have != expect:
        log.warning("oMLX %s installed; inkserve was tested with %s", have, expect)
    return have


def install_attend_patch():
    from omlx.patches.inkling2 import model as ink
    from q8attn import q8_flash_decode

    orig = ink.Inkling2Attention.attend

    def attend(self, q, k, v, rel, sink, kv):
        if isinstance(kv, Q8GlobalKVCache) and q.shape[0] == 1 and q.shape[2] == 1:
            kv.append(k, v)
            if kv.offset > FLASH_MIN:
                return q8_flash_decode(q, kv, rel, sink, self.scale)
            kc, vc = kv.codes()
            return q8_decode_attention(q, kc, vc, rel, sink, self.scale, kv.group, kv.bits)
        return orig(self, q, k, v, rel, sink, kv)

    ink.Inkling2Attention.attend = attend


class Engine:
    def __init__(self, model_path, ctx, prefill_chunk, expect_omlx=""):
        log.info("inkserve %s on oMLX %s", __version__, check_omlx(expect_omlx))
        from omlx.utils.model_loading import load_text_model

        info = mx.device_info()
        mx.set_wired_limit(int(info["max_recommended_working_set_size"]))
        mx.set_cache_limit(512 * 2**20)
        install_attend_patch()
        self.model_path = model_path
        self.model, self.tok = load_text_model(model_path)
        mx.eval(self.model.parameters())
        self.ctx = ctx
        self.chunk = prefill_chunk
        self.lock = threading.Lock()
        args = self.model.args.text if hasattr(self.model.args, "text") else self.model.args
        self.args = args
        self.schedule = self.model.model._schedule
        self.cache = None
        self.tokens = []   # tokens whose K/V are in self.cache, in order
        self.snap = None   # (n_tokens, per-visit state) at the end of the last prompt
        cfg = json.load(open(os.path.join(model_path, "config.json")))
        self.model_config = cfg
        self.stop_ids = set()
        for t in ("<|content_model_end_sampling|>",):
            tid = self.tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid >= 0:
                self.stop_ids.add(tid)
        eos = getattr(self.tok, "eos_token_ids", None) or []
        self.stop_ids.update(int(x) for x in eos)
        log.info("loaded %s: %.2f GiB weights, ctx %d, stop ids %s", model_path,
                 mx.get_active_memory() / 2**30, ctx, sorted(self.stop_ids))

    # -- cache --------------------------------------------------------------
    def _new_cache(self):
        self.cache = None
        self.tokens = []
        self.snap = None
        mx.clear_cache()
        out = []
        for idx, _p in self.schedule:
            if self.args.is_sliding(idx):
                kv = WindowKVCache(self.args.sliding_window_size)
            else:
                kv = Q8GlobalKVCache(self.ctx)
            out.append(CacheList(kv, ArraysCache(4)))
        self.cache = out

    def _take_snapshot(self):
        st = []
        for c in self.cache:
            kv, conv = c[0], c[1]
            st.append((kv.snapshot(), list(conv.cache)))
        self.snap = (len(self.tokens), st)

    def _restore_snapshot(self):
        n, st = self.snap
        for c, (kvs, conv) in zip(self.cache, st):
            c[0].restore(kvs)
            c[1].cache = list(conv)
        self.tokens = self.tokens[:n]

    def _reuse(self, ids, allow):
        """Return how many leading prompt tokens are already in the cache."""
        if not allow or self.cache is None:
            self._new_cache()
            return 0
        n = len(self.tokens)
        if n and len(ids) > n and ids[:n] == self.tokens:
            return n
        if self.snap is not None:
            m = self.snap[0]
            if m and len(ids) > m and ids[:m] == self.tokens[:m]:
                self._restore_snapshot()
                return m
        self._new_cache()
        return 0

    # -- generation -----------------------------------------------------------
    def generate(self, ids, max_tokens, sampler, ignore_eos=False, cache_prompt=True,
                 on_token=None, should_stop=None):
        """Yields token ids. Fills stats dict on self.last_stats."""
        if len(ids) + max_tokens > self.ctx:
            max_tokens = self.ctx - len(ids)
            if max_tokens < 1:
                raise ContextFull(f"prompt is {len(ids)} tokens; context is {self.ctx}")
        reused = self._reuse(ids, cache_prompt)
        todo = ids[reused:]
        t0 = time.time()
        logits = None
        for s in range(0, len(todo), self.chunk):
            x = mx.array(todo[s : s + self.chunk])[None]
            h = self.model.model(x, self.cache)
            logits = self.model.logits_from_hidden(h[:, -1:, :])[:, -1, :]
            mx.eval(logits)
            del h
            mx.clear_cache()
            self.tokens.extend(todo[s : s + self.chunk])
            if should_stop and should_stop():
                break
        t_prompt = time.time() - t0
        self._take_snapshot()
        stats = {"prompt_n": len(todo), "cache_n": reused,
                 "prompt_ms": t_prompt * 1000,
                 "prompt_per_second": len(todo) / t_prompt if t_prompt > 0 else 0.0}
        out = []
        t1 = time.time()
        y = sampler(logits)
        mx.eval(y)
        finish = "length"
        for i in range(max_tokens):
            tok = y.item()
            if i + 1 < max_tokens and not (tok in self.stop_ids and not ignore_eos):
                # Queue the next step before handing this token back.
                nxt = sampler(self.model(y.reshape(1, 1), cache=self.cache)[:, -1, :])
                mx.async_eval(nxt)
                self.tokens.append(tok)
            else:
                nxt = None
            out.append(tok)
            if on_token is not None and on_token(tok) is False:
                finish = "stop"
                if nxt is not None:
                    mx.eval(nxt)
                break
            if tok in self.stop_ids and not ignore_eos:
                finish = "stop"
                break
            if should_stop and should_stop():
                finish = "stop"
                if nxt is not None:
                    mx.eval(nxt)
                break
            if nxt is None:
                break
            y = nxt
        t_gen = time.time() - t1
        n = len(out)
        stats.update({"predicted_n": n, "predicted_ms": t_gen * 1000,
                      "predicted_per_second": (n - 1) / t_gen if n > 1 and t_gen > 0 else 0.0,
                      "finish": finish})
        self.last_stats = stats
        mx.clear_cache()
        return out


def sampler_from(body):
    temp = float(body.get("temperature", 0.0) if body.get("temperature") is not None else 0.0)
    return make_sampler(temp=temp, top_p=float(body.get("top_p", 1.0) or 1.0),
                        min_p=float(body.get("min_p", 0.0) or 0.0),
                        top_k=int(body.get("top_k", 0) or 0))


class ThinkRouter:
    """Splits parser output with <think>...</think> into reasoning and content."""

    def __init__(self):
        self.in_think = False
        self.buf = ""

    def feed(self, text, final=False):
        self.buf += text
        reasoning, content = "", ""
        while self.buf:
            marker = "</think>" if self.in_think else "<think>"
            i = self.buf.find(marker)
            if i >= 0:
                piece, self.buf = self.buf[:i], self.buf[i + len(marker):]
                if self.in_think:
                    reasoning += piece
                else:
                    content += piece
                self.in_think = not self.in_think
                continue
            keep = 0
            if not final:
                for k in range(min(len(marker) - 1, len(self.buf)), 0, -1):
                    if marker.startswith(self.buf[-k:]):
                        keep = k
                        break
            piece = self.buf[: len(self.buf) - keep]
            self.buf = self.buf[len(self.buf) - keep:]
            if self.in_think:
                reasoning += piece
            else:
                content += piece
            break
        return reasoning, content


ENGINE = None
WORKER = None
MODEL_ID = "inkling-2-mini"


class EngineThread(threading.Thread):
    """Owns every MLX call: MLX streams belong to the thread that made them,
    so the model loads here and requests run here, one at a time."""

    def __init__(self, *engine_args):
        super().__init__(daemon=True, name="engine")
        self.engine_args = engine_args
        self.jobs = queue.Queue()
        self.ready = threading.Event()
        self.error = None

    def run(self):
        global ENGINE
        try:
            ENGINE = Engine(*self.engine_args)
        except BaseException as e:  # noqa: BLE001
            self.error = e
            self.ready.set()
            return
        self.ready.set()
        while True:
            fn, box, done = self.jobs.get()
            try:
                box["result"] = fn()
            except BaseException as e:  # noqa: BLE001
                log.exception("request failed")
                box["error"] = e
            done.set()

    def call(self, fn):
        box, done = {}, threading.Event()
        self.jobs.put((fn, box, done))
        done.wait()
        if "error" in box:
            raise box["error"]
        return box["result"]


def timings(stats):
    return {k: stats[k] for k in ("prompt_n", "prompt_ms", "prompt_per_second",
                                  "predicted_n", "predicted_ms", "predicted_per_second",
                                  "cache_n")}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "inkserve/" + __version__

    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            return self._json(200, {"status": "ok"})
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            return self._json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "owned_by": "local",
                 "meta": {"n_ctx": ENGINE.ctx}}]})
        return self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        try:
            body = self._body()
        except Exception as e:  # noqa: BLE001
            return self._json(400, {"error": {"message": f"bad JSON: {e}"}})
        path = self.path.rstrip("/")
        try:
            if path in ("/completion", "/completions"):
                return self.completion(body)
            if path in ("/v1/chat/completions", "/chat/completions"):
                return self.chat(body)
        except ContextFull as e:
            return self._json(400, {"error": {"message": str(e), "type": "exceed_context_size_error"}})
        return self._json(404, {"error": {"message": "not found"}})

    # llama-server style raw completion (used by the bench)
    def completion(self, body):
        prompt = body.get("prompt", "")
        ids = ENGINE.tok.encode(prompt) if isinstance(prompt, str) else list(prompt)
        n = int(body.get("n_predict", body.get("max_tokens", 128)))
        out, st = WORKER.call(lambda: (
            ENGINE.generate(ids, max(1, n), sampler_from(body),
                            ignore_eos=bool(body.get("ignore_eos", False)),
                            cache_prompt=bool(body.get("cache_prompt", True))),
            ENGINE.last_stats))
        text = ENGINE.tok.decode(out)
        return self._json(200, {"content": text, "tokens_predicted": len(out),
                                "tokens_evaluated": len(ids), "model": MODEL_ID,
                                "stop": True, "timings": timings(st)})

    def _prompt_ids(self, body):
        kwargs = dict(body.get("chat_template_kwargs") or {})
        if body.get("reasoning_effort") and "reasoning_effort" not in kwargs:
            kwargs["reasoning_effort"] = body["reasoning_effort"]
        msgs = body.get("messages") or []
        tools = body.get("tools") or None
        text = ENGINE.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                              tokenize=False, **kwargs)
        return ENGINE.tok.encode(text, add_special_tokens=False)

    def chat(self, body):
        from omlx.adapter.output_parser import InklingOutputParserSession

        ids = self._prompt_ids(body)
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 4096)
        stream = bool(body.get("stream", False))
        parser = InklingOutputParserSession(ENGINE.tok, ENGINE.model_path)
        router = ThinkRouter()
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        parts = {"reasoning": [], "content": []}
        state = {"broken": False}

        def send_chunk(delta, finish=None):
            obj = {"id": cid, "object": "chat.completion.chunk", "created": created,
                   "model": MODEL_ID,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            try:
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                state["broken"] = True

        def emit(reasoning, content):
            if reasoning:
                parts["reasoning"].append(reasoning)
            if content:
                parts["content"].append(content)
            if stream and (reasoning or content):
                d = {}
                if reasoning:
                    d["reasoning_content"] = reasoning
                if content:
                    d["content"] = content
                send_chunk(d)

        def on_token(tok):
            r = parser.process_token(tok)
            if r.visible_text:
                emit(*router.feed(r.visible_text))
            if r.is_stop or state["broken"]:
                return False
            return True

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            send_chunk({"role": "assistant", "content": ""})
        try:
            # on_token runs on the engine thread; it only decodes and writes.
            st = WORKER.call(lambda: (
                ENGINE.generate(ids, max(1, max_tokens), sampler_from(body),
                                cache_prompt=bool(body.get("cache_prompt", True)),
                                on_token=on_token, should_stop=lambda: state["broken"]),
                ENGINE.last_stats)[1])
        except ContextFull as e:
            if not stream:
                raise
            send_chunk({"content": f"[error: {e}]"}, "length")
            return
        fin = parser.finalize()
        if fin.visible_text:
            emit(*router.feed(fin.visible_text))
        emit(*router.feed("", final=True))
        tool_calls = [{"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                       "function": {"name": t["name"], "arguments": t["arguments"]}}
                      for t in fin.tool_calls]
        finish = "tool_calls" if tool_calls else ("length" if st["finish"] == "length" else "stop")
        usage = {"prompt_tokens": len(ids), "completion_tokens": st["predicted_n"],
                 "total_tokens": len(ids) + st["predicted_n"],
                 "prompt_tokens_details": {"cached_tokens": st["cache_n"]}}
        if stream:
            if tool_calls:
                send_chunk({"tool_calls": [dict(tc, index=i) for i, tc in enumerate(tool_calls)]})
            obj = {"id": cid, "object": "chat.completion.chunk", "created": created,
                   "model": MODEL_ID, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                   "usage": usage, "timings": timings(st)}
            try:
                self.wfile.write(f"data: {json.dumps(obj)}\n\ndata: [DONE]\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        msg = {"role": "assistant", "content": "".join(parts["content"]).strip()}
        reasoning = "".join(parts["reasoning"]).strip()
        if reasoning:
            msg["reasoning_content"] = reasoning
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return self._json(200, {"id": cid, "object": "chat.completion", "created": created,
                                "model": MODEL_ID,
                                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                                "usage": usage, "timings": timings(st)})


def main():
    global ENGINE, MODEL_ID
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="MLX model directory")
    ap.add_argument("--alias", default=MODEL_ID)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ctx", type=int, default=98304)
    ap.add_argument("--prefill-chunk", type=int, default=256)
    ap.add_argument("--expect-omlx", default="", help="oMLX version this was tested with")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    global WORKER
    MODEL_ID = a.alias
    WORKER = EngineThread(a.model, a.ctx, a.prefill_chunk, a.expect_omlx)
    WORKER.start()
    WORKER.ready.wait()
    if WORKER.error is not None:
        raise SystemExit(f"model failed to load: {WORKER.error}")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    log.info("listening on http://%s:%d (model id %s)", a.host, a.port, MODEL_ID)
    srv.serve_forever()


if __name__ == "__main__":
    main()
