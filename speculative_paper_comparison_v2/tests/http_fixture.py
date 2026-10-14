"""CPU-only vLLM-shaped HTTP fixture. These are synthetic counters, not NPU results."""
import json
import os
from pathlib import Path
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

HTTPServer.allow_reuse_address = True
argv = sys.argv
port = int(argv[argv.index("--port") + 1])
model = argv[argv.index("--served-model-name") + 1]
spec = json.loads(argv[argv.index("--speculative-config") + 1])
cfg = json.loads((Path(spec["model"]) / "config.json").read_text())
kind = cfg.get("prefix_selector_kind", "dflash")
mode = os.environ["DFLASH_PREFIX_INFERENCE"]
output = Path(os.environ["RUN_DIR"])
counters = dict(finished=0, generated=0, rounds=0, accepted=0, drafted=0)
names = dict(finished="request_success", generated="generation_tokens", rounds="spec_decode_num_drafts",
             accepted="spec_decode_num_accepted_tokens", drafted="spec_decode_num_draft_tokens")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, content, mime="application/json"):
        data = content.encode()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def do_GET(self):
        if self.path == "/v1/models":
            return self.reply(json.dumps({"data": [{"id": model, "max_model_len": 32768}]}))
        text = "\n".join(f'vllm:{names[k]}_total{{model_name="{model}"}} {v}' for k, v in counters.items())
        text += f'\nvllm:num_requests_running{{model_name="{model}"}} 0\n'
        self.reply(text, "text/plain")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with (output / "fixture_requests.jsonl").open("a") as stream:
            stream.write(json.dumps(body) + "\n")
        count, k = body["max_tokens"], spec["num_speculative_tokens"]
        rounds, accepted = count // 8, count - count // 8
        hist = [0] * (k + 1)
        hist[7] = rounds
        stats = dict(num_spec_steps=rounds, num_accepted_draft_tokens=accepted,
                     num_draft_tokens=k * rounds, num_spec_tokens=k,
                     acceptance_histogram=hist, mean_acceptance_length=8.0)
        counters["finished"] += 1
        counters["generated"] += count
        counters["rounds"] += rounds
        counters["accepted"] += accepted
        counters["drafted"] += k * rounds
        if kind != "dflash" and mode != "reference" and counters["rounds"] >= 32:
            marker = ("COMPILED_PREFIX_PARITY_PASS kind=compiled_prefix_v3" if kind == "compiled_prefix_v3"
                      else "PREFIX_PARITY_PASS")
            ending = "steady-state fast path enabled" if mode == "fast" else "validation continues"
            print(f"{marker} checked_blocks=32 max_score_error=0 mode={mode} npu_graph=True "
                  f"scope=selector production_tokens_checked=True; {ending}", flush=True)
        dspark_mode = os.environ.get("VLLM_ASCEND_DSPARK_HEAD_GRAPH", "off")
        if cfg.get("speculators_model_type") == "dspark" and dspark_mode != "off" and counters["rounds"] >= 32:
            ending = "steady-state fast path enabled" if dspark_mode == "fast" else "validation continues"
            print(f"DSPARK_HEAD_PARITY_PASS checked_blocks=32 max_score_error=0 mode={dspark_mode} npu_graph=True production_tokens_checked=True; {ending}", flush=True)
        if "--cudagraph-metrics" in argv and "--enforce-eager" not in argv:
            print(f"| {k+1} | {k+1} | 0 | FULL | 1 |", flush=True)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(2):
            event = {"id": f"fixture-{counters['finished']}", "choices": [
                {"index": 0, "text": "example", "token_ids": [42] * (count // 2),
                 "finish_reason": "length" if i else None}]}
            if i:
                event.update(usage=dict(prompt_tokens=len(body["prompt"]), completion_tokens=count),
                             metrics={"speculative_decoding": stats})
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.flush()
            time.sleep(.002)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


HTTPServer(("127.0.0.1", port), Handler).serve_forever()
