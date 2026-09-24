"""OpenAI-API smoke test + streaming decode speed against the TabbyAPI server."""
import base64, json, sys, time, requests
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8096"
IMG = sys.argv[2] if len(sys.argv) > 2 else None

def chat(messages, max_tokens = 512, think = False, label = ""):
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "stream": True,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": think}, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); first = None; content = []; reasoning = []; usage = None; n_chunks = 0
    with requests.post(URL + "/v1/chat/completions", json = body, stream = True, timeout = 600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: "): continue
            data = line[6:]
            if data == b"[DONE]": break
            j = json.loads(data)
            if j.get("usage"): usage = j["usage"]
            for c in j.get("choices", []):
                d = c.get("delta", {})
                if d.get("content") or d.get("reasoning_content"):
                    if first is None: first = time.perf_counter()
                    n_chunks += 1
                if d.get("content"): content.append(d["content"])
                if d.get("reasoning_content"): reasoning.append(d["reasoning_content"])
    t1 = time.perf_counter()
    ct = usage["completion_tokens"] if usage else n_chunks
    print(f"[{label}] TTFT {(first - t0) * 1000:.0f} ms, {ct} tokens, decode {(ct - 1) / (t1 - first):.1f} tok/s, usage={usage}")
    if reasoning: print("  reasoning:", "".join(reasoning)[:200].replace("\n", " "), "...")
    print("  content:", "".join(content)[:400].replace("\n", " "))
    return "".join(content)

print(requests.get(URL + "/v1/models").json()["data"][0]["id"])
chat([{"role": "user", "content": "Write a Python function that parses ISO-8601 durations like 'P3DT4H5M' into seconds, with tests."}], 600, label = "code")
chat([{"role": "user", "content": "Tell me a short story about a robot learning to paint."}], 400, label = "prose")
chat([{"role": "user", "content": "What is 17 * 23? Answer briefly."}], 800, think = True, label = "thinking")
if IMG:
    b64 = base64.b64encode(open(IMG, "rb").read()).decode()
    chat([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "What is in this image? Be concise."}]}], 200, label = "vision")
