"""Needle-in-haystack through the OpenAI API. usage: needle_api.py <n_sentences> <depth 0..1> [url]"""
import sys, random, time, requests
n = int(sys.argv[1]); depth = float(sys.argv[2]); url = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8096"
random.seed(n)
words = "the of and to in is was for on that with as by at from his her it an are which be this had were not but have".split()
filler = [" ".join(random.choice(words) for _ in range(12)) + "." for _ in range(n)]
filler.insert(int(n * depth), "The secret launch code for Project Kestrel is 7-4-1-9-ORCHID.")
q = " ".join(filler) + "\n\nQuestion: What is the secret launch code for Project Kestrel? Reply with the code only."
t0 = time.time()
r = requests.post(url + "/v1/chat/completions", json = {"model": "x", "messages": [{"role": "user", "content": q}], "max_tokens": 40,
                  "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, timeout = 3600)
j = r.json()
if "choices" not in j: print("ERROR", j); sys.exit(1)
print("answer:", j["choices"][0]["message"]["content"].strip(), "| usage:", j.get("usage"), "| wall", round(time.time() - t0, 1), "s")
