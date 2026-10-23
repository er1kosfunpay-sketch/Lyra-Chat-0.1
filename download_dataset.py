import os, time, requests
url = "https://huggingface.co/datasets/Den4ikAI/russian_dialogues_2/resolve/main/dataset.jsonl.gz"
out = r"data/raw/dataset.jsonl.gz"
tmp = out + ".part"
total = 271726499
t0 = time.time()
for attempt in range(1, 15):
    done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if done >= total:
        break
    headers = {"Range": f"bytes={done}-", "Accept-Encoding": "identity"}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as r:
            if r.status_code == 416:
                break
            r.raise_for_status()
            with open(tmp, "ab") as f:
                for chunk in r.iter_content(1024 * 512):
                    if chunk:
                        f.write(chunk); done += len(chunk)
                        if done // (40 * 1024 * 1024) != (done - len(chunk)) // (40 * 1024 * 1024):
                            print(f"{done/1e6:.0f}/{total/1e6:.0f} MB {done/1e6/(time.time()-t0):.1f} MB/s", flush=True)
    except Exception as e:
        print("attempt", attempt, "failed:", type(e).__name__, str(e)[:200], "at", done, flush=True)
        time.sleep(3)
        continue
if os.path.getsize(tmp) == total:
    os.replace(tmp, out)
    print("DONE", total, f"{time.time()-t0:.1f}s")
else:
    print("INCOMPLETE", os.path.getsize(tmp))
