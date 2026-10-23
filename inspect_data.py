import gzip, json, itertools, collections
p = "data/raw/dataset.jsonl.gz"
keys = collections.Counter(); types = collections.Counter(); n = 0
samples = []
turn_counts = collections.Counter()
with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        n += 1
        if not line.strip(): continue
        try: obj = json.loads(line)
        except Exception as e:
            if n < 5: print("BAD", n, repr(line[:100]))
            continue
        keys.update(obj.keys())
        for k,v in obj.items(): types[(k, type(v).__name__)] += 1
        sample = obj.get("sample")
        if isinstance(sample, list):
            turn_counts[len(sample)] += 1
            if len(samples) < 8 and len(sample) >= 2:
                samples.append(sample)
print("lines", n)
print("keys", keys)
print("types", types)
print("turn counts", turn_counts.most_common(12))
for s in samples: print(json.dumps(s, ensure_ascii=False))
