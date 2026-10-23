import gzip, json, collections, statistics
p = "data/raw/dataset.jsonl.gz"
rows=[]; q=collections.Counter(); lens=collections.defaultdict(list); n=0
with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        n+=1
        obj=json.loads(line); s=obj["sample"]
        for i,t in enumerate(s):
            q[(i%2, t.strip().endswith("?"))]+=1
            lens[i%2].append(len(t))
        if len(rows)<6 and len(s)>=4: rows.append(s)
print("records", n, "turns", sum(len(s) for s in rows) if False else "n/a")
print("question rate by parity:")
for par in (0,1):
    tot=q[(par,True)]+q[(par,False)]
    print(par, "turns", tot, "questions", q[(par,True)], round(q[(par,True)]/tot,3))
print("mean chars parity", {p: round(statistics.mean(v),1) for p,v in lens.items()})
for s in rows:
    print("---")
    for i,t in enumerate(s): print(i, repr(t.strip()[:500]))
