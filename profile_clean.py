import cProfile, pstats, io, time, json
from pathlib import Path
from conversation_llm.config import DataConfig
from conversation_llm.data.preprocessing import clean_dialogue
from conversation_llm.data.downloader import iter_raw_dialogues
from conversation_llm.data.quality import DialogueQualityStats, DialogueDeduplicator
cfg=DataConfig(source_url='x')
stats=DialogueQualityStats(); dedup=DialogueDeduplicator()
raws=[]
for i,d in enumerate(iter_raw_dialogues('data/raw/dataset.jsonl.gz')):
    if i>=20000: break
    raws.append(d)
pr=cProfile.Profile(); t=time.time(); pr.enable()
for r in raws: clean_dialogue(r,cfg,stats,dedup)
pr.disable(); print('elapsed', time.time()-t, 'kept', stats.kept)
s=io.StringIO(); pstats.Stats(pr,stream=s).sort_stats('cumulative').print_stats(25); print(s.getvalue())
