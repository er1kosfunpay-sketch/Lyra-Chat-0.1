import torch
from conversation_llm.tokenizer import train_tokenizer, ConversationTokenizer, SPECIAL_TOKEN_IDS
from conversation_llm.config import ModelConfig
from conversation_llm.model import GPT

texts = ["Привет, как дела?", "Нормально, а у тебя?", "Всё хорошо, спасибо! 😊", "Hello world 123", "длинноеслово"*10]
tok = train_tokenizer(iter(texts*50), vocab_size=512)
w = ConversationTokenizer(tok, tok.get_vocab_size(), {k: tok.token_to_id(k) for k in SPECIAL_TOKEN_IDS})
for s in ["Привет, как дела?", "Hello world 123", "Привет 😊 ёжик", "  пробелы  "]:
    ids = w.encode(s); out = w.decode(ids)
    print(repr(s), len(ids), repr(out), s==out)
ids = w.encode_turn("user", "Привет"); print(ids, w.decode(ids), w.decode(ids, skip_special_tokens=True))
cfg = ModelConfig(vocab_size=w.vocab_size, n_layer=2, n_head=4, n_embd=64, block_size=64, dropout=0.0)
m = GPT(cfg); print("params", m.num_parameters())
x=torch.randint(0,cfg.vocab_size,(2,16)); y=torch.randint(0,cfg.vocab_size,(2,16))
o=m(x, labels=y); print(o.logits.shape, o.loss.item()); o.loss.backward(); print("grad", next(m.parameters()).grad is not None)
# cache equivalence
m.eval()
with torch.no_grad():
    full=m(x[:1,:8]).logits
    first=m(x[:1,:4], use_cache=True)
    second=m(x[:1,4:8], past_key_values=first.past_key_values, use_cache=True)
    print("cache diff", (full[:,-4:]-second.logits).abs().max().item())
