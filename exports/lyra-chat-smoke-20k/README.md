# lyra-chat-smoke-20k

> **ВНИМАНИЕ:** это smoke-экспорт для проверки цепочки (4 шагов обучения), а не полноценно обученная модель. Ответы будут бессвязными.

Локальная русскоязычная разговорная модель, обученная с нуля
(decoder-only Transformer, byte-level BPE). Внешние LLM/API не используются.

## Состав

- `model.safetensors` — веса модели;
- `model_config.json` — архитектура;
- `tokenizer/` — обученный токенизатор (стабильные id специальных токенов);
- `generation.yaml` — параметры генерации по умолчанию;
- `export_info.json`, `training_metrics.json`, `dataset_info.json`;
- `checksums.sha256`.

## Запуск

Из корня проекта:

```bash
python scripts/chat.py --export-dir "C:\Users\erikh\OneDrive\Документы\Lyra Chat 0.1\exports\lyra-chat-smoke-20k"
```

Одна реплика без интерактивного режима:

```bash
python scripts/chat.py --export-dir "C:\Users\erikh\OneDrive\Документы\Lyra Chat 0.1\exports\lyra-chat-smoke-20k" --message "Привет! Как дела?"
```

Inference работает на CPU (медленнее) и CUDA. Требуются только PyTorch, tokenizers,
numpy, PyYAML, safetensors (см. `requirements.txt`).

## Данные и ограничения

Источник: Den4ikAI/russian_dialogues_2 (лицензия MIT).
Роли говорящих в исходном корпусе отсутствуют и были восстановлены эвристикой
(чередование, начиная с пользователя). Модель небольшая (~104.3M
параметров), поэтому ошибки, повторы и фактические неточности ожидаемы.
Метрики: train loss 8.822046279907227, validation loss
9.028329902971295, best val loss 9.028329902971295.
