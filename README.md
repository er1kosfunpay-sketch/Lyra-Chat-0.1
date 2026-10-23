# Lyra Chat 0.1 — русская разговорная LLM с нуля

Небольшая decoder-only языковая модель (GPT-1/GPT-2-класса), полностью собранная
в этом проекте: свой byte-level BPE-токенизатор, свой корпус, свой цикл обучения,
локальный инференс, оценка и экспорт. **Предобученные веса LLM и внешние API не
используются.** Все тексты генерируются только локальной моделью.

> **Честный статус.** Код, токенизатор, подготовка данных и тесты проверены на
> этой машине (Windows, Python 3.12, CPU-only PyTorch). Полноценное обучение
> 104M-модели на GPU здесь **не выполнялось**: `torch.cuda.is_available()` равно
> `False`, NVIDIA GPU отсутствует. Для облачного обучения подготовлен ноутбук
> `notebooks/train_free_gpu.ipynb` (Kaggle T4 / Colab). Результаты smoke-прогона
> и оценки явно помечены ниже; не путайте их с полноценным обучением.

---

## 1. Архитектура

| Параметр | Значение |
| --- | ---: |
| Тип | decoder-only Transformer, pre-LN |
| Слоёв | 12 |
| Голов внимания | 12 |
| Скрытая размерность | 768 |
| FFN | 3072, GELU (tanh) |
| Контекст | 1024 токена |
| Словарь | 24 000 (byte-level BPE) |
| Позиции | обучаемые, до 1024 |
| Связывание весов | embedding ↔ lm_head |
| Dropout | 0.1 |
| Параметров | **104 274 432** |

Реализация: `src/conversation_llm/model/`. Используется
`torch.nn.functional.scaled_dot_product_attention` с явной causal/padding-маской
и KV-кэшем для генерации. Архитектура и размер проверяются тестами.

### Токенизатор

Byte-level BPE (`tokenizers`), обучается на очищенном train-корпусе. Специальные
токены имеют фиксированные ID и проверяются при каждой загрузке:

| ID | Токен |
| ---: | --- |
| 0 | `<|pad|>` |
| 1 | `<|unk|>` |
| 2 | `<|bos|>` |
| 3 | `<|eos|>` |
| 4 | `<|system|>` |
| 5 | `<|user|>` |
| 6 | `<|assistant|>` |
| 7 | `<|turn_end|>` |

Пользовательский текст санитизируется, чтобы нельзя было «ввести» служебный
токен через сообщение. Round-trip проверен на русском, английском, эмодзи,
пробелах и длинных словах (`tests/test_tokenizer.py`).

---

## 2. Данные

Источник: **`Den4ikAI/russian_dialogues_2`**, лицензия MIT, файл
`dataset.jsonl.gz` (~271.7 МБ, 1 701 649 записей). В исходнике только поле
`sample` — упорядоченный список реплик **без меток говорящих**.

### Роли

Так как меток нет, роли восстанавливаются эвристикой: чередование, начиная с
`user`. Это **не ground truth**, а приближение; в каждом диалоге сохраняется
`roles_inferred: true` и `confidence`. На 20k-подвыборке средняя уверенность
составила ~0.54, а доля вопросов в «чётных» репликах (0.158) лишь немного выше,
чем в «нечётных» (0.120). Ограничение явно отражено в отчётах, экспорте и
README — модель не выдаёт себя за размеченный диалоговый датасет.

### Очистка и сплит

- нормализация Unicode, исправление типичной mojibake, удаление control/zero-width;
- отбрасывание пустых, бессмысленных, спамных и деградировавших реплик;
- маскирование URL, e-mail, телефонов, @упоминаний и длинных номеров;
- ограничение длины реплики/диалога, удаление хвоста слишком длинных тредов;
- точные и near-duplicate (мешок слов) дубликаты;
- сплит **98/1/1 по диалогу до нарезки на окна**; ключ — стабильный BLAKE2b
  от нормализованного текста, поэтому реплики одного диалога не расходятся по
  разным сплитам;
- test-сплит не используется ни для обучения, ни для выбора чекпойнта.

### Два формата

1. **Pretrain** — непрерывный поток токенов:
   `<|bos|> <|user|> … <|turn_end|> <|assistant|> … <|turn_end|> <|eos|>`.
2. **SFT** — примеры с ответом только на токенах ассистента:
   промпт (включая предыдущие реплики) имеет labels `-100`, целевой ответ и
   `<|turn_end|>` — реальные labels. Промпт всегда начинается с `user`,
   контекст ограничен 8 репликами и обрезается до лимита 1024 токена.

---

## 3. Быстрый старт (CPU)

Требуется Python 3.10+.

```bash
python -m pip install -r requirements.txt
python scripts/check_env.py
```

Быстрая проверка на 20 000 диалогах (несколько минут):

```bash
python scripts/prepare_data.py --limit 20000
```

Полный корпус (около 1.7 млн диалогов; подготовка занимает десятки минут):

```bash
python scripts/prepare_data.py
```

Повторный запуск использует кэш. `--force` пересобирает всё.

### Обучение

```bash
# Предобучение
python scripts/train.py --stage pretrain --run-dir runs/pretrain \
  --device auto --max-steps 6000 --max-minutes 600

# SFT: инициализация весами лучшего pretrain-чекпойнта
python scripts/train.py --stage sft --run-dir runs/sft \
  --device auto --max-steps 4000 --max-minutes 600 \
  --init-checkpoint runs/pretrain/checkpoints/best.pt
```

`--device auto` выберет CUDA, если она есть, иначе CPU. На CPU полное обучение
практически нецелесообразно; для проверки цепочки используйте `--smoke` и
`--allow-cpu`. На Windows число workers автоматически сбрасывается в 0.

Что делает тренер:

- AMP `auto` (BF16 на Ampere+, FP16 + GradScaler на T4), `off`, `bf16`, `fp16`;
- gradient accumulation и clipping, AdamW, warmup + cosine;
- периодическая валидация, `best.pt` и `last.pt`, ротация `step_*.pt`;
- точное продолжение: веса, optimizer, scheduler, scaler, RNG, номер шага,
  счётчики токенов и **позиция в потоке данных** (эпоха + батч);
- graceful stop по времени с сохранением чекпойнта;
- защита от NaN/Inf (пропуск шага, остановка после 5 подряд) и понятная ошибка
  при CUDA OOM без «тихого» уменьшения батча.

Продолжить после лимита времени или разрыва сессии:

```bash
python scripts/train.py --stage pretrain --run-dir runs/pretrain \
  --device auto --resume auto
```

### Оценка

```bash
python scripts/evaluate.py \
  --checkpoint runs/sft/checkpoints/best.pt \
  --tokenizer artifacts/tokenizer \
  --device auto \
  --max-batches 50 \
  --output reports/evaluation
```

Скрипт считает val/test loss и perplexity (для SFT — только по response-токенам),
а также генерирует 10 фиксированных сценариев (приветствие, сленг, контекст,
смена темы, эмоции, память, многоходовость и т.д.). Сохраняются
`evaluation_report.json`, `evaluation_report.md` и `generations.jsonl`.
Примеры **не фильтруются**: пустые и плохие ответы видны в отчёте.

> Perplexity на разных сплитах/форматах несравнима напрямую: у SFT loss считается
> только по токенам ответа, у pretrain — по всем токенам.

### Чат

```bash
python scripts/chat.py \
  --checkpoint runs/sft/checkpoints/best.pt \
  --tokenizer artifacts/tokenizer \
  --device auto
```

Или одна реплика:

```bash
python scripts/chat.py --export-dir exports/lyra-chat-sft \
  --device cpu --message "Привет! Как дела?" --max-new-tokens 60
```

В интерактивном режиме доступны `/reset`, `/history`, `/length N`, `/temp X`,
`/topk N`, `/topp X`, `/help`, `/quit`. Генерация останавливается на
`<|turn_end|>`, `<|eos|>`, ролевых токенах и `<|pad|>`; работают repetition
penalty и запрет повторяющихся n-грамм.

### Экспорт

```bash
python scripts/export_model.py \
  --checkpoint runs/sft/checkpoints/best.pt \
  --tokenizer artifacts/tokenizer \
  --evaluation-report reports/evaluation \
  --output-dir exports \
  --name lyra-chat-sft \
  --overwrite
```

Экспорт содержит `model.safetensors`, `model_config.json`, токенизатор,
`generation.yaml`, `export_info.json`, `training_metrics.json`,
`dataset_info.json`, `checksums.sha256` и ZIP-архив. Скрипт загружает экспорт в
**новом процессе** и делает короткую генерацию — это проверка самодостаточности,
а не оценка качества.

---

## 4. Бесплатный GPU: Kaggle / Colab

Ноутбук: **`notebooks/train_free_gpu.ipynb`**.

1. Создайте архив проекта (готовые архивы лежат в `dist/`).
2. Kaggle → *Datasets* → *Upload* → загрузите `lyra-chat-with-data.zip`
   (с сырым корпусом, работает без интернета) или `lyra-chat-project.zip`
   (лёгкий; тогда включите *Settings → Internet*).
3. Откройте новый Notebook, добавьте датасет через *Add Input*.
4. *Settings → Accelerator → GPU T4 x2* (используется один GPU).
5. Включите *Persistence → Files* или сохраняйте версию ноутбука.
6. Запускайте ячейки по порядку.

Ноутбук сам находит проект, копирует его в `/kaggle/working/lyra-chat`,
проверяет зависимости и GPU, готовит данные, делает smoke-прогон, затем
предобучение, SFT, оценку, экспорт и проверку чата. Обучение можно прерывать:
`--resume auto` продолжит с `checkpoints/last.pt`, включая позицию в данных.

Colab: загрузите архив в `/content`, включите T4 GPU, при необходимости
поставьте `USE_DRIVE = True` в первой ячейке. Бесплатный Colab часто обрывает
сессии — чекпойнты и `--resume auto` обязательны.

---

## 5. Структура проекта

```text
configs/                 model.yaml, data.yaml, training.yaml, generation.yaml
scripts/                 prepare_data, train_tokenizer, train, evaluate, chat, export, check_env, package_project
src/conversation_llm/
  config.py              типизированные конфиги
  tokenizer.py           byte-level BPE + стабильные special tokens
  data/                  downloader, cleaning, roles, quality, tokenization, batching
  model/                 GPT, causal attention, init
  training/              trainer, checkpoints, optimizer, scheduler, metrics
  inference/             загрузка, сэмплирование, KV-кэш, REPL
  evaluation/            loss/perplexity, сценарии, отчёты
  utils/                 device/AMP, окружение, логи, воспроизводимость
tests/                   50 тестов
notebooks/               ноутбук для Kaggle/Colab
dist/                    архивы проекта для загрузки в Kaggle
```

---

## 6. Проверенные результаты

### Тесты

```bash
# PowerShell
$env:PYTHONPATH="src"; python -m pytest -q
# bash
PYTHONPATH=src python -m pytest -q
```

На этой машине: **50 passed** (данные, токенизатор, модель, обучение,
чекпойнты/резюме, генерация, экспорт). Это проверяет инженерную целостность,
но не качество диалогов.

### Подготовка на 20k-подвыборке (реально выполнена)

| Метрика | Значение |
| --- | ---: |
| Прочитано диалогов | 20 000 |
| Оставлено диалогов | 19 999 |
| Оставлено реплик | 90 473 |
| train / val / test (диалоги) | 19 610 / 207 / 182 |
| Pretrain train токенов | 1 583 128 |
| Pretrain val / test токенов | 17 011 / 14 697 |
| SFT train примеров | 41 957 |
| SFT train токенов (все / response) | 2 413 151 / 676 251 |
| PII-маскирование | 3 телефона, 7 длинных номеров |

Полный корпус обрабатывается отдельно; итоговые цифры появятся в
`reports/data_report.json` и `reports/tokenization_report.json` после завершения.
(Если в вашей копии отчёт содержит `limit: 20000` — значит, полный прогон ещё не
выполнен.)

### Smoke-обучение (реально выполнено, только проверка цепочки)

На 20k-подвыборке, CPU, полная конфигурация 104M, 4 шага SFT (batch 2,
grad accumulation 1, всего 652 токена):

| Метрика | Значение |
| --- | ---: |
| train loss (последний) | 8.8220 |
| val loss | 9.0283 (ppl 8336) |
| eval val loss (5 батчей) | 8.9813 (ppl 7953) |
| eval test loss (5 батчей) | 8.8706 (ppl 7120) |
| сгенерировано ответов | 16, все непустые, stop rate 1.0 |
| средняя длина ответа | 5.25 символа |

Ответы после 4 шагов — мусор (`«на»`, `«,»`, отдельные буквы); это **ожидаемо**
и не скрывается. Отчёт: `reports/smoke_evaluation/evaluation_report.md`.
Экспорт: `exports/lyra-chat-smoke-20k/` (+ ZIP), проверен в новом процессе:
`EXPORT_OK 104274432 <|turn_end|>`.

Этот smoke-прогон доказывает, что данные, forward/backward, AMP-путь, eval,
чекпойнты, экспорт и чат работают. Он **не является** обученной моделью.

### Полное GPU-обучение

**Не выполнено на этой машине.** Ноутбук и конфиги готовы; запуск требует
Kaggle/Colab с GPU. Не считайте проект «обученной моделью», пока вы не выполнили
ноутбук и не получили реальные `training_report.json` и `evaluation_report.md`.

---

## 7. Ограничения

- Роли в корпусе восстановлены эвристикой; диалоги форумные, а не
  ассистентские. Качество ответов ограничено не только размером модели, но и
  разметкой.
- 104M параметров и несколько тысяч шагов на бесплатном GPU — это уровень
  «небольшая экспериментальная модель», а не ChatGPT. Ожидайте короткие,
  иногда бессвязные ответы, повторы и фактические ошибки.
- Полный корпус (~130M pretrain-токенов) при 6000 шагах × 61 440 токенов
  означает несколько эпох; SFT при 4000 шагах видит лишь малую часть примеров.
  Для заметно лучшего качества нужны больше шагов/данных и чистые метки ролей.
- CPU-инференс возможен, но медленный; 8 потоков этой машины дают считанные
  токены в секунду.
- Test-сплит не использовался для выбора модели, но из-за ограниченного объёма
  метрики имеют широкие доверительные интервалы.

---

## 8. Лицензия и данные

Код проекта — MIT. Исходный датасет `Den4ikAI/russian_dialogues_2` — MIT;
сырой корпус в репозиторий не включён (кроме опционального архива в `dist/`).
При публикации модели укажите происхождение данных и ограничения ролевой
разметки.
