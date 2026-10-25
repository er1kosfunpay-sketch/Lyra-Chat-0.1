"""Build a Kaggle-only notebook that uses prebuilt tokenizer/data artifacts."""

from __future__ import annotations

import json
from pathlib import Path

cells: list[dict] = []


def md(source: str) -> None:
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": source.strip("\n").splitlines(keepends=True),
        }
    )


def code(source: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.strip("\n").splitlines(keepends=True),
        }
    )


md(
    r"""
# Lyra Chat 0.1 — обучение на Kaggle

Ноутбук рассчитан **только на Kaggle Notebooks** и использует готовые artifacts:
токенизатор и уже токенизированные данные. Заново скачивать и токенизировать корпус
не нужно.

## Что загрузить в Kaggle

Создайте два **Kaggle Dataset** и добавьте их через **Add Input**:

1. `lyra-chat-project.zip` — код проекта.
2. `lyra-artifacts.tar.gz` — `artifacts/tokenizer` + `artifacts/data/tokenized`.

Можно также распаковать `lyra-artifacts.tar.gz` локально и загрузить папку
`artifacts/` как Dataset — ноутбук найдёт файлы сам.

## Настройки ноутбука

- **Accelerator:** GPU T4 x2 (используется один GPU).
- **Internet:** On — для установки зависимостей; с готовыми artifacts можно Off.
- **Persistence:** Files only, либо сохраняйте версию ноутбука.
- **Disk:** T4 x2 даёт около 20 GB в `/kaggle/working`.

## Порядок запуска

1. Ячейки 1–6: окружение, параметры, artifacts, проверка.
2. Ячейка smoke: быстрый тест.
3. Pretrain.
4. SFT.
5. Evaluation, export, chat.

**Не запускайте Run All**, если pretrain + SFT не влезают в 12 часов. Лучше разнести
этапы по сессиям: pretrain → Save Version → новая сессия → SFT → eval/export.

## Быстрый тест

В ячейке с параметрами поставьте:

```python
PRETRAIN_STEPS = 300
SFT_STEPS = 200
MAX_MINUTES = 60
```
"""
)

md("## 1. Окружение, проект и GPU")

code(
    r'''
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
PROJECT = WORK / "lyra-chat"

if not WORK.exists():
    raise SystemExit("Этот ноутбук рассчитан на Kaggle Notebooks: нет /kaggle/working")


def looks_like_project(path: Path) -> bool:
    return (
        (path / "src" / "conversation_llm").is_dir()
        and (path / "scripts" / "train.py").exists()
    )


def find_project() -> Path:
    # Если проект уже скопирован в working ( persistence / resume ), используем его.
    if looks_like_project(PROJECT):
        return PROJECT.resolve()

    candidates = [Path.cwd(), Path.cwd().parent]

    if INPUT.exists():
        candidates.extend(path.parent for path in INPUT.glob("**/pyproject.toml"))

        for archive in INPUT.glob("**/*.zip"):
            if "lyra" not in archive.name.lower() and "chat" not in archive.name.lower():
                continue
            target = WORK / "extracted" / archive.stem
            if not target.exists():
                target.mkdir(parents=True, exist_ok=True)
                print("Распаковываю", archive)
                shutil.unpack_archive(str(archive), str(target))
            candidates.append(target)
            candidates.extend(target.glob("**"))

    for candidate in candidates:
        if looks_like_project(candidate):
            return candidate.resolve()

    raise FileNotFoundError(
        "Проект не найден. Загрузите lyra-chat-project.zip как Kaggle Dataset "
        "и добавьте его через Add Input."
    )


SOURCE = find_project()

if SOURCE != PROJECT.resolve():
    if PROJECT.is_symlink():
        PROJECT.unlink()
    elif PROJECT.is_file():
        PROJECT.unlink()
    elif PROJECT.exists() and not looks_like_project(PROJECT):
        shutil.rmtree(PROJECT)

    if not looks_like_project(PROJECT):
        shutil.copytree(
            SOURCE,
            PROJECT,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                ".git",
                ".pytest_cache",
                "artifacts",
                "exports",
                "data",
                "*.zip",
            ),
        )

        # Если архив содержит data/raw, переносим его отдельно.
        source_raw = SOURCE / "data" / "raw"
        if source_raw.is_dir():
            target_raw = PROJECT / "data" / "raw"
            target_raw.mkdir(parents=True, exist_ok=True)
            for item in source_raw.iterdir():
                if item.is_file() and not item.name.endswith(".part"):
                    shutil.copy2(item, target_raw / item.name)

os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / "src"))


packages = {
    "torch": None,
    "tokenizers": "tokenizers>=0.15,<1",
    "numpy": None,
    "yaml": "PyYAML>=6.0,<7",
    "tqdm": "tqdm>=4.66,<5",
    "safetensors": "safetensors>=0.4,<1",
    "requests": "requests>=2.31,<3",
    "datasets": "datasets>=2.19,<6",
    "huggingface_hub": "huggingface_hub>=0.23,<1",
}
missing = [
    requirement
    for module, requirement in packages.items()
    if requirement is not None and importlib.util.find_spec(module) is None
]
if missing:
    print("Устанавливаю:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
else:
    print("Зависимости на месте; PyTorch не переустанавливается.")

import torch

if not torch.cuda.is_available():
    raise SystemExit("GPU не найден: Settings -> Accelerator -> GPU T4 x2")

props = torch.cuda.get_device_properties(0)
print("Project:", PROJECT)
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("GPU:", props.name, f"{props.total_memory / 1024**3:.1f} GB")
print("GPU count:", torch.cuda.device_count())


RUN_PRETRAIN = PROJECT / "runs" / "pretrain"
RUN_SFT = PROJECT / "runs" / "sft"
RUN_SMOKE = PROJECT / "runs" / "smoke"
EVAL_DIR = PROJECT / "reports" / "evaluation"
EXPORT_DIR = PROJECT / "exports"


def run(cmd, check=True):
    print("\n$", " ".join(str(part) for part in cmd), flush=True)
    process = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(PROJECT),
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and process.returncode != 0:
        raise RuntimeError(f"Команда завершилась с кодом {process.returncode}")
    return process.returncode


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None
'''
)

md(
    r"""
## 2. Параметры запуска

`PRETRAIN_STEPS` и `SFT_STEPS` — **общее** число optimizer-шагов.
Один шаг потребляет `BATCH_SIZE * GRAD_ACCUM` микробатчей.
При `6 * 10 * 1024` это примерно 61k токенов на шаг.
"""
)

code(
    r'''
# ===== Режим данных =====
PREPARE_FROM_SCRATCH = False   # True = скачать/очистить/токенизировать заново
REDOWNLOAD_RAW = True          # True = удалить локальный raw и скачать полный
DATA_LIMIT = None              # например, 200_000 для быстрого прогона
FORCE_PREP = False             # True = перезаписать cleaned/tokenized

# ===== Обучение =====
PRETRAIN_STEPS = 6000          # всего optimizer-шагов
SFT_STEPS = 4000               # всего optimizer-шагов
MAX_MINUTES = 600              # лимит на одну сессию; Kaggle max ~12h
BATCH_SIZE = 6
GRAD_ACCUM = 10

# ===== Бэкап чекпойнтов =====
HF_REPO = None                 # например: "username/lyra-chat-runs"

print("PREPARE_FROM_SCRATCH:", PREPARE_FROM_SCRATCH)
print("PRETRAIN_STEPS:", PRETRAIN_STEPS)
print("SFT_STEPS:", SFT_STEPS)
print("MAX_MINUTES:", MAX_MINUTES)
print("HF_REPO:", HF_REPO)
'''
)

md(
    r"""
## 3. Подключение готовых artifacts

Ноутбук ищет `tokenizer.json` рядом с `special_tokens.json` и `pretrain/train/meta.json`
во всём `/kaggle/input`. Если загружен `lyra-artifacts.tar.gz`, он распакуется
в `/kaggle/working/_artifacts` (около 2.3 GB).
"""
)

code(
    r'''
if PREPARE_FROM_SCRATCH:
    print("PREPARE_FROM_SCRATCH=True: artifacts не подключаются, данные будут собраны заново")
else:
    def find_artifacts(base: Path):
        tokenizer_dir = None
        for candidate in base.rglob("tokenizer.json"):
            if (candidate.parent / "special_tokens.json").exists():
                tokenizer_dir = candidate.parent
                break

        meta = next(base.rglob("pretrain/train/meta.json"), None)
        tokenized_dir = meta.parents[2] if meta is not None else None
        return tokenizer_dir, tokenized_dir

    tokenizer_dir, tokenized_dir = find_artifacts(INPUT)

    if tokenizer_dir is None or tokenized_dir is None:
        archive = (
            next(INPUT.rglob("lyra-artifacts.tar.gz"), None)
            or next(INPUT.rglob("lyra-artifacts.tgz"), None)
            or next(INPUT.rglob("lyra-artifacts.zip"), None)
        )
        if archive is not None:
            extract_dir = WORK / "_artifacts"
            if not extract_dir.exists():
                print("Распаковываю", archive)
                archive_format = "zip" if archive.suffix.lower() == ".zip" else "gztar"
                shutil.unpack_archive(str(archive), str(extract_dir), format=archive_format)
            tokenizer_dir, tokenized_dir = find_artifacts(extract_dir)

    if tokenizer_dir is None or tokenized_dir is None:
        raise FileNotFoundError(
            "Не найдены artifacts/tokenizer и artifacts/data/tokenized. "
            "Загрузите lyra-artifacts.tar.gz как Kaggle Dataset."
        )

    dst_tokenizer = PROJECT / "artifacts" / "tokenizer"
    dst_tokenized = PROJECT / "artifacts" / "data" / "tokenized"
    dst_tokenizer.parent.mkdir(parents=True, exist_ok=True)
    dst_tokenized.parent.mkdir(parents=True, exist_ok=True)

    for dst in (dst_tokenizer, dst_tokenized):
        if dst.is_symlink() or dst.exists():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()

    for src, dst in ((tokenizer_dir, dst_tokenizer), (tokenized_dir, dst_tokenized)):
        try:
            os.symlink(src, dst, target_is_directory=True)
        except OSError:
            print("symlink недоступен, копирую:", src)
            shutil.copytree(src, dst)

    print("tokenizer:", tokenizer_dir)
    print("tokenized:", tokenized_dir)
'''
)

code(
    r'''
import json
from conversation_llm.tokenizer import ConversationTokenizer

tokenizer = ConversationTokenizer.load(PROJECT / "artifacts" / "tokenizer")
print("Tokenizer vocab:", tokenizer.vocab_size)
print("Tokenizer fingerprint:", tokenizer.fingerprint())

for stage in ("pretrain", "sft"):
    base = PROJECT / "artifacts" / "data" / "tokenized" / stage
    for split in ("train", "val"):
        meta_path = base / split / "meta.json"
        if not meta_path.exists():
            print(f"{stage}/{split}: meta.json не найден")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        keys = ("tokens", "examples", "block_size", "max_len", "response_tokens")
        print(stage, split, {key: meta[key] for key in keys if key in meta})
'''
)

md(
    r"""
## 4. Опционально: подготовка данных с нуля

Выполняйте только если `PREPARE_FROM_SCRATCH = True`. Нужен **Internet On**.
Скачивается полный `dataset.jsonl.gz` (~272 MB), затем очистка, сплиты и токенизация.
Это занимает несколько часов и требует заметно больше диска.
"""
)

code(
    r'''
if PREPARE_FROM_SCRATCH:
    for path in (
        PROJECT / "artifacts" / "tokenizer",
        PROJECT / "artifacts" / "data" / "tokenized",
    ):
        if path.is_symlink():
            path.unlink()

    data_yaml = PROJECT / "configs" / "data.yaml"
    text = data_yaml.read_text(encoding="utf-8")
    text = text.replace(
        "https://huggingface.co/datasets/ZeroAgency/ru-instruct-conversation-v1",
        "https://huggingface.co/datasets/Den4ikAI/russian_dialogues_2/resolve/main/dataset.jsonl.gz",
    )
    text = text.replace(
        'source_name: "ZeroAgency/ru-instruct-conversation-v1"',
        'source_name: "Den4ikAI/russian_dialogues_2"',
    )
    text = text.replace(
        '  source_license: "MIT"\n  source_license: "MIT"\n',
        '  source_license: "MIT"\n',
    )
    data_yaml.write_text(text, encoding="utf-8")

    raw = PROJECT / "data" / "raw" / "dataset.jsonl.gz"
    if REDOWNLOAD_RAW and raw.exists():
        raw.unlink()
        print("Удалён старый raw: будет скачан полный файл")

    cmd = [sys.executable, "scripts/prepare_data.py"]
    if DATA_LIMIT is not None:
        cmd += ["--limit", str(DATA_LIMIT)]
    if FORCE_PREP:
        cmd.append("--force")
    run(cmd)
else:
    print("Пропущено: используются готовые artifacts")
'''
)

md(
    r"""
## 5. Smoke-проверка

Быстрый прогон SFT на несколько шагов. Проверяет данные, tokenizer, forward/backward,
validation и сохранение чекпойнта. После проверки чекпойнты удаляются, чтобы не занимать диск.
"""
)

code(
    r'''
run([
    sys.executable, "scripts/train.py",
    "--stage", "sft",
    "--run-dir", str(RUN_SMOKE),
    "--resume", "none",
    "--smoke",
    "--device", "cuda",
])

report_path = RUN_SMOKE / "training_report.json"
if report_path.exists():
    report = json.loads(report_path.read_text(encoding="utf-8"))
    keys = ("steps", "final_train_loss", "final_val_loss", "stop_reason")
    print("Smoke:", {key: report.get(key) for key in keys})

shutil.rmtree(RUN_SMOKE, ignore_errors=True)
print("Smoke-чекпойнты удалены")
'''
)

md(
    r"""
## 6. Предобучение

`--resume auto` продолжит с `runs/pretrain/checkpoints/last.pt`, если сессия уже
запускалась. При достижении лимита времени тренер сохраняет чекпойнт и завершает работу.
"""
)

code(
    r'''
run([
    sys.executable, "scripts/train.py",
    "--stage", "pretrain",
    "--run-dir", str(RUN_PRETRAIN),
    "--resume", "auto",
    "--device", "cuda",
    "--max-steps", str(PRETRAIN_STEPS),
    "--max-minutes", str(MAX_MINUTES),
    "--batch-size", str(BATCH_SIZE),
    "--grad-accum-steps", str(GRAD_ACCUM),
])
'''
)

code(
    r'''
report_path = RUN_PRETRAIN / "training_report.json"
if report_path.exists():
    report = json.loads(report_path.read_text(encoding="utf-8"))
    keys = (
        "steps",
        "tokens_seen",
        "best_val_loss",
        "final_train_loss",
        "final_val_loss",
        "stop_reason",
    )
    print(json.dumps({key: report.get(key) for key in keys}, ensure_ascii=False, indent=2))
    print("checkpoints:", sorted(path.name for path in (RUN_PRETRAIN / "checkpoints").glob("*.pt")))
    print(f"runs: {dir_size(PROJECT / 'runs') / 1e9:.2f} GB")
else:
    print("training_report.json не найден — обучение ещё не завершало шаг")
'''
)

md(
    r"""
## 7. SFT

При первом запуске веса инициализируются из `runs/pretrain/checkpoints/best.pt`.
При повторном `--resume auto` восстанавливает SFT-чекпойнт вместе с оптимизатором
и scheduler. Перед SFT удаляются промежуточные pretrain-чекпойнты, чтобы освободить диск.
"""
)

code(
    r'''
best_pretrain = RUN_PRETRAIN / "checkpoints" / "best.pt"
last_sft = RUN_SFT / "checkpoints" / "last.pt"

if not last_sft.exists() and not best_pretrain.exists():
    raise FileNotFoundError("Нет pretrain best.pt — сначала выполните предобучение")

for path in (RUN_PRETRAIN / "checkpoints").glob("step_*.pt"):
    path.unlink(missing_ok=True)

init_args = [] if last_sft.exists() else ["--init-checkpoint", str(best_pretrain)]

run([
    sys.executable, "scripts/train.py",
    "--stage", "sft",
    "--run-dir", str(RUN_SFT),
    "--resume", "auto",
    "--device", "cuda",
    "--max-steps", str(SFT_STEPS),
    "--max-minutes", str(MAX_MINUTES),
    "--batch-size", str(BATCH_SIZE),
    "--grad-accum-steps", str(GRAD_ACCUM),
    *init_args,
])
'''
)

code(
    r'''
report_path = RUN_SFT / "training_report.json"
if report_path.exists():
    report = json.loads(report_path.read_text(encoding="utf-8"))
    keys = (
        "steps",
        "tokens_seen",
        "best_val_loss",
        "final_train_loss",
        "final_val_loss",
        "stop_reason",
    )
    print(json.dumps({key: report.get(key) for key in keys}, ensure_ascii=False, indent=2))
    print("checkpoints:", sorted(path.name for path in (RUN_SFT / "checkpoints").glob("*.pt")))
    print(f"runs: {dir_size(PROJECT / 'runs') / 1e9:.2f} GB")
else:
    print("training_report.json не найден")
'''
)

code(
    r'''
# Оставляем best.pt и last.pt, удаляем промежуточные step_*.pt.
for run_dir in (RUN_PRETRAIN, RUN_SFT):
    for path in (run_dir / "checkpoints").glob("step_*.pt"):
        path.unlink(missing_ok=True)

print(f"runs: {dir_size(PROJECT / 'runs') / 1e9:.2f} GB")
print(f"free: {shutil.disk_usage(PROJECT).free / 1e9:.2f} GB")
'''
)

md(
    r"""
## 8. Оценка на validation/test

Считает loss на val/test и генерирует фиксированные сценарии для `evaluation_report.md`.
"""
)

code(
    r'''
best_sft = RUN_SFT / "checkpoints" / "best.pt"
if not best_sft.exists():
    best_sft = RUN_SFT / "checkpoints" / "last.pt"
if not best_sft.exists():
    raise FileNotFoundError("Нет SFT-чекпойнта")

run([
    sys.executable, "scripts/evaluate.py",
    "--stage", "sft",
    "--checkpoint", str(best_sft),
    "--tokenizer", str(PROJECT / "artifacts" / "tokenizer"),
    "--device", "cuda",
    "--max-batches", "50",
    "--output", str(EVAL_DIR),
])
'''
)

code(
    r'''
report_path = EVAL_DIR / "evaluation_report.json"
if report_path.exists():
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print("metrics:", json.dumps(report.get("metrics", {}), ensure_ascii=False, indent=2))
    print("val loss:", (report.get("loss", {}).get("val") or {}).get("loss"))
    print("test loss:", (report.get("loss", {}).get("test") or {}).get("loss"))
    print("elapsed_sec:", report.get("elapsed_sec"))

    md_path = EVAL_DIR / "evaluation_report.md"
    if md_path.exists():
        print("\n" + md_path.read_text(encoding="utf-8")[:6000])
else:
    print("evaluation_report.json не найден")
'''
)

md(
    r"""
## 9. Экспорт самодостаточного архива

Создаёт `exports/lyra-chat-sft/` и zip для inference.
"""
)

code(
    r'''
checkpoint = RUN_SFT / "checkpoints" / "best.pt"
if not checkpoint.exists():
    checkpoint = RUN_SFT / "checkpoints" / "last.pt"
if not checkpoint.exists():
    raise FileNotFoundError("Нет SFT-чекпойнта")

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
run([
    sys.executable, "scripts/export_model.py",
    "--checkpoint", str(checkpoint),
    "--tokenizer", str(PROJECT / "artifacts" / "tokenizer"),
    "--evaluation-report", str(EVAL_DIR),
    "--output-dir", str(EXPORT_DIR),
    "--name", "lyra-chat-sft",
    "--overwrite",
])
print("export:", EXPORT_DIR / "lyra-chat-sft")
print("zip:", EXPORT_DIR / "lyra-chat-sft.zip")
'''
)

md("## 10. Проверка чата с экспортированными весами")

code(
    r'''
run([
    sys.executable, "scripts/chat.py",
    "--export-dir", str(EXPORT_DIR / "lyra-chat-sft"),
    "--device", "cuda",
    "--message", "Привет! Как дела?",
    "--max-new-tokens", "60",
])
'''
)

md(
    r"""
## 11. Бэкап чекпойнтов в Hugging Face (опционально)

Добавьте Kaggle Secret `HF_TOKEN` и укажите `HF_REPO` в ячейке параметров.
Загружаются только `best.pt`, `last.pt` и отчёты, без промежуточных `step_*.pt`.
"""
)

code(
    r'''
if HF_REPO:
    from huggingface_hub import HfApi

    token = hf_token()
    if not token:
        raise SystemExit(
            "Нет HF_TOKEN: добавьте Secret HF_TOKEN в Kaggle "
            "или задайте переменную окружения"
        )

    api = HfApi(token=token)
    api.create_repo(HF_REPO, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=HF_REPO,
        folder_path=str(PROJECT / "runs"),
        path_in_repo="runs",
        repo_type="model",
        allow_patterns=[
            "**/best.pt",
            "**/last.pt",
            "**/training_report.json",
            "**/run_config.json",
            "**/train.jsonl",
            "**/train.log",
        ],
    )
    print("Загружено в", HF_REPO)
else:
    print("HF_REPO=None — бэкап пропущен")
'''
)

md(
    r"""
### Восстановление в новой сессии

Выполняйте **до** продолжения обучения. Не запускайте после обучения: remote `last.pt`
может перезаписать более новый локальный чекпойнт.
"""
)

code(
    r'''
if HF_REPO:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPO,
        repo_type="model",
        local_dir=str(PROJECT),
        token=hf_token(),
        allow_patterns=[
            "runs/**/best.pt",
            "runs/**/last.pt",
            "runs/**/training_report.json",
            "runs/**/run_config.json",
            "runs/**/train.jsonl",
            "runs/**/train.log",
        ],
    )
    print("Чекпойнты восстановлены в", PROJECT / "runs")
else:
    print("HF_REPO=None — скачивание пропущено")
'''
)

md("## 12. Очистка и диагностика")

code(
    r'''
shutil.rmtree(RUN_SMOKE, ignore_errors=True)

for run_dir in (RUN_PRETRAIN, RUN_SFT):
    for path in (run_dir / "checkpoints").glob("step_*.pt"):
        path.unlink(missing_ok=True)

for label, path in (
    ("artifacts", PROJECT / "artifacts"),
    ("runs", PROJECT / "runs"),
    ("exports", PROJECT / "exports"),
    ("reports", PROJECT / "reports"),
):
    print(f"{label}: {dir_size(path) / 1e9:.2f} GB")

print(f"free disk: {shutil.disk_usage(PROJECT).free / 1e9:.2f} GB")

# Опционально: уменьшить output, удалив raw/cleaned
# shutil.rmtree(PROJECT / "data" / "raw", ignore_errors=True)
# shutil.rmtree(PROJECT / "artifacts" / "data" / "cleaned", ignore_errors=True)
'''
)

md(
    r"""
## Памятка и типичные ошибки

### Продолжение обучения

- `PRETRAIN_STEPS` / `SFT_STEPS` — общее число optimizer-шагов.
  Если сессия остановилась на 3500/6000, следующий запуск продолжит с 3500.
  Если 6000 уже достигнуто — увеличьте число (например, до 10000),
  иначе обучение сразу остановится.
- `MAX_MINUTES=600` оставляет запас до лимита Kaggle. Для T4 x2 можно поставить 660.
- `best.pt` — лучший по val loss; `last.pt` — для resume; `step_*.pt` — промежуточные
  (prune до 3). Перед SFT и eval промежуточные удаляются.

### Ошибки

- **Project not found** — загрузите `lyra-chat-project.zip` как Kaggle Dataset
  и добавьте через Add Input.
- **Artifacts not found** — загрузите `lyra-artifacts.tar.gz`; проверьте, что рядом
  с `tokenizer.json` есть `special_tokens.json`.
- **GPU not found** — Settings → Accelerator → GPU T4 x2.
- **CUDA out of memory** — уменьшите `BATCH_SIZE` до 4 и увеличьте `GRAD_ACCUM` до 15,
  либо уменьшите `block_size`/`sft_max_len` в конфигах.
- **Disk full** — удалите `runs/*/checkpoints/step_*.pt`, smoke и старые run-каталоги.
  Для бэкапа используйте HF.
- **No pretrain best.pt** — сначала выполните pretrain.
- **Loss NaN** — тренер пропускает non-finite шаги и падает после 5 подряд;
  уменьшите learning rate.
- **Kaggle 12h** — запускайте этапы в разных сессиях, `--resume auto` продолжит.

### Важно про artifacts

Готовые artifacts — симлинк на `/kaggle/input`, поэтому в output ноутбука они не попадут.
Скачивайте `lyra-artifacts.tar.gz` отдельно или используйте HF-бэкап.

### Если обновили код проекта

Ноутбук сначала ищет уже существующий `/kaggle/working/lyra-chat` и использует его.
Если вы загрузили новую версию `lyra-chat-project.zip`, удалите старую папку
`/kaggle/working/lyra-chat` (или запустите новую сессию), иначе будет использован старый код.

### Что честно ожидать

Это 104M-модель с ограниченным pretrain-бюджетом. Ответы могут быть короткими,
иногда бессвязными и с повторами. Оценивайте реальные примеры из
`reports/evaluation/evaluation_report.md`, а не только loss.

В текущем корпусе специальный токен `<|system|>` не встречался, поэтому системный
промпт может игнорироваться моделью. Это ограничение данных, а не ноутбука.
"""
)

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = Path("notebooks/kaggle_train_artifacts.ipynb")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", path, "cells", len(cells))
