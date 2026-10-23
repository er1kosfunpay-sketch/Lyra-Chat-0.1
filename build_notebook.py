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
    """
# Lyra Chat — обучение разговорной LLM с нуля на бесплатном GPU

Ноутбук для **Kaggle Notebooks** (основной вариант) и **Google Colab** (альтернатива).
PyTorch, токенизатор и веса создаются с нуля; внешние LLM/API не используются.

## Перед запуском

1. Загрузите архив проекта (`lyra-chat-project.zip`) как **Kaggle Dataset** (Add Input → Upload).
   Ноутбук найдёт `pyproject.toml` и `scripts/train.py` во входных данных, скопирует
   проект в `/kaggle/working/lyra-chat` и будет писать артефакты туда.
   Если в архиве нет `data/raw/dataset.jsonl.gz`, включите *Settings → Internet*:
   корпус скачается с Hugging Face. Архив `lyra-chat-with-data.zip` уже содержит
   сырой корпус и работает без интернета.
2. Включите GPU: *Settings → Accelerator → GPU T4 x2* (используется один GPU).
3. Включите *Settings → Persistence → Files* (или сохраняйте версию ноутбука), чтобы
   чекпойнты остались после сессии.
4. Запускайте ячейки по порядку. Обучение можно прерывать и продолжать: чекпойнты
   лежат в `runs/*/checkpoints`, а `--resume auto` продолжит с последнего шага.

## Бюджет

Полная подготовка корпуса (1.7 млн диалогов) на Kaggle занимает около 1–1.5 часа CPU.
Обучение 104M-модели на T4 идёт медленнее, чем хотелось бы; за сессию реально пройти
несколько тысяч шагов. Чекпойнты и `--resume auto` позволяют продолжить в следующей
сессии. Если нужно быстрее — задайте `DATA_LIMIT` в ячейке конфигурации (например,
`400000` диалогов) или уменьшите `PRETRAIN_STEPS`/`SFT_STEPS`.
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

IS_KAGGLE = Path("/kaggle").exists()
IS_COLAB = "google.colab" in sys.modules or Path("/content").exists()
WORK_BASE = Path("/kaggle/working") if IS_KAGGLE else Path("/content")
WORK_BASE.mkdir(parents=True, exist_ok=True)


def _looks_like_project(path: Path) -> bool:
    return (path / "src" / "conversation_llm").is_dir() and (path / "scripts" / "train.py").exists()


def find_project() -> Path:
    candidates = [
        WORK_BASE / "lyra-chat",
        WORK_BASE / "conversation-llm",
        Path.cwd(),
        Path.cwd().parent,
    ]
    for base in (Path("/kaggle/input"), Path("/content")):
        if base.exists():
            for config in base.glob("**/pyproject.toml"):
                candidates.append(config.parent)
    for candidate in candidates:
        if _looks_like_project(candidate):
            return candidate.resolve()
    for base in (Path("/kaggle/input"), Path("/content")):
        if not base.exists():
            continue
        for archive in base.glob("**/*.zip"):
            target = WORK_BASE / "extracted"
            target.mkdir(exist_ok=True)
            shutil.unpack_archive(str(archive), str(target))
            for candidate in target.glob("**"):
                if _looks_like_project(candidate):
                    return candidate.resolve()
    raise FileNotFoundError(
        "Не найден проект. Загрузите lyra-chat-project.zip как Kaggle Dataset "
        "или распакуйте проект в /content."
    )


SOURCE = find_project()
PROJECT = WORK_BASE / "lyra-chat"
if SOURCE != PROJECT.resolve():
    if not _looks_like_project(PROJECT):
        shutil.copytree(
            SOURCE,
            PROJECT,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                "__pycache__", ".git", ".pytest_cache", "artifacts", "exports", "data", "*.zip"
            ),
        )
        # Keep the downloaded raw corpus when the archive contains it.  This
        # avoids a 272 MB download in every fresh Kaggle session.
        source_raw = SOURCE / "data" / "raw"
        if source_raw.is_dir():
            target_raw = PROJECT / "data" / "raw"
            target_raw.mkdir(parents=True, exist_ok=True)
            for item in source_raw.iterdir():
                if item.is_file() and not item.name.endswith(".part"):
                    shutil.copy2(item, target_raw / item.name)
    PROJECT = PROJECT.resolve()
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / "src"))

# Colab: optionally keep checkpoints on Google Drive.
USE_DRIVE = False
PERSIST = PROJECT
if IS_COLAB and USE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    PERSIST = Path("/content/drive/MyDrive/lyra-chat")
    PERSIST.mkdir(parents=True, exist_ok=True)

packages = {
    "torch": "torch>=2.3,<3",
    "tokenizers": "tokenizers>=0.15,<1",
    "numpy": "numpy>=1.26,<3",
    "yaml": "PyYAML>=6.0,<7",
    "tqdm": "tqdm>=4.66,<5",
    "safetensors": "safetensors>=0.4,<1",
    "requests": "requests>=2.31,<3",
    "datasets": "datasets>=2.19,<6",
    "huggingface_hub": "huggingface_hub>=0.23,<1",
}
missing = [requirement for module, requirement in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    print("Устанавливаю отсутствующие пакеты:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
else:
    print("Все зависимости уже установлены; PyTorch не переустанавливается.")

import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "GPU не найден. В Kaggle включите Settings → Accelerator → GPU. "
        "Полное обучение на CPU в этом ноутбуке не запускается."
    )
props = torch.cuda.get_device_properties(0)
print("Project:", PROJECT)
print("Persist:", PERSIST)
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("GPU:", props.name, f"{props.total_memory / 1024**3:.1f} GB")
'''
)

md("## 2. Параметры запуска")

code(
    r'''
# None = весь корпус (1.7 млн диалогов). Для быстрого прогона поставьте, например, 200000.
DATA_LIMIT = None
FORCE_PREP = False

PRETRAIN_STEPS = 6000
SFT_STEPS = 4000
MAX_MINUTES_PER_SESSION = 600   # оставляем запас до лимита Kaggle (12 ч)
BATCH_SIZE = 6
GRAD_ACCUM = 10

RUN_PRETRAIN = PERSIST / "runs" / "pretrain"
RUN_SFT = PERSIST / "runs" / "sft"
RUN_SMOKE = PERSIST / "runs" / "smoke"
EVAL_DIR = PERSIST / "reports" / "evaluation"
EXPORT_DIR = PERSIST / "exports"

print("DATA_LIMIT:", DATA_LIMIT)
print("Pretrain run:", RUN_PRETRAIN)
print("SFT run:", RUN_SFT)


def run(cmd, check=True):
    print("\n$", " ".join(str(part) for part in cmd), flush=True)
    process = subprocess.run([str(part) for part in cmd], cwd=str(PROJECT), text=True)
    if check and process.returncode != 0:
        raise RuntimeError(f"Команда завершилась с кодом {process.returncode}")
    return process.returncode
'''
)

md(
    """
## 3. Подготовка данных и токенизатора

Скачивание, очистка, дедупликация, восстановление ролей, сплит 98/1/1, обучение
byte-level BPE (24k), упаковка pretrain-потока и SFT-примеров с маской только на ответе.
Повторный запуск использует кэш; `FORCE_PREP=True` пересобирает всё.
"""
)

code(
    r'''
prepare_cmd = [sys.executable, "scripts/prepare_data.py"]
if DATA_LIMIT is not None:
    prepare_cmd += ["--limit", str(DATA_LIMIT)]
if FORCE_PREP:
    prepare_cmd.append("--force")
run(prepare_cmd)

report = json.loads((PROJECT / "reports" / "data_report.json").read_text(encoding="utf-8"))
quality = report["quality"]
print("\nОставлено диалогов:", f"{quality['kept']:,}")
print("Оставлено реплик:", f"{quality['turns_kept']:,}")
print("Причины удаления диалогов:", quality["dropped"])
print("Причины удаления реплик:", quality["turns_dropped"])
print("PII:", quality["pii"])
for split, entry in report["tokenization"]["pretrain"]["splits"].items():
    print(f"pretrain/{split}: {entry['tokens']:,} токенов")
for split, entry in report["tokenization"]["sft"]["splits"].items():
    print(
        f"sft/{split}: {entry['examples']:,} примеров, "
        f"{entry['tokens']:,} токенов, {entry['response_tokens']:,} target-токенов"
    )
'''
)

md(
    """
## 4. Короткий smoke test

Проверяет реальные данные, forward/backward, loss, validation и сохранение чекпойнта
на полной конфигурации модели. Это не доказательство качества диалога, а проверка
инженерной цепочки перед многочасовым запуском.
"""
)

code(
    r'''
run(
    [
        sys.executable, "scripts/train.py",
        "--stage", "sft",
        "--run-dir", str(RUN_SMOKE),
        "--resume", "none",
        "--smoke",
        "--device", "cuda",
    ]
)
smoke_report = json.loads((RUN_SMOKE / "training_report.json").read_text(encoding="utf-8"))
print("Smoke steps:", smoke_report["steps"])
print("Smoke train loss:", smoke_report["final_train_loss"])
print("Smoke val loss:", smoke_report["final_val_loss"])
assert smoke_report["steps"] >= 1 and smoke_report["final_val_loss"] is not None
print("Smoke test пройден.")
# The smoke run uses the full 104M model and writes several GB of checkpoints.
# Keep only its report; the real runs are separate.
shutil.rmtree(RUN_SMOKE, ignore_errors=True)
print("Smoke checkpoints removed to save Kaggle output space.")
'''
)

md(
    """
## 5. Предобучение (causal LM на полных диалогах)

`--resume auto` продолжит с `runs/pretrain/checkpoints/last.pt`, если сессия уже
запускалась. При достижении лимита времени тренер сохраняет чекпойнт и завершает
работу без потери прогресса.
"""
)

code(
    r'''
run(
    [
        sys.executable, "scripts/train.py",
        "--stage", "pretrain",
        "--run-dir", str(RUN_PRETRAIN),
        "--resume", "auto",
        "--device", "cuda",
        "--max-steps", str(PRETRAIN_STEPS),
        "--max-minutes", str(MAX_MINUTES_PER_SESSION),
        "--batch-size", str(BATCH_SIZE),
        "--grad-accum-steps", str(GRAD_ACCUM),
    ]
)
pretrain_report = json.loads((RUN_PRETRAIN / "training_report.json").read_text(encoding="utf-8"))
print("Pretrain:", {key: pretrain_report[key] for key in ("steps", "tokens_seen", "best_val_loss", "stop_reason")})
'''
)

md(
    """
## 6. SFT: ответы только на токенах ассистента

При первом запуске SFT веса инициализируются из лучшего pretrain-чекпойнта.
При повторном — `--resume auto` восстанавливает SFT-чекпойнт вместе с оптимизатором
и scheduler. SFT не использует test-сплит.
"""
)

code(
    r'''
sft_checkpoint = RUN_SFT / "checkpoints" / "last.pt"
pretrain_best = RUN_PRETRAIN / "checkpoints" / "best.pt"
if not sft_checkpoint.exists() and not pretrain_best.exists():
    raise FileNotFoundError(
        "Нет pretrain best.pt. Сначала выполните предобучение или укажите другой чекпойнт."
    )

sft_cmd = [
    sys.executable, "scripts/train.py",
    "--stage", "sft",
    "--run-dir", str(RUN_SFT),
    "--resume", "auto",
    "--device", "cuda",
    "--max-steps", str(SFT_STEPS),
    "--max-minutes", str(MAX_MINUTES_PER_SESSION),
    "--batch-size", str(BATCH_SIZE),
    "--grad-accum-steps", str(GRAD_ACCUM),
]
if not sft_checkpoint.exists():
    sft_cmd += ["--init-checkpoint", str(pretrain_best)]
run(sft_cmd)
sft_report = json.loads((RUN_SFT / "training_report.json").read_text(encoding="utf-8"))
print("SFT:", {key: sft_report[key] for key in ("steps", "tokens_seen", "best_val_loss", "stop_reason")})
'''
)

md("## 7. Оценка на validation и test + фиксированные диалоги")

code(
    r'''
best = RUN_SFT / "checkpoints" / "best.pt"
if not best.exists():
    best = RUN_SFT / "checkpoints" / "last.pt"
run(
    [
        sys.executable, "scripts/evaluate.py",
        "--checkpoint", str(best),
        "--tokenizer", str(PROJECT / "artifacts" / "tokenizer"),
        "--device", "cuda",
        "--output", str(EVAL_DIR),
        "--max-batches", "50",
    ]
)
evaluation = json.loads((EVAL_DIR / "evaluation_report.json").read_text(encoding="utf-8"))
print(json.dumps(evaluation["metrics"], ensure_ascii=False, indent=2))
print("Val loss:", evaluation["loss"]["val"]["loss"])
print("Test loss:", evaluation["loss"]["test"]["loss"])
print((EVAL_DIR / "evaluation_report.md").read_text(encoding="utf-8")[:6000])
'''
)

md("## 8. Экспорт самодостаточного архива и проверка в новом процессе")

code(
    r'''
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
run(
    [
        sys.executable, "scripts/export_model.py",
        "--checkpoint", str(best),
        "--tokenizer", str(PROJECT / "artifacts" / "tokenizer"),
        "--evaluation-report", str(EVAL_DIR),
        "--output-dir", str(EXPORT_DIR),
        "--name", "lyra-chat-sft",
        "--overwrite",
    ]
)
print("Экспорт:", EXPORT_DIR / "lyra-chat-sft")
print("Архив:", EXPORT_DIR / "lyra-chat-sft.zip")
'''
)

md("## 9. Проверка чата с экспортированными весами")

code(
    r'''
run(
    [
        sys.executable, "scripts/chat.py",
        "--export-dir", str(EXPORT_DIR / "lyra-chat-sft"),
        "--device", "cuda",
        "--message", "Привет! Как дела?",
        "--max-new-tokens", "60",
    ]
)
'''
)

md(
    """
## 10. Сохранение прогресса между сессиями

- Kaggle: нажмите **Save Version** (или включите Persistence). Чекпойнты лежат в
  `/kaggle/working/lyra-chat/runs/...`; в следующей сессии скопируйте их из output
  либо используйте HF Hub ниже.
- Colab: включите `USE_DRIVE = True` в ячейке 1, чтобы `runs/` и `exports/` писались
  в Google Drive.
- Продолжение обучения: снова выполните ячейки 1–2, затем ячейку нужного этапа.
  `--resume auto` подхватит `checkpoints/last.pt`.

Опциональное резервное копирование на Hugging Face Hub (проверьте приватность):
"""
)

code(
    r'''
HF_REPO = None          # например, "your-name/lyra-chat-checkpoints"
UPLOAD_EXPORT = True    # загрузить только итоговый export, не весь датасет
UPLOAD_RUNS = False     # True = загрузить чекпойнты (может быть несколько ГБ)

if HF_REPO:
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token and IS_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass
    api = HfApi(token=token)
    api.create_repo(repo_id=HF_REPO, repo_type="model", private=True, exist_ok=True)
    if UPLOAD_EXPORT:
        api.upload_folder(
            repo_id=HF_REPO,
            folder_path=str(EXPORT_DIR / "lyra-chat-sft"),
            path_in_repo="export",
            repo_type="model",
        )
    if UPLOAD_RUNS:
        api.upload_folder(
            repo_id=HF_REPO,
            folder_path=str(PERSIST / "runs"),
            path_in_repo="runs",
            repo_type="model",
        )
    print("Загружено в", HF_REPO)
else:
    print("HF_REPO не задан — резервное копирование пропущено.")
'''
)

md(
    """
## Google Colab (альтернатива)

1. Загрузите проект в `/content` или смонтируйте Drive, распакуйте архив.
2. Runtime → Change runtime type → T4 GPU.
3. Запустите ячейку 1; при необходимости выставьте `USE_DRIVE = True` **до** запуска,
   чтобы чекпойнты сохранялись на Drive.
4. Далее порядок тот же. Бесплатный Colab может отключать сессию без предупреждения;
   `--resume auto` и чекпойнты каждые 250 шагов минимизируют потери.
5. Если проект лежит на Drive, скопируйте его в `/content` для быстрого DataLoader,
   а `PERSIST` оставьте на Drive.

## Что честно ожидать

Это модель класса GPT-1/GPT-2 по архитектуре, но не по интеллекту. 104M параметров,
~130M pretrain-токенов и ограниченный GPU-бюджет означают, что ответы будут короткими,
иногда бессвязными и с повторами. Оценивайте реальные примеры из
`reports/evaluation/evaluation_report.md`, а не только loss.
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

path = Path("notebooks/train_free_gpu.ipynb")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", path, "cells", len(cells))
