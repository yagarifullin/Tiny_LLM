# README.md
# TinyLLM

Независимый пайплайн для обучения собственных языковых моделей на Rust + tch/PyTorch + ROCm. От обучения BPE-токенизатора до претрейна трансформера и инференса — полностью самостоятельный стек без зависимости от HuggingFace и сторонних фреймворков.

---

## Содержание

- [Что это](#что-это)
- [Ключевые особенности](#ключевые-особенности)
- [Требования](#требования)
- [Архитектура](#архитектура)
- [Быстрый старт](#быстрый-старт)
- [Обучение токенизатора](#обучение-токенизатора)
- [Обучение модели](#обучение-модели)
- [Инференс](#инференс)
- [Структура проекта](#структура-проекта)
- [Конфигурация](#конфигурация)
- [Метрики и логирование](#метрики-и-логирование)
- [Производительность](#производительность)
- [Известные проблемы и обходы](#известные-проблемы-и-обходы)
- [Планы развития](#планы-развития)

---

## Что это

**TinyLLM** — проект для обучения компактных языковых моделей (10M–250M параметров) с нуля, на собственном железе, с собственным токенизатором и без зависимости от внешних ML-фреймворков.

Проект рассчитан на:
- **Ограниченную VRAM** (16 ГБ и меньше).
- **Большие корпуса** (30+ ГБ), которые не влезают в RAM при классическом подходе.
- **AMD GPU** (ROCm) — но может быть запущен и на CUDA после адаптации.

**Целевой сценарий**: 100M модель на 33.5 ГБ русского текста.

---

## Ключевые особенности

### Токенизатор

- **Собственная реализация BPE** на Rust с алгоритмом `IterativeBPE` для больших корпусов.
- **Шарды с агрегацией `(word, freq)`** — 33.5 ГБ текста превращаются в 1.3 ГБ шардов, что даёт **26× сжатие** и **40× ускорение обучения**.
- **Точные частоты без семплинга** — в отличие от reservoir sampling (toktoktok), подход сохраняет полную статистику.
- **500 МБ RAM** на 33.5 ГБ корпуса (против OOM у HuggingFace `tokenizers`, которому нужно 30+ ГБ RAM).
- **20 минут** обучения vocab=50K на 33.5 ГБ (в 8–12× быстрее SentencePiece и fastBPE).

### Модель

- **Трансформер** с современными оптимизациями:
  - **RMSNorm** вместо LayerNorm.
  - **RoPE** (Rotary Position Embedding).
  - **SwiGLU** FFN.
  - **SDPA** (Scaled Dot-Product Attention, Flash Attention на поддерживаемых GPU).
  - **Weight tying** между embedding и output projection.
- **Mixed precision** (FP16) с FP32 master weights.
- **Gradient clipping** через встроенный `clip_grad_norm` из tch 0.26.
- **LoRA** для fine-tuning.
- **KV-cache** для быстрого инференса.

### Обучение

- **Шардовый pipeline** для больших корпусов.
- **Prefetch chunk iterator** — параллельная загрузка чанков с диска.
- **Cosine LR schedule** с warmup (20% от total).
- **Gradient accumulation**.
- **Early stopping**.
- **Checkpoints** каждые N эпох.
- **Resume** с сохранением состояния модели и metadata.

---

## Требования

| Компонент | Версия | Примечание |
|---|---|---|
| **Rust** | 1.75+ | |
| **PyTorch** | 2.13.0+rocm10.0.0 | Собран против ROCm 10 |
| **ROCm** | 10.0.0 | Для gfx1200/gfx1201 (RDNA4) |
| **Python** | 3.11–3.14 | Для PyTorch |
| **GPU** | AMD RDNA4 (RX 9060 XT, RX 9070) или NVIDIA | 16+ ГБ VRAM рекомендуется |
| **Disk** | NVMe SSD | Обязательно для корпусов >10 ГБ |
| **RAM** | 16+ ГБ | Для 100M модели |

### Установка окружения

```bash
# Создаём venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Устанавливаем PyTorch для ROCm 10
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
    "torch[device-gfx1200]==2.13.0+rocm10.0.0" \
    "torchvision[device-gfx1200]==0.28.0+rocm10.0.0" \
    "torchaudio==2.11.0.2+rocm10.0.0"

# Проверяем
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
# Ожидание: 2.13.0+rocm10.0.0 10.0.xxxxx True
```

### Настройка окружения

Создайте `env.sh`:

```bash
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm/hip
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1

export HIP_VISIBLE_DEVICES=0

# Flash Attention (экспериментальный)
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Для training на RDNA4 (безопаснее)
export TORCH_BLAS_PREFER_HIPBLASLT=0

# Обход известного бага hipBLASLt на RDNA4 + FP16
export PYTORCH_NO_HIP_MEMORY_CACHING=1

# Пути к библиотекам
export TORCH_LIB=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib/host-math/lib:/opt/rocm/hip/lib:$TORCH_LIB
```

Примените: `source env.sh`.

> **Важно**: симлинки на `librocm-openblas.so.0` могут потребоваться для линковки:
> ```bash
> sudo mkdir -p /opt/rocm/lib
> sudo ln -sf /opt/rocm/core-10.0/lib/host-math/lib/librocm-openblas.so.0 /opt/rocm/lib/librocm-openblas.so.0
> sudo ln -sf /opt/rocm/core-10.0/lib/host-math/lib/librocm-openblas.so /opt/rocm/lib/librocm-openblas.so
> ```

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                    ЭТАП 1: Токенизатор (offline, 20 минут)       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  train_tokenizer                                                │
│    ├── читает .txt из папки корпуса (33.5 ГБ)                  │
│    ├── собирает шарды с агрегацией (word, freq) → 1.3 ГБ        │
│    ├── обучает IterativeBPE (vocab 50K)                        │
│    └── пишет ru_50k_100m.json                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │  (артефакт на диске)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ЭТАП 2: Претрейн модели (дни)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  tiny_llm train                                                 │
│    ├── читает корпус                                          │
│    ├── загружает tokenizer_loader                              │
│    ├── кодирует батчи                                          │
│    └── обучает модель                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │  (обученная модель)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ЭТАП 3: Inference                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  infer                                                          │
│    ├── загружает tokenizer_loader + модель                     │
│    └── генерирует ответы                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Вариации моделей и алгоритмов

Проект спроектирован как **расширяемая платформа**. Привязки к одному типу модели или токенизации нет. Ниже — текущее состояние и план по каждому направлению.

### Архитектуры моделей

| Архитектура | Статус | Заметки |
|---|---|---|
| **Decoder-only Transformer** (GPT-style) | ✅ **Реализовано** | Текущая база: RMSNorm, RoPE, SwiGLU, SDPA, causal attention. Используется для претрейна 100M. |
| **Encoder-only** (BERT-style) | 📋 **План** | Требует: bidirectional attention, `[CLS]`/`[SEP]`, MLM-голова, парные сегменты. ~500 строк в `model.rs`. |
| **Encoder-Decoder** (T5-style) | 📋 **План** | Требует: cross-attention между encoder и decoder, span corruption objective, relative position bias (в T5) вместо RoPE. ~1000 строк. |
| **Prefix LM** (UniLM-style) | 📋 **План** | Промежуточный вариант между GPT и BERT. Causal для первой части, bidirectional для второй. |
| **Mixture-of-Experts** (MoE) | 📋 **План** | Роутер + N экспертов. Актуально для моделей 1B+ для снижения compute при том же числе параметров. |

### Текущая архитектура: Decoder-only Transformer

**Основа**:
- **RMSNorm** вместо LayerNorm (быстрее, стабильнее, как в LLaMA/Mistral).
- **RoPE** (Rotary Position Embedding) — базовое значение `theta=10000`.
- **SwiGLU** FFN: `down(silu(gate(x)) * up(x))`.
- **SDPA** (Scaled Dot-Product Attention) — использует Flash Attention, если доступно на GPU.
- **Weight tying** между входным embedding и output projection.
- **Pre-norm**: `x = x + sublayer(norm(x))`.

**Что можно добавить**:
- **GQA** (Grouped Query Attention) — как в LLaMA 2/3, Mistral. Уменьшает KV-cache в N раз (N = число групп).
- **MQA** (Multi-Query Attention) — крайний случай GQA с одной группой.
- **Sliding Window Attention** — как в Mistral. Ограничивает контекст внимания.
- **ALiBi** — альтернатива RoPE с линейным bias.
- **SwiGLU с разными `hidden_dim`** — текущий ffn_size фиксирован, можно варьировать.

### Алгоритмы токенизации

| Алгоритм | Статус | Заметки |
|---|---|---|
| **BPE** (классический) | ✅ **Реализовано** | `IterativeBPE` — основной алгоритм, 20 минут на 33.5 ГБ. |
| **PreciseBPE** | ✅ **Реализовано** | Точный BPE с дельта-обновлением. Требует RAM ≈ 3× корпус. Для малых корпусов. |
| **Unigram** (Kudo 2018) | ⚠️ **Упрощённый** | EM-алгоритм на выборке корпуса. Viterbi-сегментация. Не полноценный. |
| **WordPiece** | ⚠️ **Заготовка** | Пока = `PreciseBPE` без `##`-разделения. TODO: реальный WordPiece для BERT. |
| **SuperBPE** | 📋 **План** | Слияния через пробелы. Даёт −33% токенов. Требует переработки архитектуры шардов. |
| **BoundlessBPE** | 📋 **План** | Аналог SuperBPE, другая стратегия выбора пар. |
| **Byte-level BPE** (GPT-2-style) | ✅ **Де-факто** | Базовые байты 0..255 в vocab + merges. Именно так и работает. |

### Текущее состояние `IterativeBPE`

**Что реализовано**:
- Шарды с агрегацией `(word, freq)` — сжатие 26×.
- Top-k пар (5120 по умолчанию) — баланс точности и RAM.
- Дельта-обновление `pair_counts` после каждого батча.
- Пересчёт по шардам на каждой итерации.
- Аппроксимация внутри батча (`batch_size=256` → ~99.7% точности).

**Что можно добавить**:
- **Multi-phase training** — распределение бюджета словаря по доменам (русский / английский / код).
- **Warm start** — продолжение обучения существующего словаря.
- **Adaptive batch_size** — уменьшение к концу обучения для точности.
- **Better tie-breaking** — детерминированный порядок при равных частотах.

### Форматы данных

**Корпус**:
- **`.txt`** — одна строка = одно предложение / абзац. Поддерживается.
- **`.jsonl`** — каждая строка = JSON с полем `text`. TODO.
- **Парquet / Arrow** — для больших корпусов. TODO.

**Токенизатор**:
- **`.json`** — собственный текстовый формат (hex-кодированные байты + ID).
- **`.shard`** — бинарный формат шарда (SHRD02).
- **`.ltlm`** — планируемый формат состояния тренера (optimizer state).

**Модель**:
- **`.safetensors`** — для весов модели (совместимо с большинством инструментов).
- **`.meta.json`** — метаданные рядом с моделью.

### Как добавить свою архитектуру

**Модель** (например, GQA):
1. Форкнуть `model.rs` → `model_gqa.rs`.
2. Изменить `MultiHeadAttention::forward` — разделить головы на группы.
3. Изменить `TinyLLM::new` — принимать `num_kv_heads`.
4. Добавить выбор архитектуры через CLI-флаг.

**Токенизацию** (например, SuperBPE):
1. Форкнуть `tokenizer-trainer/iterative_bpe.rs`.
2. Добавить фазу superword merges (2-й проход по корпусу).
3. Изменить `tokenizer_loader.rs` — поддержать новые токены.

**Плагинность не реализована**. Форк — единственный путь сейчас.

---

## Быстрый старт

```bash
# 1. Активировать окружение
source .venv/bin/activate
source env.sh

# 2. Собрать проект
cargo build --profile tokenizer --bin train_tokenizer
cargo build --release --bin tiny_llm --bin infer

# 3. Обучить токенизатор (если нет готового)
./target/tokenizer/train_tokenizer \
    --dataset-folder datasets/Pretrain/txt_utf8/ \
    --vocab-size 50000 \
    --min-freq 2 \
    --threads 10 \
    --batch-size 256 \
    --shard-size-mb 1024 \
    --top-k-multiplier 20 \
    --keep-shard-cache \
    --output tokenizer/tokenizer.json

# 4. Обучить модель
./target/release/tiny_llm --create my_model --params 100M \
    --dataset datasets/Pretrain/txt_utf8/ \
    --tokenizer tokenizer/tokenizer.json \
    --epochs 3 --batch-size 8 --block-size 512 \
    --accumulation-steps 2 --lr 3e-4 --weight-decay 0.01 \
    --label-smoothing 0.1 --early-stopping 5 --save-every 1 \
    --sync-half-every 10 --device 0 --threads 12 \
    --mixed-precision true --checkpointing false
```
---

## Обучение токенизатора

### CLI

```bash
./target/tokenizer/train_tokenizer \
    --dataset-folder <путь> \
    --vocab-size <N> \
    --min-freq <N> \
    --threads <N> \
    --batch-size <N> \
    --shard-size-mb <N> \
    --top-k-multiplier <N> \
    --keep-shard-cache \
    --output <путь.json>
```

### Параметры

| Параметр | Обязателен | Описание |
|---|---|---|
| `--dataset-folder` | ✅ | Папка с `.txt` файлами |
| `--vocab-size` | ✅ | Целевой размер словаря (50000 для 100M модели) |
| `--min-freq` | ✅ | Минимальная частота пары (2 — оптимум) |
| `--threads` | ✅ | Число потоков CPU |
| `--batch-size` | ✅ | Merges за итерацию (256 — баланс точности и скорости) |
| `--shard-size-mb` | ✅ | Размер шарда (1024 МБ для 33.5 ГБ) |
| `--top-k-multiplier` | ✅ | Множитель top-k (20 — оптимум) |
| `--keep-shard-cache` | ❌ | Не удалять шарды после обучения |
| `--output` | ❌ | Путь для сохранения (по умолчанию `tokenizer/tokenizer.json`) |

### Что происходит внутри

1. **Сборка шардов** (2–3 минуты на NVMe для 33.5 ГБ):
   - 10 потоков параллельно читают файлы.
   - Каждый поток агрегирует `(word, freq)` в `HashMap`.
   - При достижении лимита — пишет временный шард.
   - Финализация: слияние временных в целевые шарды.

2. **Начальный `pair_counts`** (~3.5 сек).

3. **Итерации BPE** (~5–7 сек каждая, всего 195):
   - Взять top-batch_size пар.
   - Слить, обновить `pair_counts`.
   - Пересчитать по шардам.

4. **Сохранение** словаря.

### Метрики обучения токенизатора

```
[IterativeBPE] batch_size=256, shard_size=1024 МБ, top_k_mult=20, потоков=10
[IterativeBPE] top_k = 256 × 20 = 5120
Параллельная сборка шардов: 55249 файлов, 10 потоков, shard_size=1024 МБ
Шарды: 55249/55249 файлов | 33.5/33.5 ГБ (100.0%) | 469 МБ/с
Собрано 65 временных шардов, финализация...
Готово: 4 целевых шардов
[IterativeBPE] Начальный pair_counts...
Пересчитано: 5120 пар за 3.5 сек
[IterativeBPE] Итерация 1: +256 merges (vocab=516/50000) за 0.0 сек
...
[IterativeBPE] Итерация 195: +76 merges (vocab=50000/50000) за 0.0 сек
[IterativeBPE] Готово: 50000 токенов за 20.3 мин (1220 сек)
```

---

## Обучение модели

### CLI

```bash
./target/release/tiny_llm --create <имя> --params <N> \
    --dataset <путь> --tokenizer <путь.json> \
    --epochs <N> --batch-size <N> --block-size <N> \
    --accumulation-steps <N> --lr <F> --weight-decay <F> \
    --label-smoothing <F> --early-stopping <N> --save-every <N> \
    --sync-half-every <N> --device <N> --threads <N> \
    --mixed-precision <bool> --checkpointing <bool>
```

### Обязательные параметры

| Параметр | Описание |
|---|---|
| `--create <имя>` | Создать новую модель |
| `--params <N>` | Размер: `100K`, `1M`, `5M`, `10M`, `25M`, `50M`, `100M`, `250M` |
| `--dataset <путь>` | Папка с корпусом |
| `--tokenizer <путь>` | Путь к `.json` токенизатора |
| `--epochs <N>` | Общее число эпох |
| `--batch-size <N>` | Размер батча |
| `--block-size <N>` | Размер контекста |
| `--accumulation-steps <N>` | Накопление градиентов |
| `--lr <F>` | Learning rate |
| `--weight-decay <F>` | Weight decay |
| `--label-smoothing <F>` | Label smoothing |
| `--early-stopping <N>` | Терпение early stopping |
| `--save-every <N>` | Сохранять каждые N эпох |
| `--sync-half-every <N>` | Обновлять FP16 веса каждые N шагов |
| `--device <N>` | GPU устройство |
| `--threads <N>` | Потоки CPU |
| `--mixed-precision <bool>` | FP16 вкл/выкл |
| `--checkpointing <bool>` | Gradient checkpointing |

### Resume

```bash
./target/release/tiny_llm --model models/my_model_epoch_0.safetensors \
    --dataset datasets/Pretrain/txt_utf8/ \
    --tokenizer tokenizer/tokenizer.json \
    --epochs 5 --batch-size 8 --block-size 512 \
    ...
```

При resume:
- Metadata загружается из `.meta.json` рядом с моделью.
- `start_epoch` определяется из имени файла (`_epoch_N`).
- Optimizer state **не восстанавливается** (в текущей версии).

### Метрики обучения модели

Прогресс-бар в терминале:
```
Эпоха 0 (24%) | Chunk 4/7 | Батч 10896/25000 | Loss: 6.3405, LR: 0.000150
Time: 0d 00h 50m 47s | ETA: 0d 02h 36m 05s | Общий: 24.5% (0/2 эпох)
[████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 24.5%
```

---

## Инференс

```bash
./target/release/infer \
    --model models/my_model_final.safetensors \
    --tokenizer tokenizer/tokenizer.json \
    --temperature 0.7 \
    --max-tokens 50 \
    --device 0
```

Интерактивные команды:
- `exit` / `quit` — выход.
- `clear` — очистка экрана.

---

## Структура проекта

```
tiny_llm/
├── Cargo.toml                       # зависимости и бинарники
├── config.json                      # архитектуры моделей
├── env.sh                           # переменные окружения
├── README.md                        # этот файл
├── datasets/                        # корпус (вне репозитория)
├── models/                          # чекпоинты и metadata
├── tokenizer/                       # токенизаторы, шарды, логи
└── src/
    ├── main.rs                      # точка входа обучения
    ├── lib.rs                       # (задел на будущее)
    ├── model.rs                     # архитектура TinyLLM
    ├── train.rs                     # обучающий контур
    ├── tokenizer_loader.rs          # загрузчик токенизатора
    ├── data_loader.rs               # загрузка корпуса
    ├── chunked_cache.rs             # чанкированный кэш
    ├── kv_cache.rs                  # KV-cache для инференса
    ├── lora.rs                      # LoRA-адаптеры
    ├── metadata.rs                  # метаданные модели
    ├── rocm.rs                      # preload ROCm библиотек
    └── bin/
        ├── infer.rs                 # inference CLI
        ├── check_gpu.rs             # проверка GPU
        └── tokenizer-trainer/       # обучение токенизатора
            ├── main.rs
            ├── args.rs
            ├── errors.rs
            ├── help.rs
            ├── io_utils.rs
            ├── iterative_bpe.rs
            ├── pair.rs
            ├── pair_counts.rs
            ├── precise_bpe.rs
            ├── shard.rs
            ├── token_store.rs
            ├── tokenizer.rs
            ├── trainer.rs
            ├── trie.rs
            ├── unigram.rs
            └── wordpiece.rs
```

---

## Конфигурация

`config.json` содержит **только архитектуры моделей**:

```json
{
  "model_configs": [
    { "name": "100K", "embed_size": 64,   "num_layers": 2,  "ffn_size": 256,  "num_heads": 4 },
    { "name": "1M",   "embed_size": 128,  "num_layers": 3,  "ffn_size": 512,  "num_heads": 4 },
    { "name": "5M",   "embed_size": 192,  "num_layers": 4,  "ffn_size": 768,  "num_heads": 4 },
    { "name": "10M",  "embed_size": 256,  "num_layers": 4,  "ffn_size": 1024, "num_heads": 4 },
    { "name": "25M",  "embed_size": 384,  "num_layers": 6,  "ffn_size": 1536, "num_heads": 8 },
    { "name": "50M",  "embed_size": 512,  "num_layers": 8,  "ffn_size": 2048, "num_heads": 8 },
    { "name": "100M", "embed_size": 768,  "num_layers": 12, "ffn_size": 3072, "num_heads": 8 },
    { "name": "250M", "embed_size": 1024, "num_layers": 24, "ffn_size": 4096, "num_heads": 8 }
  ]
}
```

**Все параметры обучения** (lr, batch_size, epochs и т.д.) — **через CLI**. **Никаких дефолтов** для критичных значений.

---

## Метрики и логирование

### Файлы логов

| Файл | Содержимое | Как смотреть |
|---|---|---|
| `tokenizer/train_full.log` | Все события (эпохи, чекпоинты, ошибки) | `tail -f` |
| `tokenizer/train_metrics.log` | Таймеры в текстовом формате | `tail -f` |
| `tokenizer/train_metrics.csv` | Таймеры в CSV (для парсинга) | `column -t -s,` |
| `tokenizer/gpu_monitor.csv` | GPU метрики (rocm-smi каждые 5 сек) | `tail -f` |

### Мониторинг

**Терминал 1**: обучение (прогресс-бар).
**Терминал 2**: `tail -f tokenizer/train_full.log`.
**Терминал 3**: `tail -f tokenizer/train_metrics.log`.
**Терминал 4**: `tail -f tokenizer/gpu_monitor.csv`.

### Метрики в CSV

**`train_metrics.csv`**:
```
timestamp,epoch,chunk,batch,loss,lr,create_ms,fwd_ms,loss_ms,dv_ms,bwd_ms,step_ms,total_ms
```

**`gpu_monitor.csv`**:
```
timestamp,gpu_pct,vram_used_mb,vram_total_mb,temp_edge_c,temp_junction_c,temp_memory_c,power_w
```

---

## Производительность

### Токенизатор (33.5 ГБ, vocab 50K, NVMe)

| Метрика | Значение |
|---|---|
| Сборка шардов | **2–3 минуты** |
| Обучение BPE | **17 минут** (195 итераций × 5–7 сек) |
| **Итого** | **20.3 минуты** |
| Пик RAM | **~500 МБ** |

### Модель (5M, Чехов 677K строк)

| Метрика | Значение |
|---|---|
| Обучение 2 эпохи | **~3 ч 45 мин** |
| Скорость | **~57 прим/сек** |

**Примечание**: цифры зависят от ROCm-стека, workaround'ов и текущей версии оптимизаций.

### Известные ограничения

- **`fwd`/`bwd` не масштабируются** от размера модели при малых моделях (kernel launch overhead).
- **`PYTORCH_NO_HIP_MEMORY_CACHING=1`** даёт **×2.4 замедление** (workaround для бага hipBLASLt).

---

## Известные проблемы и обходы

### 1. `PYTORCH_NO_HIP_MEMORY_CACHING=1` — обязателен для RDNA4 + FP16

**Симптом**: `hipErrorIllegalAddress` в `Cijk_Ailk_Bjlk_HHS_BH_...` (hipBLASLt) на батче ~8300.

**Причина**: известный upstream-баг в ROCm 10 + RDNA4 + hipBLASLt.

**Обход**: `export PYTORCH_NO_HIP_MEMORY_CACHING=1` — но **×2.4 замедление**.

**Статус**: ждём фикс в ROCm 10.1.

### 2. NaN в loss

**Симптом**: NaN на эпохе 1+.

**Причина**: `clip_gradients` в старом коде **не работал** (`let _ = grad * scale` не мутирует градиент).

**Обход**: используем `optimizer.clip_grad_norm(1.0)` из tch 0.26 + `has_nan_grad` (раз в 50 шагов).

### 3. Линковка `librocm-openblas`

**Симптом**: `undefined reference to zgemm_` при сборке.

**Обход**: симлинки в `/opt/rocm/lib` (см. секцию «Настройка окружения»).

---

## Планы развития

### Краткосрочные

- ✅ Токенизатор на шардах с агрегацией.
- ✅ Trie для encode.
- ✅ Переход на ROCm 10 + tch 0.26.
- ⏳ Прогресс-бар: терминал — статус, файлы — история.
- ⏳ CSV-логирование метрик.
- ⏳ Оптимизация kernel launch overhead (batch increase, RMSNorm без FP32).

### Среднесрочные

- 🔜 Resume с сохранением optimizer state (формат `.ltlm`).
- 🔜 Multi-phase training для токенизатора (бюджет словаря по доменам).
- 🔜 Warm start для токенизатора.
- 🔜 Упрощённый SuperBPE (2-й проход по корпусу).

### Долгосрочные

- 🔮 Fused kernels через torch-sys.
- 🔮 HIP Graphs для снижения launch overhead.
- 🔮 Полноценный SuperBPE.
- 🔮 Multi-GPU training (DDP).

---

## Лицензия

Проект для внутреннего использования.
Принадлежит Артём Гарифуллин @gunt3er

---

## Благодарности

- [tch-rs](https://github.com/LaurentMazare/tch-rs) — Rust bindings для PyTorch.
- [rayon](https://github.com/rayon-rs/rayon) — параллельные вычисления.
- [serde](https://github.com/serde-rs/serde) — сериализация.
- ROCm — AMD GPU Compute.
