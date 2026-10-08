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
- [Логирование](#логирование)
- [Инференс](#инференс)
- [Структура проекта](#структура-проекта)
- [Конфигурация](#конфигурация)
- [Метрики](#метрики)
- [Производительность](#производительность)
- [Известные проблемы и обходы](#известные-проблемы-и-обходы)
- [Планы развития](#планы-развития)

---

## Что это

**TinyLLM** — проект для обучения компактных языковых моделей (0.13M–403M параметров) с нуля, на собственном железе, с собственным токенизатором и без зависимости от внешних ML-фреймворков.

Проект рассчитан на:
- **Ограниченную VRAM** (16 ГБ и меньше).
- **Большие корпуса** (30+ ГБ), которые не помещаются в RAM при классическом подходе.
- **AMD GPU** (ROCm 10.1) — но потенциально будет работать и на CUDA.

**Целевой сценарий**: 113M модель на 33.5 ГБ русского текста.

---

## Ключевые особенности

### Токенизатор

- **Собственная реализация BPE** на Rust с алгоритмом `IterativeBPE` для больших корпусов.
- **Шарды с агрегацией `(word, freq)`** — 33.5 ГБ текста превращаются в 1.3 ГБ шардов, что даёт **26× сжатие** и **40× ускорение обучения**.
- **Точные частоты без семплинга** — в отличие от reservoir sampling (toktoktok), наш подход сохраняет полную статистику.
- **500 МБ RAM** на 33.5 ГБ корпуса (против OOM у HuggingFace `tokenizers`, которому нужно 30+ ГБ RAM).
- **20 минут** обучения vocab=50K на 33.5 ГБ (в 8–12× быстрее SentencePiece и fastBPE).
- **Trie-encode** — жадный longest-match, 0.19 мкс/текст (в 100× быстрее линейного прохода по merges).

### Модель

- **Трансформер** с современными оптимизациями:
  - **RMSNorm** вместо LayerNorm.
  - **RoPE** (Rotary Position Embedding).
  - **SwiGLU** FFN.
  - **SDPA** (Scaled Dot-Product Attention, Flash Attention на поддерживаемых GPU).
  - **Weight tying** между embedding и output projection.
- **Mixed precision** (FP16) с FP32 master weights.
- **Gradient clipping** через собственную реализацию (не зависит от tch API).
- **LoRA** для fine-tuning.
- **KV-cache** для быстрого инференса.

### Обучение

- **Потоковый pipeline** для больших корпусов:
  - **Потоковая токенизация** без загрузки всего корпуса в RAM.
  - **Resume** токенизации через `metadata.json` после сбоя.
  - **Порог 5 ГБ**: датасеты меньше — в RAM (быстрее), больше — потоково.
- **Шардовый pipeline** для обучения: чтение по чанкам с диска.
- **Prefetch chunk iterator** — параллельная загрузка чанков с диска.
- **Cosine LR schedule** с warmup (2 эпохи).
- **Gradient accumulation**.
- **Early stopping**.
- **Checkpoints** каждые N эпох.
- **Resume** с сохранением состояния модели и metadata.

### Логирование

- **`TerminalManager`** — единственный владелец живой области терминала (2 строки: статус + полоса прогресса).
- **`FileWriteManager` + `LogMode`** — единый владелец лог-файлов.
- **CLI-флаг `--log-mode`** — обязателен, выбирает набор активных файлов.
- **CSV-метрики** — таймеры каждого шага.
- **`[GRAD_NORM]`** — логирование нормы градиента раз в 100 шагов.

---

## Требования

| Компонент | Версия | Примечание |
|---|---|---|
| **Rust** | 1.85+ | Для edition 2024 в зависимостях |
| **PyTorch** | **2.13.0+rocm10.1.0** | Собран против ROCm 10.1 |
| **ROCm** | **10.1** | Для gfx1200/gfx1201 (RDNA4) |
| **Python** | 3.11–3.14 | Для PyTorch |
| **GPU** | AMD RDNA4 (RX 9060 XT, RX 9070) или NVIDIA | 16+ ГБ VRAM рекомендуется |
| **Disk** | NVMe SSD | Обязательно для корпусов >10 ГБ |
| **RAM** | 16+ ГБ | Для 113M модели |

### Установка окружения

```bash
# Создаём venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Устанавливаем PyTorch для ROCm 10.1
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
    "torch[device-gfx1200]==2.13.0+rocm10.1.0" \
    "torchvision[device-gfx1200]==0.28.0+rocm10.1.0" \
    "torchaudio==2.11.0.2+rocm10.1.0"

# Проверяем
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
# Ожидание: 2.13.0+rocm10.1.0 10.1.xxxxx True

Настройка окружения

Создайте env.sh:
bash

export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm/hip
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1

export HIP_VISIBLE_DEVICES=0

# Flash Attention (экспериментальный)
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Пути к библиотекам
export TORCH_LIB=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH=$TORCH_LIB:/opt/rocm/lib:/opt/rocm/hip/lib:$LD_LIBRARY_PATH

Примените: source env.sh.

    Важно: симлинки на librocm-openblas.so.0 могут потребоваться для линковки:
    bash

    sudo mkdir -p /opt/rocm/lib
    sudo ln -sf /opt/rocm/core-10.1/lib/host-math/lib/librocm-openblas.so.0 /opt/rocm/lib/librocm-openblas.so.0
    sudo ln -sf /opt/rocm/core-10.1/lib/host-math/lib/librocm-openblas.so /opt/rocm/lib/librocm-openblas.so

Архитектура
text

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
│    ├── проверяет .chunked_cache/metadata.json                  │
│    ├── если кэша нет — токенизирует (в RAM или потоково)       │
│    ├── читает чанки с диска                                    │
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

Быстрый старт
bash

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
./target/release/tiny_llm --create my_model --params 113M \
    --dataset datasets/Pretrain/txt_utf8/ \
    --tokenizer tokenizer/tokenizer.json \
    --epochs 3 --batch-size 4 --block-size 512 \
    --accumulation-steps 4 --lr 3e-4 --weight-decay 0.01 \
    --label-smoothing 0.1 --early-stopping 5 --save-every 1 \
    --sync-half-every 10 --device 0 --threads 12 \
    --mixed-precision true --checkpointing false \
    --log-mode full \
    --clip-grad-norm 1.0

Обучение токенизатора
CLI
bash

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

Что происходит внутри

    Сборка шардов (2–3 минуты на NVMe для 33.5 ГБ):

        10 потоков параллельно читают файлы.

        Каждый поток агрегирует (word, freq) в HashMap.

        При достижении лимита — пишет временный шард.

        Финализация: слияние временных в целевые шарды.

    Начальный pair_counts (~3.5 сек).

    Итерации BPE (~5–7 сек каждая, всего 195):

        Взять top-batch_size пар.

        Слить, обновить pair_counts.

        Пересчитать по шардам.

    Сохранение словаря.

Обучение модели
CLI
bash

./target/release/tiny_llm --create <имя> --params <N> \
    --dataset <путь> --tokenizer <путь.json> \
    --epochs <N> --batch-size <N> --block-size <N> \
    --accumulation-steps <N> --lr <F> --weight-decay <F> \
    --label-smoothing <F> --early-stopping <N> --save-every <N> \
    --sync-half-every <N> --device <N> --threads <N> \
    --mixed-precision <bool> --checkpointing <bool> \
    --log-mode <режим> \
    --clip-grad-norm <F>

Обязательные параметры
Параметр	Описание
--create <имя>	Создать новую модель
--params <N>	Размер: 0.13M, 0.8M, 2.4M, 4.2M, 14M, 34M, 113M, 403M
--dataset <путь>	Папка с корпусом
--tokenizer <путь>	Путь к .json токенизатора
--epochs <N>	Общее число эпох
--batch-size <N>	Размер батча
--block-size <N>	Размер контекста
--accumulation-steps <N>	Накопление градиентов
--lr <F>	Learning rate
--weight-decay <F>	Weight decay
--label-smoothing <F>	Label smoothing
--early-stopping <N>	Терпение early stopping
--save-every <N>	Сохранять каждые N эпох
--sync-half-every <N>	Обновлять FP16 веса каждые N шагов
--device <N>	GPU устройство
--threads <N>	Потоки токенизации
--mixed-precision <bool>	FP16 вкл/выкл
--checkpointing <bool>	Gradient checkpointing
--log-mode <режим>	Обязателен. Режим логирования
--clip-grad-norm <F>	Максимальная норма градиента (default: 1.0)
Resume
bash

./target/release/tiny_llm --model models/my_model_epoch_0.safetensors \
    --dataset datasets/Pretrain/txt_utf8/ \
    --tokenizer tokenizer/tokenizer.json \
    --epochs 5 --batch-size 4 --block-size 512 \
    ... \
    --log-mode full

При resume:

    Metadata загружается из .meta.json рядом с моделью.

    start_epoch определяется из имени файла (_epoch_N).

    Optimizer state не восстанавливается (в текущей версии).

Логирование
Режимы --log-mode
Режим	Что делает
off	Ничего не пишем в файлы (терминал работает)
debug	Всё, включая покадровые [TIMER]
terminal	Только терминал, файлы не пишем
metrics	Только train_metrics.csv и train_metrics.log
full	train_full.log, train_metrics.csv, train_metrics.log, generated_text.log

Файлы пишутся в models/.
Архитектура логирования

    TerminalManager (src/terminal.rs) — единственный владелец живой области терминала. Рисует 2 строки (статус + полоса прогресса), обрезает по ширине, не даёт им переноситься. Все сообщения идут через log() (обрезается) или log_raw() (без обрезки — для JSON).

    FileWriteManager (src/file_write.rs) — единый владелец лог-файлов. Регистрация с режимом Append/New, методы write / write_raw / flush_all / unregister / count. Внутри — Mutex<HashMap<String, BufWriter<File>>>, передаётся по &self.

    LogMode — управляет набором активных файлов.

Файлы логов
Файл	Содержимое	Как смотреть
models/train_full.log	Все события (эпохи, чекпоинты, ошибки, [GRAD_NORM], [TIMER] в Debug)	tail -f
models/train_metrics.csv	Таймеры каждого 100-го шага в CSV	column -t -s,
models/train_metrics.log	Таймеры текстом (пустой по умолчанию)	tail -f
models/generated_text.log	Тексты, сгенерированные моделью каждые 5 эпох	tail -f
Инференс
bash

./target/release/infer \
    --model models/my_model_final.safetensors \
    --tokenizer tokenizer/tokenizer.json \
    --temperature 0.7 \
    --max-tokens 50 \
    --device 0

Интерактивные команды:

    exit / quit — выход.

    clear — очистка экрана.

Структура проекта
text

tiny_llm/
├── Cargo.toml                       # зависимости и бинарники
├── config.json                      # архитектуры моделей
├── env.sh                           # переменные окружения
├── README.md                        # этот файл
├── datasets/                        # корпус (вне репозитория)
├── models/                          # чекпоинты, metadata, логи
├── tokenizer/                       # токенизаторы, шарды, логи
└── src/
    ├── main.rs                      # точка входа обучения
    ├── lib.rs                       # сборник модулей для библиотеки
    ├── model.rs                     # архитектура TinyLLM
    ├── train.rs                     # обучающий контур
    ├── tokenizer_loader.rs          # загрузчик токенизатора
    ├── data_loader.rs               # загрузка корпуса
    ├── chunked_cache.rs             # чанкированный кэш + потоковая токенизация
    ├── kv_cache.rs                  # KV-cache для инференса
    ├── lora.rs                      # LoRA-адаптеры
    ├── metadata.rs                  # метаданные модели
    ├── terminal.rs                  # менеджер терминала
    ├── file_write.rs                # менеджер лог-файлов
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

Конфигурация

config.json содержит только архитектуры моделей:
json

{
  "model_configs": [
    { "name": "0.13M", "embed_size": 64,   "num_layers": 2,  "ffn_size": 256,  "num_heads": 4 },
    { "name": "0.8M",  "embed_size": 128,  "num_layers": 3,  "ffn_size": 512,  "num_heads": 4 },
    { "name": "2.4M",  "embed_size": 192,  "num_layers": 4,  "ffn_size": 768,  "num_heads": 4 },
    { "name": "4.2M",  "embed_size": 256,  "num_layers": 4,  "ffn_size": 1024, "num_heads": 4 },
    { "name": "14M",   "embed_size": 384,  "num_layers": 6,  "ffn_size": 1536, "num_heads": 8 },
    { "name": "34M",   "embed_size": 512,  "num_layers": 8,  "ffn_size": 2048, "num_heads": 8 },
    { "name": "113M",  "embed_size": 768,  "num_layers": 12, "ffn_size": 3072, "num_heads": 8 },
    { "name": "403M",  "embed_size": 1024, "num_layers": 24, "ffn_size": 4096, "num_heads": 8 }
  ]
}

Имена конфигов отражают число «полезных» параметров модели (attention + FFN + norm), без учёта embedding. Embedding (50000 × embed × 2) не считается — это стандарт в промышленных реализациях (GPT-2, Llama).

Все параметры обучения (lr, batch_size, epochs и т.д.) — через CLI. Никаких дефолтов для критичных значений.
Метрики
Мониторинг

Терминал 1: обучение (прогресс-бар, 2 строки).
Терминал 2: tail -f models/train_full.log.
Терминал 3: tail -f models/train_metrics.log.
Терминал 4: tail -f models/train_metrics.csv.
Метрики в CSV

train_metrics.csv:
text

timestamp,epoch,chunk,batch,create_ms,fwd_ms,loss_ms,dv_ms,bwd_ms,step_ms,total_ms

[GRAD_NORM] — в train_full.log, раз в 100 шагов:
text

[GRAD_NORM] epoch=0 chunk=0 batch=100 norm=12.3456

Производительность
Токенизатор (33.5 ГБ, vocab 50K, NVMe)
Метрика	Значение
Сборка шардов	2–3 минуты
Обучение BPE	17 минут (195 итераций × 5–7 сек)
Итого	20.3 минуты
Пик RAM	~500 МБ
Модель (113M, Чехов 677K строк)
Метрика	Значение
ETA 3 эпохи	~9 часов
Batch	4 (эффективный 16 через accumulation)
Block size	512

Примечание: цифры зависят от ROCm-стека, workaround'ов и текущей версии оптимизаций.
Известные ограничения

    fwd/bwd не масштабируются от размера модели при малых моделях (kernel launch overhead).

    pin_memory deprecated warning — от tch 0.26.

Известные проблемы и обходы
1. pin_memory deprecated warning

Симптом:
text

Warning: The argument 'device' of Tensor.pin_memory() is deprecated.

Причина: tch 0.26 вызывает старый API с аргументом device. PyTorch 2.13 использует новый API без аргумента.

Обход: игнорировать — не влияет на работу.

Статус: косметический.
2. Медленный optimizer.step() на малых моделях

Симптом: fwd/bwd не масштабируются с размером модели.

Причина: kernel launch overhead + sync-и + RMSNorm с FP32-конверсиями + apply_rope через stack.

Обход: batch increase, in-place RoPE, fused RMSNorm (в планах).

Статус: в работе (card-0008, card-0010).
