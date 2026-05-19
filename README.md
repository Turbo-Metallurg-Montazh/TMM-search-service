# EMK Search Service

FastAPI-сервис для поиска соответствий между позициями тендерной документации и каталогами поставщиков.

## Что перенесено

- Рабочие эндпоинты из черновика: `/health`, `/metrics`, `/index-status`, `/build-index`, `/suggest`, `/import-xlsx`, `/export-xlsx`.
- Модель и текущий индекс: `biencoder_model/`, `index_data/`.
- Каталоги поставщиков: `price_lists/`.
- Материалы для обучения: `dapt/`, `dapt_model/`, `samples.xlsx`, `samplescleaned.xlsx`, `scripts/`.

## Запуск

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Каталоги

Загрузить один каталог:

```bash
curl -F "file=@catalog.xlsx" "http://localhost:8000/catalogs/upload?rebuild_index=true"
```

Загрузить несколько каталогов:

```bash
curl -F "files=@catalog1.xlsx" -F "files=@catalog2.xls" "http://localhost:8000/catalogs/upload-many?rebuild_index=true"
```

Если `rebuild_index=false`, файл сохраняется в `price_lists/`, а `/index-status` вернет `stale: true`; новый каталог начнет участвовать в поиске после `/build-index`.

## Обучение

Скрипты перенесены в `scripts/` и рассчитаны на запуск из корня проекта:

```bash
python scripts/dapt_pretrain.py
python scripts/train_biencoder.py
python scripts/predict_scores.py
```

