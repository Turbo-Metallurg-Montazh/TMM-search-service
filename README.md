# EMK Search Service

FastAPI-сервис для поиска соответствий между позициями тендерной документации и каталогами поставщиков. Разворачивается рядом с основным `EMK CRM Backend` в namespace `backend` и публикуется как отдельный Kubernetes Service.

## Технологии

- Python 3.12
- FastAPI / Uvicorn
- PyTorch + Transformers
- pandas / openpyxl / xlrd
- Prometheus metrics
- Docker / Docker Compose
- Kubernetes manifests (`k8s/`)
- GitHub Actions deploy в GHCR + k3s

## Структура проекта

- `app/api.py` - FastAPI endpoints, health, metrics, Excel import/export
- `app/model.py` - загрузка bi-encoder модели и кодирование текста
- `app/indexer.py` - построение и загрузка поискового индекса
- `app/catalogs.py` - загрузка и чтение Excel-каталогов поставщиков
- `biencoder_model/` - рабочая модель для inference
- `index_data/` - стартовый поисковый индекс
- `price_lists/` - стартовые каталоги поставщиков
- `dapt/`, `dapt_model/`, `samples*.xlsx`, `scripts/` - материалы и скрипты обучения
- `k8s/` - манифесты для деплоя в k3s

## Переменные окружения

- `MODEL_DIR` - путь к модели, по умолчанию `/app/biencoder_model` в Docker
- `INDEX_DIR` - путь к индексу, по умолчанию `/data/index_data` в Docker/Kubernetes
- `PRICE_LIST_DIR` - путь к каталогам, по умолчанию `/data/price_lists` в Docker/Kubernetes
- `MAX_LEN` - максимальная длина токенизации, по умолчанию `128`
- `DEFAULT_BATCH_SIZE` - batch size для построения индекса, по умолчанию `16`
- `API_PREFIX` - внешний path prefix, по умолчанию пустой
- `ALLOWED_ORIGINS` - разрешенные browser origins, в k3s выставлен `https://search.turbo-metallurg-montazh.ru`
- `ALLOWED_HOSTS` - разрешенные Host header для прямых запросов и ingress

## Быстрый старт локально

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

Проверка:

- Health: `http://localhost:8000/health`
- Metrics: `http://localhost:8000/metrics`
- Index status: `http://localhost:8000/index-status`

## Локально через Docker Compose

```bash
docker compose -f docker-compose.local.yml up --build
```

Compose использует volumes для `/data/index_data` и `/data/price_lists`. При первом старте они заполняются стартовыми данными из образа.

## API

Swagger/OpenAPI-описание всех endpoint'ов лежит в `docs/swagger.yaml`.

Поиск:

- `POST /suggest` - вернуть варианты совпадений для строки тендерной документации
- `POST /build-index` - пересобрать индекс по текущим каталогам
- `GET /index-status` - состояние индекса и количество каталогов

Каталоги:

- `GET /catalogs` - список загруженных Excel-каталогов
- `POST /catalogs/upload` - загрузить один `.xls/.xlsx`
- `POST /catalogs/upload-many` - загрузить несколько `.xls/.xlsx`

Excel:

- `POST /import-xlsx` - импорт Excel в формат фронтенд-таблицы
- `POST /export-xlsx` - экспорт фронтенд-таблицы в Excel

Если каталог загружен без `rebuild_index=true`, файл сохраняется, а `/index-status` возвращает `stale: true`. Новый каталог начнет участвовать в поиске после `/build-index`.

Пример загрузки с немедленной пересборкой:

```bash
curl -F "file=@catalog.xlsx" "http://localhost:8000/catalogs/upload?rebuild_index=true"
```

## Kubernetes

Манифесты находятся в `k8s/`:

- `k8s/namespace.yaml`
- `k8s/search-service/pvc.yaml`
- `k8s/search-service/deployment.yaml`
- `k8s/search-service/service.yaml`
- `k8s/search-service/ingress.yaml`

Сервис разворачивается в namespace `backend` рядом с основным backend:

- Deployment: `emk-search-service`
- Service: `emk-search-service`
- Internal URL: `http://emk-search-service`
- External URL: `https://search.turbo-metallurg-montazh.ru/`

PVC `emk-search-service-data` хранит каталоги и индекс, чтобы загрузки через API переживали рестарт Pod. Стратегия Deployment - `Recreate`, потому что индекс и каталоги лежат на `ReadWriteOnce` volume.

Внешний доступ ограничен host/origin проверками:

- Ingress принимает host `search.turbo-metallurg-montazh.ru`
- CORS разрешает только `https://search.turbo-metallurg-montazh.ru`
- Middleware отклоняет чужие `Host`, `Origin` и `Referer`

## CI/CD

На push в `master` workflow `.github/workflows/deploy.yml`:

1. Checkout с Git LFS.
2. Логинится в GHCR.
3. Собирает и публикует образ:
   - `ghcr.io/turbo-metallurg-montazh/tmm-search-service:${GITHUB_SHA}`
   - `ghcr.io/turbo-metallurg-montazh/tmm-search-service:latest`
4. Применяет `k8s/` через `K3S_KUBECONFIG`.
5. Обновляет image в Deployment `emk-search-service`.

Для работы deploy нужны те же секреты, что в основном backend:

- `K3S_KUBECONFIG`
- `GITHUB_TOKEN` используется автоматически для публикации в GHCR

## Обучение

Скрипты рассчитаны на запуск из корня проекта:

```bash
python scripts/dapt_pretrain.py
python scripts/train_biencoder.py
python scripts/predict_scores.py
```

Результаты обучения сохраняются в `dapt_model/` и `biencoder_model/`. После обновления модели нужно пересобрать Docker image и пересобрать индекс.
