PY := python
CONFIG := configs/config.yaml
IMAGE_NAME := telco-churn-api

init:
	$(PY) -m pip install -U pip -r requirements.txt

get-data:
	$(PY) -m src.get_data --config $(CONFIG)

preprocess: get-data
	$(PY) -m src.preprocess --config $(CONFIG)

train: preprocess
	$(PY) -m src.train --config $(CONFIG)

evaluate: train
	$(PY) -m src.evaluate --config $(CONFIG)

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests app

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

run_api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000

run_app:
	streamlit run app/streamlit_app.py --server.port 8501

build_docker:
	docker build -t $(IMAGE_NAME):latest .

run_docker:
	docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=$(or $(MLFLOW_TRACKING_URI),http://host.docker.internal:5000) $(IMAGE_NAME):latest

compose_up:
	docker compose up -d --build

compose_train:
	docker compose run --rm api $(PY) -m src.get_data --config $(CONFIG)
	docker compose run --rm api $(PY) -m src.preprocess --config $(CONFIG)
	docker compose run --rm api $(PY) -m src.train --config $(CONFIG)

compose_down:
	docker compose down

clean:
	rm -rf data/raw.csv data/train.csv data/test.csv reports mlruns mlflow.db

all: init get-data preprocess train evaluate test

.PHONY: init get-data preprocess train evaluate test lint mlflow-ui run_api run_app build_docker run_docker compose_up compose_train compose_down clean all
