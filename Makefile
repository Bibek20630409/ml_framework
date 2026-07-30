.PHONY: install lint format type test cov train serve docker-serve clean

install:
	pip install -e ".[dev,serve,hpo,image,diagnostics]"

format:
	black src tests
	isort src tests

lint:
	ruff check src tests
	black --check src tests

type:
	mypy src

test:
	pytest

cov:
	pytest --cov=ml_framework --cov-report=term-missing --cov-report=xml

train:
	mlf train --config configs/example_tabular.yaml

serve:
	mlf serve --artifacts outputs --host 0.0.0.0 --port 8000

docker-serve:
	docker build --target serve -t ml-framework-serve .
	docker run --rm -p 8000:8000 -v $(PWD)/outputs:/app/outputs:ro ml-framework-serve

# ── MLOps ────────────────────────────────────────────────────────────
stack-up:            ## full local stack: api + mlflow + prometheus + grafana + minio
	docker compose up --build

stack-down:
	docker compose down

dvc-repro:
	dvc repro

k8s-validate:        ## render the kustomize manifests without a cluster
	kubectl kustomize deploy/k8s > /dev/null && echo "kustomize OK"

k8s-deploy:
	kubectl apply -k deploy/k8s

airflow-up:
	docker compose -f orchestration/airflow/docker-compose.airflow.yml up

clean:
	rm -rf outputs wandb lightning_logs .pytest_cache .mypy_cache .ruff_cache \
	       .coverage coverage.xml build dist *.egg-info src/*.egg-info
