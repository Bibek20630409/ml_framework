.PHONY: install install-mlops lint format type test cov train select serve \
        benchmark docker-serve stack-up stack-down dvc-repro k8s-validate \
        k8s-deploy airflow-up clean

# `dev` carries torch, the tree libraries and transformers, but NOT polars, shap,
# onnx or prophet — so without the four extras appended here, `make test` silently
# skips test_polars_backend.py, the SHAP tier of test_explain.py, the ONNX export
# tests and the statistical forecasters. A target named `install` that cannot run
# the suite is the wrong kind of quiet.
install:
	pip install -e ".[dev,serve,hpo,image,diagnostics,parquet,fast,explain,export,timeseries]"

# Separate because it is heavy (pyspark + dvc, ~400 MB) and, without a JVM on
# PATH, the Spark tests it unlocks still skip. See the `spark-contract` CI job.
install-mlops:
	pip install -e ".[dev,serve,mlops]"

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

select:              ## compare model families on the shipped sample dataset
	mlf select --config configs/example_selection.yaml

serve:
	mlf serve --artifacts outputs --host 0.0.0.0 --port 8000

benchmark:           ## pandas vs polars: parse + full build_bundle (minutes, not a test)
	python benchmarks/data_backends.py

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
	kubectl kustomize deploy > /dev/null && echo "kustomize OK"

k8s-deploy:
	kubectl apply -k deploy

airflow-up:
	docker compose -f orchestration/airflow/docker-compose.airflow.yml up

clean:
	rm -rf outputs wandb lightning_logs .pytest_cache .mypy_cache .ruff_cache \
	       .coverage coverage.xml build dist *.egg-info src/*.egg-info
