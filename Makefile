.PHONY: install compile lint test gate artifacts dashboard

install:
	python -m pip install -e ".[dev,dashboard]"

compile:
	python -m compileall -q src tests

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest

gate:
	python -m llm_eval_obs gate --tolerance 0

artifacts:
	python scripts/refresh_artifacts.py

dashboard:
	streamlit run src/llm_eval_obs/dashboard/app.py
