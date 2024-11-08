.PHONY: test format lint check

test:
	poetry run pytest

coverage:
	poetry run pytest --cov=src --cov-report=term-missing

format:
	poetry run black src tests
	poetry run isort src tests

lint:
	poetry run mypy src
	poetry run black --check src tests
	poetry run isort --check-only src tests

check: format lint test