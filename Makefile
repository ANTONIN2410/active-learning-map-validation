.PHONY: install test benchmark notebooks

install:
	pip install -e ".[dev]"

test:
	pytest -q

benchmark:
	python scripts/run_benchmark.py --replicates 1000 --workers 8

notebooks:
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
