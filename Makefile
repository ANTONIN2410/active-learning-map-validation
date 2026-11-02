.PHONY: install test benchmark notebooks

install:
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:
	pytest -q

benchmark:
	python scripts/run_benchmark.py --replicates 1000 --workers 10

notebooks:
	cd notebooks && jupyter nbconvert --to notebook --execute --inplace *.ipynb
