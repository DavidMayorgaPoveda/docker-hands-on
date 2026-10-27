.PHONY: test test-backend test-mongo

test: test-backend test-mongo

test-backend:
	docker run --rm -v "$(CURDIR):$(CURDIR)" -w "$(CURDIR)/backend" python:3.10-slim sh -c 'pip install --no-cache-dir -r requirements.txt pytest-cov && python -m pytest -q -p no:cacheprovider -o junit_family=xunit1 --junitxml=junit.xml --cov=app --cov-report=xml:coverage.xml'

test-mongo:
	docker run --rm -v "$(CURDIR)/mongo:/app" -w /app python:3.10-slim sh -c 'pip install --no-cache-dir -r requirements.txt && python -m pytest -q -p no:cacheprovider -o junit_family=xunit1 --junitxml=junit.xml'
