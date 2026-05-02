.PHONY: test cov

test:
	pytest tests/ -v

cov:
	pytest tests/ --cov=agents --cov=core --cov=api --cov-report=term-missing --cov-report=html -v
