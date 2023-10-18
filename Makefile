start-dev:
	docker compose up

stop-dev:
	docker compose down

start-prod:
	gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 127.0.0.1:80

check_typings:
	mypy ./app ./main.py

check_linting:
	isort . && ruff check --fix .