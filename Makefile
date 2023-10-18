start-dev:
	docker compose up

stop-dev:
	sudo docker compose down

start-prod:
	gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 127.0.0.1:8000

check_typings:
	mypy ./src ./main.py ./settings.py ./config.py

check_linting:
	isort . && ruff check --fix .