.DEFAULT_GOAL := help

.PHONY: help train-voice db-up db-down db-reset migrate makemigrations runserver createsuperuser

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

prepare-data: ## Unzip data for models
	unzip -o models/facial/data.zip -d models/facial/
	unzip -o models/voice/data.zip -d models/voice/
	rm models/facial/data.zip
	rm models/voice/data.zip

train-facial-cnn: ## Train facial CNN model
	cd models/facial && uv run python pipeline.py -c cnn --no-augmentation

train-voice-nn: ## Train voice NN model
	cd models/voice && uv run python pipeline.py -c nn

train-facial-nn: ## Train facial NN model
	cd models/facial && uv run python pipeline.py -c nn

train-voice-cnn: ## Train voice CNN model
	cd models/voice && uv run python pipeline.py -c cnn

tensorboard: ## Run TensorBoard
	uv run tensorboard --logdir models/facial/logs/fit

db-up: ## Start PostgreSQL database with Docker Compose
	docker compose up -d

db-down: ## Stop PostgreSQL database
	docker compose down

db-reset: ## Reset database (stop, remove volumes, start)
	docker compose down -v
	docker compose up -d

format: ## Format and check code with Ruff
	uv run ruff format . && uv run ruff check . --fix

makemigrations: ## Create Django database migrations
	cd src && uv run python manage.py makemigrations

migrate: ## Run Django database migrations
	cd src && uv run python manage.py migrate

run: ## Run Django development server (WSGI)
	cd src && uv run python manage.py runserver

runws: ## Run Django with WebSocket support (Daphne ASGI)
	cd src && uv run daphne -b 0.0.0.0 -p 8000 auth_platform.asgi:application

shell: ## Open Django shell
	cd src && uv run python manage.py shell

test: ## Run tests
	cd src && uv run python manage.py test

clean: ## Clean Python cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf .ruff_cache

createsuperuser: ## Create Django superuser
	cd src && uv run python manage.py createsuperuser
