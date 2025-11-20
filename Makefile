.DEFAULT_GOAL := help

.PHONY: help train-voice db-up db-down db-reset migrate makemigrations runserver createsuperuser

help:
	@echo "Available targets:"
	@echo ""
	@echo "  make train-voice      Train voice recognition model using SVM and SpeechBrain"
	@echo "  make train-facial     Train facial recognition model using SVM and Face Recognition"
	@echo "  make db-up            Start PostgreSQL database with Docker Compose"
	@echo "  make db-down          Stop PostgreSQL database"
	@echo "  make db-reset         Reset database (stop, remove volumes, start)"
	@echo "  make migrate          Run Django database migrations"
	@echo "  make makemigrations   Create Django database migrations"
	@echo "  make runserver        Run Django development server"
	@echo "  make createsuperuser  Create Django superuser"
	@echo "  make help            Show this help message"
	@echo ""

prepare-data:
	unzip -o models/facial/data.zip -d models/facial/
	unzip -o models/voice/data.zip -d models/voice/
	rm models/facial/data.zip
	rm models/voice/data.zip

train-voice-svm:
	cd models/voice && uv run python pipeline.py

train-facial-svm:
	cd models/facial && uv run python pipeline.py

train-facial-cnn:
	cd models/facial && uv run python pipeline.py -c cnn

train-voice-nn:
	cd models/voice && uv run python pipeline.py -c nn

train-facial-nn:
	cd models/facial && uv run python pipeline.py -c nn

train-voice-cnn:
	cd models/voice && uv run python pipeline.py -c cnn

db-up:
	docker compose up -d

db-down:
	docker compose down

db-reset:
	docker compose down -v
	docker compose up -d

makemigrations:
	cd src && uv run python manage.py makemigrations

migrate:
	cd src && uv run python manage.py migrate

runserver:
	cd src && uv run python manage.py runserver

createsuperuser:
	cd src && uv run python manage.py createsuperuser

