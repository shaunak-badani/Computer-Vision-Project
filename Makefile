.PHONY: install run

# Target to install dependencies
install:
	python3 -m venv .venv
	source .venv/bin/activate
	@pip install -r requirements.txt

# Target to run the main.py script
train:
	python3 setup.py

run:
	python3 src.py

# Default target
all: install run