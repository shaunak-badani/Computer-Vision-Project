.PHONY: install run

# Target to install dependencies
# Make sure you create a virtual environment before
install:
	@pip install -r requirements.txt

# Target to run the fine-tuning.py script
train:
	python3 setup.py

# Target to run the main.py script
run:
	python3 main.py

# Default target
all: install run