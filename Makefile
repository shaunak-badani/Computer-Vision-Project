.PHONY: install run

# Target to install dependencies
# Make sure you create a virtual environment before
install:
	@pip install -r requirements.txt

# Target to run the fine-tuning.py script
train:
	python setup.py
	python scripts/extract_features_from_masks.py

# Target to run the main.py script
run:
	streamlit run main.py

# Default target
all: install run