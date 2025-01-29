import numpy as np
from scripts.build_features import build_features
from scripts.make_dataset import make_dataset
from scripts.model import train_model

def train_pipeline():
    """
    Runs the training pipeline.
    - Fetches the data
    - Runs pipeline and generates features
    - Trains model and predicts final outcome
    """
    print("Commencing training of pipeline!")

    build_features()
    make_dataset()
    train_model()

if __name__ == "__main__":
    train_pipeline()