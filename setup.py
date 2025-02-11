import numpy as np
from scripts.train_classifier import train_classifier
from scripts.decision_tree import train_decision_tree
from scripts.fine_tuning import fine_tune_sam

def train_pipeline():
    """
    Runs the training pipeline.
    - Runs pipeline and generates features
    - Fine tunes model and predicts final outcome
    """
    print("Commencing training of pipeline!")
    train_classifier()
    train_decision_tree()

    print("Fine tuning SAM2!")
    fine_tune_sam()



if __name__ == "__main__":
    train_pipeline()