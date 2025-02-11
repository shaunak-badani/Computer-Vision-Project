from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib


# Paths to images and masks
base_dir = "data/raw/AneRBC-I"
image_dirs = {
    'healthy': os.path.join(base_dir, 'Healthy_individuals/Original_images'),
    'anemic': os.path.join(base_dir, 'Anemic_individuals/Original_images')
}
mask_dirs = {
    'healthy': os.path.join(base_dir, 'Healthy_individuals/Binary_Segmented'),
    'anemic': os.path.join(base_dir, 'Anemic_individuals/Binary_Segmented')
}
features_file = os.path.join(base_dir, 'segmentation_features.csv')
labels_file = os.path.join(base_dir, 'labels.csv')

# Load image file names and labels
labels = pd.read_csv(labels_file)['label']
data = pd.read_csv(features_file)


# Convert to DataFrame

def train_decision_tree():
    feature_names = ["rbc_count", "area", "contrast", "correlation", "mean_intensity", "mean_red", "std_red", "red_green_ratio"]

    df = pd.DataFrame(data, columns=feature_names)
    df.to_csv(os.path.join(base_dir, 'segmentation_features_fancy.csv'), index=False)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = DecisionTreeClassifier(max_depth=5, random_state=42)  # Adjust max_depth for clarity
    clf.fit(X_train, y_train)


    # Visualize the tree
    plt.figure(figsize=(15, 8))
    plot_tree(clf, feature_names=df.columns, class_names=['Healthy', 'Anemic'], filled=True)
    plt.show()

    # Print decision rules
    tree_rules = export_text(clf, feature_names=df.columns)
    print(tree_rules)

    # Feature importance plot
    plt.figure(figsize=(10, 5))
    plt.barh(df.columns, clf.feature_importances_)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance in Decision Tree")
    plt.show()

    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Precision Score:", precision_score(y_test, y_pred))
    print("Recall Score:", recall_score(y_test, y_pred))


    classifier = {
        'model': clf,
        'scaler': scaler,
    }
    joblib.dump(classifier, 'models/decision_tree_model.joblib')


if __name__ == "__main__":
    train_decision_tree()