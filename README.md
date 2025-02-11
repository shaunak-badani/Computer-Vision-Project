# **Anemia Detection with RBC Segmentation**

## How to Run

The application is hosted at: http://174.109.75.233

- Create a virtual environment and install dependencies
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   make install
   ```
   
- Run the project / user interface:
   ```
   make run
   ```

## **Problem Statement**
Anemia is a condition characterized by a deficiency of red blood cells (RBCs) or hemoglobin, leading to reduced oxygen transport in the blood. Detecting anemia through automated blood sample image analysis can enhance diagnostic efficiency. This project focuses on segmenting RBCs from blood smear images and classifying them as healthy or anemic.

## **Data Sources**
- **Dataset:** [AneRBC: Anemia Diagnosis Using RBC Images](https://www.kaggle.com/datasets/jocelyndumlao/anerbc-anemia-diagnosis-using-rbc-images?resource=download)
- **Size:** 1000 images (500 normal, 500 anemic)
- **Annotations:** Binary and RGB-segmented ground truths
- **Additional Data:** CBC & morphology reports
- **License:** CC0 (Public Domain)

## **Previous Work & Literature Review**
- The dataset has been used in prior studies involving UNet, LinkNet, and Attention-UNet for RBC segmentation.

- Our approach extends previous work by fine-tuning the **Segment Anything Model (SAM)** and comparing it to both naive and machine learning models.

## **Data Processing Pipeline**
1. **Raw Data Processing:**
   - Images resized while maintaining aspect ratio (max dimension: 1024px).
   - Masks converted to binary (foreground = 1, background = 0).
   - Noise reduction via erosion.
   - Random sampling of 100 points within masked regions for robust learning.
   
2. **Data Splitting:**
   - **600 images** for fine-tuning/training.
   - **100 images** for validation.
   - **300 images** for testing.

3. **Model Training:**
   - **Optimizer:** AdamW
   - **Scheduler:** Cosine annealing
   - **Loss Function:** BCE + Dice loss with gradient accumulation (2 steps).
   - **Validation:** IoU-based, performed every 100 steps.
   - **Checkpoints:** Saved every 500 steps.

## **Modeling Approach**
### **Naïve Baseline:**
- **Using the SAM model without fine-tuning** as a baseline to compare against more optimized approaches.

### **Non-Deep Learning Model:**
- **HOG + LightGBM Classifier**
  - Histogram of Oriented Gradients (HOG) for feature extraction.
  - LightGBM (LGBM) for RBC classification.

### **Deep Learning Models:**
- **Fine-tuned SAM Model** (our proposal).
- **UNet, LinkNet, and Attention-UNet** (previous literature models).

## **Model Evaluation & Metrics**
### **Segmentation Metrics:**
- **IoU (Intersection over Union)**, **Dice Coefficient** , **Precision & Recall**, **Loss & Accuracy**

### **Classification Metrics:**
- **Accuracy, Precision, Recall, F1 Score**

## **Comparison to Naïve Approach**

| Models                     | Loss  | Accuracy | Dice Coefficient | IoU   | Precision | Recall | Specificity |
|----------------------------|-------|----------|------------------|-------|-----------|--------|-------------|
| Fine Tuned SAM2 Current (Our) | 0.0639 | 0.9189   | 0.9361           | 0.8799 | 0.9849    | 0.8919 | 0.9727      |
| Fine Tuned SAM2 Avg (Our)    | 0.0666 | 0.9562   | 0.9634           | 0.9265 | 0.9839    | 0.9562 | 0.9654      |
| Otsu ML Approach (Our)       | 0.0880 | 0.8848   | 0.9120           | 0.8383 | 0.9899    | 0.8455 | 0.9793      |
| UNet (Paper)                 | 0.2503 | 0.9174   | 0.9414           | 0.8916 | 0.9332    | 0.9448 | 0.8982      |
| LinkNet (Paper)              | 0.2018 | 0.9426   | 0.9497           | 0.9083 | 0.9474    | 0.9557 | 0.9232      |
| Atten-UNet (Paper)           | 0.2719 | 0.9780   | 0.9829           | 0.9665 | 0.9833    | 0.9825 | 0.9700      |



## **Demo & Deployment**
The model is deployed via **Streamlit**, providing an interactive interface to upload images, segment RBCs, and classify anemia risk.


## **Ethics Statement**

