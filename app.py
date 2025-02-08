import streamlit as st
import cv2
import numpy as np
from inference import run_inference, calculate_metrics, process_img, run_sam2_out_of_the_box_with_prompt, show_otsus_thresholding
import pandas as pd

st.set_page_config(layout="wide")
st.title("AneRBC Image Segmentation")

st.sidebar.title("About This Project")
st.sidebar.write("This tool segments red blood cells (RBCs) from microscopic images using fine-tuned SAM2 and other approaches.")
st.sidebar.write("**Dataset:** AneRBC - A benchmark for anemia diagnosis using RBC images. [https://www.kaggle.com/datasets/jocelyndumlao/anerbc-anemia-diagnosis-using-rbc-images/data]")
st.sidebar.write("**Paper:** Benchmark dataset for computer-aided anemia diagnosis using RBC images. [https://doi.org/10.1093/database/baae120]")
st.sidebar.write("**Models:**\n- Fine-tuned SAM2\n- Out-of-the-box SAM2\n- Naïve approach \n- ML-based approach ")

cola, colb = st.columns(2)

with cola:
    image = st.file_uploader("Upload Image [Required]")

with colb:
    mask = st.file_uploader("Upload Mask [Optional]")

if image and st.button("Segment"):
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with st.spinner("Running Inference on Fine-Tuned SAM2 Model..."):
        segmented_img, seg_map = run_inference(img)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original Image")
        st.image(img, caption="Input Image")

    with col2:
        if mask:
            st.subheader("Ground Truth")
            mask_img = cv2.imdecode(np.asarray(bytearray(mask.read())), cv2.IMREAD_GRAYSCALE)
            # convert mask_img to binary
            mask_img = np.where(mask_img > 128, 255, 0).astype(np.uint8)
            st.image(mask_img, caption="Ground Truth Mask")
            pixel_acc, iou_score, dice_score, precision_score, recall_score, specificity_score, loss = calculate_metrics(process_img(mask_img), seg_map)
        
        else:
            st.subheader("Ground Truth")
            st.write("No mask uploaded")

    with col3:
        st.subheader("Predicted Mask")
        st.image(seg_map, caption="Segmentation Map (Fine-Tuned SAM2)")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.subheader("SAM2 - Naive Approach")
        with st.spinner("Running Inference on Out-of-the-box SAM2 Model..."):
            mask_sam2 = run_sam2_out_of_the_box_with_prompt(img)

        st.image(mask_sam2, caption="Segmentation Map (SAM2)")

    with col5:
        st.subheader("Otsu Thresholding")
        hog_features, otsu_image = show_otsus_thresholding(img)
        st.image(otsu_image)
        pixel_acc_otsu, iou_score_otsu, dice_score_otsu, precision_score_otsu, recall_score_otsu, specificity_score_otsu, loss_otsu = calculate_metrics(process_img(otsu_image), seg_map)
        
        
    with col6:
        st.subheader("HOG Features")
        st.image(hog_features)


    if mask:
        metrics_data = {
            "Models": ["Fine Tuned SAM2 Current (Our)", "Fine Tuned SAM2 Avg (Our)", "Otsu ML Approach (Our)", "UNet (Paper)", "LinkNet (Paper)", "Atten-UNet (Paper)"],
            "Loss": [loss, 0.0666, loss_otsu, 0.2503, 0.2018, 0.2719],
            "Accuracy": [pixel_acc, 0.9562, pixel_acc_otsu, 0.9174, 0.9426, 0.9780],
            "Dice Coefficient": [dice_score, 0.9634, dice_score_otsu, 0.9414, 0.9497, 0.9829],
            "IoU": [iou_score, 0.9265, iou_score_otsu, 0.8916, 0.9083, 0.9665],
            "Precision": [precision_score, 0.9839, precision_score_otsu, 0.9332, 0.9474, 0.9833],
            "Recall": [recall_score, 0.9562, recall_score_otsu, 0.9448, 0.9557, 0.9825],
            "Specificity": [specificity_score, 0.9654, specificity_score_otsu, 0.8982, 0.9232, 0.9700]
        }

        df_metrics = pd.DataFrame(metrics_data)

        st.table(df_metrics)
        