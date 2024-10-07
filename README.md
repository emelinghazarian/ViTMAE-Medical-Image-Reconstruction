# ViTMAE-Medical-Image-Reconstruction

## Overview
This project focuses on familiarizing with Masked Autoencoders (ViTMAE) and their application to the field of medical images. The repository implements the following key steps:

1. **Downloading Medical Images**: The ROCOv2 dataset is used, consisting of both training and testing images for medical image analysis. 
2. **ViTMAE Model Evaluation**: Utilizes a pre-trained Vision Transformer Masked Autoencoder (ViTMAE) from the Hugging Face Transformers library to reconstruct masked images (with 75% masking) from the ROCOv2 dataset.
3. **Fine-Tuning**: Fine-tunes the ViTMAE model using the training collection of medical images for a minimum of 5 epochs to optimize performance for medical imaging tasks.
4. **Performance Evaluation**: Evaluates the reconstruction capabilities of both the pre-trained and fine-tuned ViTMAE models on test images with 75% masking.

## Dataset
The **ROCOv2** dataset is used for training and testing purposes. The dataset consists of medical images in two parts: Train and Test. 
- **ROCOv2 Dataset**: [https://example.com/roco-dataset](https://zenodo.org/records/10821435)

## Pre-Trained Model
This project uses a pre-trained Vision Transformer Masked Autoencoder (**ViTMAE**) from the Hugging Face Transformers library. You can download and load the pre-trained model from the library using:

```python
from transformers import ViTMAEForImageProcessing
model = ViTMAEForImageProcessing.from_pretrained('facebook/vit-mae-base')
```

## Steps to Run the Project

1. **Download Dataset**
   - Run the dataset download script to get both Train and Test parts from the ROCOv2 dataset.

2. **Run Evaluation on Pre-Trained ViTMAE**
   - Use the pre-trained ViTMAE model to reconstruct 5 images from the Test dataset with a 75% mask.

3. **Fine-Tune the Model**
   - Fine-tune the ViTMAE model on the Train dataset for at least 5 epochs using:
   ```bash
   python train_vitmae.py
   ```

4. **Reconstruction Performance Evaluation**
   - Use the fine-tuned ViTMAE model to reconstruct 5 examples from the Test dataset, with 75% masking applied.

## Results for part 2:
![image](https://github.com/user-attachments/assets/c1fa066a-9200-4f41-a4a7-1a388d03d386)
![image](https://github.com/user-attachments/assets/f5dd4d7b-23d8-49db-a974-c5276ad21cb3)

## Results for part 4:
![image](https://github.com/user-attachments/assets/f59c573d-89f6-43b1-b418-85a32b3f44b4)


