![cover](https://github.com/harishmuh/AI-biomedical-research-and-disease-prediction/blob/main/Data/Multi-label%20Thoracic%20Diseases/Banner%20cover.png?raw=true)

# Explainability Analysis of Multi-label Thoracic Disease Predictions from Chest X-ray Images Using Pretrained DenseNet21 Model and Grad-CAM

### **1. Introduction**
#### **Background**

Chest radiography (chest X-ray) is one of the most widely used diagnostic imaging modalities for evaluating thoracic diseases because it is inexpensive, widely accessible, and capable of providing rapid assessment of the lungs, heart, and pleural cavity. It is routinely employed to detect a broad range of abnormalities, including pneumonia, cardiomegaly, pleural effusion, pneumothorax, pulmonary nodules, and other thoracic conditions (WHO, 2024).

Thoracic diseases continue to represent a major global health challenge. Lower respiratory infections remain among the leading causes of mortality worldwide, while lung cancer is the leading cause of cancer-related death, accounting for approximately 1.8 million deaths annually (WHO, 2024; GBD 2021 Diseases and Injuries Collaborators, 2024). Accurate interpretation of chest radiographs is therefore essential for timely diagnosis and appropriate clinical management. However, radiographic interpretation can be challenging because many thoracic diseases exhibit subtle imaging features and multiple abnormalities may coexist within a single image.

Recent advances in deep learning have demonstrated remarkable performance in automated chest X-ray analysis. Large-scale datasets such as the NIH ChestX-ray14 dataset have enabled the development of convolutional neural networks capable of predicting multiple thoracic diseases simultaneously (Wang et al., 2017). Nevertheless, despite their high predictive performance, these models are often criticized as black-box systems because the reasoning behind their predictions is not directly observable.

Explainable Artificial Intelligence (XAI) has emerged as an important research area for improving the transparency and trustworthiness of deep learning models. Among the available XAI techniques, Gradient-weighted Class Activation Mapping (Grad-CAM) is widely used to visualize image regions that contribute most strongly to a model's predictions, thereby providing valuable insights into model behavior without modifying the underlying network architecture (Selvaraju et al., 2020).

Accordingly, this notebook focuses on the explainability of a pretrained DenseNet121-based multi-label thoracic disease classification model rather than developing a new classification model. By utilizing Grad-CAM visualizations, the study investigates how the pretrained model identifies disease-related image regions for both single-disease and multi-label predictions.

#### **Objectives**
Specifically, this notebook aims to:

* Explore the characteristics of a curated subset of the NIH ChestX-ray14 dataset used for explainability analysis.
* Utilize a pretrained DenseNet121-based multi-label classification model to generate probability predictions for fourteen thoracic diseases.
* Visualize model attention using Grad-CAM for single-disease predictions across all fourteen disease categories.
* Investigate Grad-CAM visualizations for multi-label predictions by analyzing chest X-ray images with the highest cumulative disease probabilities.

#### **Scope of the study**

This study utilizes a curated subset of 1,000 chest X-ray images from the NIH ChestX-ray14 dataset to demonstrate a reproducible explainability workflow while maintaining computational feasibility. The selected subset is sufficient to investigate Grad-CAM visualizations using a pretrained DenseNet121-based model, consistent with the study's emphasis on model interpretability rather than model development.

### **2. Dataset**

**Table 1. Summary of the Dataset Used in This Study**

| Attribute | Description |
|-----------|-------------|
| **Dataset Name** | NIH ChestX-ray14 |
| **Original Dataset** | 112,120 frontal chest X-ray images from 30,805 unique patients |
| **Dataset Source** | NIH Clinical Center |
| **Original Disease Labels** | 14 thoracic disease categories (multi-label classification) |
| **Image Modality** | Frontal chest radiographs (PA/AP views) |
| **Image Format** | PNG |
| **Subset Used in This Study** | 1,000 chest X-ray images |
| **Purpose of the Subset** | Inference and explainability analysis using a pretrained DenseNet121 model |
| **Primary Explainability Method** | Gradient-weighted Class Activation Mapping (Grad-CAM) |
| **Dataset License** | Publicly available for research purposes |

**Dataset Resources**
*  Original dataset paper: [ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks](https://arxiv.org/abs/1705.02315)
* Official NIH clinical center, US research hospital, ChestX-ray14 dataset: https://nihcc.app.box.com/v/ChestXray-NIHCC
* Curated 1,000-image subset used in this notebook: https://drive.google.com/file/d/1U23OnS30DkPxR4rPTXf2Msul7AJMc7A5/view


**Thoracic Disease Labels**

The fourteen thoracic disease labels included in the dataset are summarized in Table 2.

| Pathological Conditions | Description (According to Goodman (2020)) |
|-------------------------|------------|
| **Atelectasis** | Partial or complete collapse of lung tissue caused by loss of air within the alveoli, resulting in reduced lung volume. |
| **Cardiomegaly** | Enlargement of the cardiac silhouette on a chest radiograph, usually indicating an enlarged heart. |
| **Consolidation** | Replacement of normal air in the alveoli with fluid, pus, blood, or cells, producing increased lung opacity on chest radiographs. |
| **Edema** | Accumulation of excess fluid within the lung interstitium and alveoli, most commonly caused by heart failure. |
| **Emphysema** | Chronic destruction of alveolar walls leading to enlarged air spaces, lung hyperinflation, and reduced gas exchange. |
| **Effusion** | Accumulation of excess fluid within the pleural space between the lung and the chest wall. |
| **Fibrosis** | Formation of scar tissue within the lung parenchyma, causing permanent lung stiffness and volume loss. |
| **Hernia** | Protrusion of abdominal contents into the thoracic cavity, most commonly through the esophageal hiatus (hiatal hernia). |
| **Infiltration** | A nonspecific radiographic finding indicating abnormal material, such as inflammatory cells or fluid, within the lung parenchyma. |
| **Mass** | A focal pulmonary lesion larger than 3 cm in diameter that may represent benign or malignant disease. |
| **Nodule** | A small, rounded pulmonary opacity measuring up to 3 cm in diameter. |
| **Pleural Thickening** | Thickening of the pleural membrane caused by previous inflammation, infection, trauma, or chronic pleural disease. |
| **Pneumonia** | Infection of the lung parenchyma that commonly appears as focal or diffuse air-space opacity due to inflammatory exudate. |
| **Pneumothorax** | Presence of air within the pleural space resulting in partial or complete collapse of the affected lung. |

### **3. Methodology**

![Workflow](https://github.com/harishmuh/AI-biomedical-research-and-disease-prediction/blob/main/Data/Multi-label%20Thoracic%20Diseases/Overall%20workflow.png?raw=true)


**Pretrained DenseNet121-based Multi-label Classification Model**

Rather than developing a new deep learning model, this study employs a pretrained DenseNet121-based multi-label classification model provided through the AI for Medicine Specialization. Using a validated pretrained model allows the investigation to focus on model explainability while avoiding the substantial computational resources required to train a deep convolutional neural network on the complete NIH ChestX-ray14 dataset.

The implementation consists of two pretrained model files that serve different purposes. The `densenet.hdf5` file contains the pretrained DenseNet121 feature extraction backbone, while the `pretrained_model.h5` file contains the complete multi-label classification model, including the feature extraction network and the disease classification layers. The pretrained model predicts the probability of fourteen thoracic diseases independently using sigmoid activation, enabling multiple disease labels to be assigned to a single chest radiograph.

**Pre-trained model files**
* `densenet.hdf5`: https://drive.google.com/file/d/1ARZ9mToVX4vuD7n27RdtY8vhqi2Fw6N1/view
* `pretrained_model.h5`:
https://drive.google.com/file/d/1sw4kxkSZWQa29mlob39lDdCuCBhAvSro/view
