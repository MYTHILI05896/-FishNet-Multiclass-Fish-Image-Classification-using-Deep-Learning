
# 🐟 FishNet — Multiclass Fish Image Classification using Deep Learning with accuracy of 0.75 and loss of 0.8% :

FishNet is a deep learning-based image classification project that identifies **11 species of fish** from photographs. This project is built using Convolutional Neural Networks (CNN) and Transfer Learning, trained on a labeled fish image dataset. A Streamlit app is provided for interactive model inference.

---

## 🧠 Project Highlights

- 📊 **Dataset**: Labeled images of 11 fish categories.
- 🧹 **Preprocessing**: Image resizing, normalization, augmentation.
- 🧠 **Models Used**:
  - Custom CNN from scratch
  - Transfer Learning with VGG16
- 📈 **Evaluation**:
  - Accuracy & Loss curves
  - Classification report & Confusion matrix
- 🚀 **Deployment**: Streamlit-based web app for predictions.

---

## 🗂️ Project Structure
FishNet-Multiclass-Fish-Image-Classification/
│
├── multiclass-fish-classification/
│ ├── app/ # Streamlit app
│ │ └── app.py
│ ├── notebooks/ # All notebooks
│ │ ├── 01_data_preprocessing.ipynb
│ │ ├── 02_custom_cnn_model.ipynb
│ │ ├── 03_transfer_learning.ipynb
│ │ ├── 04_model_evaluation.ipynb
│ │ └── best_model_cnn.h5
│ └── data/ # Image data folders (train/test)
│
├── fishnet-venv/ # Python virtual environment (optional)
├── requirements.txt
└── README.md

---

## 📁 Dataset Overview

- **Total Images**: ~9412
- **Classes (11 total)**:
  - `fish sea_food black_sea_sprat`
  - `fish sea_food shrimp`
  - `fish sea_food sea_bass`
  - `animal fish`
  - `fish sea_food red_mullet`
  - `fish sea_food gilt_head_bream`
  - `fish sea_food trout`
  - `fish sea_food hourse_mackerel`
  - `fish sea_food red_sea_bream`
  - `animal fish bass`
  - `fish sea_food striped_red_mullet`

---

## 🔧 Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MYTHILI05896/-FishNet-Multiclass-Fish-Image-Classification-using-Deep-Learning.git
   cd -FishNet-Multiclass-Fish-Image-Classification-using-Deep-Learning
2.Create and activate virtual environment:
python -m venv fishnet-venv
source fishnet-venv/bin/activate  # (Linux/Mac)
fishnet-venv\\Scripts\\activate   # (Windows)

3.install dependencies
pip install -r requirements.txt

4.run the streamlit app
cd multiclass-fish-classification
streamlit run app/app.py

🧪 Model Performance

Model	Accuracy	Loss
Custom CNN	~63%	1.08
VGG16 (Transfer)	75%	0.80

📷 Sample Output (Streamlit App)
Upload an image of a fish.

The model predicts its class.

Shows the prediction with confidence.

📌 Future Improvements
🔁 Model ensembling for higher accuracy

🧠 Try different pretrained models (ResNet, EfficientNet)

🧼 Use advanced augmentation and test-time augmentation

🌐 Deploy using Docker or HuggingFace Spaces

🙌 Acknowledgements
Dataset: Roboflow

CNN & Transfer Learning inspiration from the deep learning community

Built with 💙 using TensorFlow, Keras, and Streamlit

📬 Contact
Mythili N
📧 Email: m9808262@gmail.com
🔗 GitHub:https://github.com/MYTHILI05896/-FishNet-Multiclass-Fish-Image-Classification-using-Deep-Learning




