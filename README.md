# Hand_Written_Word_Recognition
This project implements a Convolutional Neural Network (CNN) in PyTorch to classify handwritten digits from the MNIST dataset. The model is trained in Google Colab, evaluated for accuracy, and includes visualization, confusion matrix analysis, and optional custom image prediction.
# ✍️ Handwritten Digit Recognition (PyTorch + MNIST)

## 📌 Project Overview

This project builds a **Convolutional Neural Network (CNN)** using **PyTorch** to recognize handwritten digits (0–9) from the **MNIST dataset**.
The model is trained and tested in **Google Colab** and achieves high accuracy in classifying digit images.

---

## 🚀 Features

* CNN-based handwritten digit classifier
* Training and evaluation using PyTorch
* Visualization of predictions and confusion matrix
* Model saving and downloading
* Optional prediction on custom handwritten images

---

## 🧠 Tech Stack

* Python
* PyTorch
* Torchvision
* NumPy & Matplotlib
* Google Colab

---

## 📊 Dataset

**MNIST Dataset**

* 60,000 training images
* 10,000 test images
* 28×28 grayscale handwritten digits

Source: torchvision.datasets.MNIST

---

## ⚙️ Model Architecture

**CNN Structure:**

* Conv Layer → ReLU → MaxPool
* Conv Layer → ReLU → MaxPool
* Fully Connected Layer (128 neurons)
* Dropout Regularization
* Output Layer (10 classes)

---

## 📈 Results

Typical performance:

* Training Accuracy: ~99%
* Test Accuracy: ~98–99%

(Results may vary slightly depending on runtime.)

---

## 🖼️ Example Workflow

1. Load MNIST dataset
2. Preprocess images (normalize)
3. Train CNN model
4. Evaluate accuracy
5. Visualize predictions
6. Save trained model

---

## 💾 How to Run

### Option 1 — Google Colab

1. Open the notebook in Google Colab
2. Enable GPU (optional):
   Runtime → Change runtime type → GPU
3. Run all cells sequentially

### Option 2 — Local Machine

Install dependencies:

```bash
pip install torch torchvision matplotlib numpy pillow
```

Then run notebook or scripts.

---

## 📂 Project Structure

```
handwritten-digit-recognition/
│
├── digit_recognition.ipynb
├── mnist_cnn_state.pt
└── README.md
```

---

## 🔮 Future Improvements

* Deploy as web app (Streamlit/FastAPI)
* Improve preprocessing for custom digits
* Add real-time drawing interface
* Experiment with deeper CNNs

---

## 🎯 Learning Outcomes

* Understanding CNN fundamentals
* Image preprocessing techniques
* Model training and evaluation
* Working with PyTorch in Colab
* ML project documentation

---

## 👤 Author

**Adarsh Pandey**

(Feel free to connect and collaborate.)

---

⭐ If you found this useful, consider starring the repository!
