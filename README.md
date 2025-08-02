# GAN-Image-



---

# Creating Realistic Images with GANs

## 📌 Overview

This project demonstrates the use of **Generative Adversarial Networks (GANs)** to generate realistic images from noise using deep learning. A GAN is composed of two neural networks, the **Generator** and the **Discriminator**, which are trained simultaneously in a game-theoretic setup. The Generator tries to create images that are as realistic as possible, while the Discriminator attempts to distinguish between real images (from the dataset) and fake ones (produced by the Generator).

This repository contains:

* GAN architecture implementation from scratch using PyTorch
* Training on the MNIST dataset
* Visualization of outputs generated across training epochs
* Techniques for stabilizing GAN training

---

## 📂 Folder Structure

```
GAN-Image-Generation/
│
├── GAN_Image_Generator.ipynb     # Jupyter notebook for training the GAN
├── outputs/                      # Folder storing generated images after every 10 epochs
├── models/                       # (Optional) To save Generator/Discriminator model weights
├── data/                         # Automatically downloads MNIST dataset
├── requirements.txt              # List of required packages
├── GAN_Report_Slides.pptx        # Short project presentation
└── README.md                     # Project documentation
```

---

## 🧠 Dataset

The GAN is trained on the **MNIST** dataset, which consists of 70,000 grayscale images of handwritten digits (0–9). Each image is of size 28x28 pixels. The dataset is loaded and preprocessed using `torchvision.datasets`.

---

## 🏗️ Model Architecture

### Generator

* Input: Random noise vector (`z`) of size 100
* Hidden Layers: Fully connected layers with LeakyReLU activations
* Output: A 784-dimensional vector reshaped into a 28x28 image
* Output Activation: `Tanh`

### Discriminator

* Input: Flattened 28x28 image
* Hidden Layers: Fully connected layers with Dropout and LeakyReLU
* Output: Binary classification (`real` or `fake`) using `Sigmoid`

---

## ⚙️ Training Details

* **Loss Function**: Binary Cross Entropy Loss (BCELoss)
* **Optimizer**: Adam (learning rate = 0.0002, β1 = 0.5, β2 = 0.999)
* **Batch Size**: 128
* **Epochs**: 50
* **Stabilization Techniques**:

  * Label smoothing
  * Normalization of input data
  * Use of LeakyReLU in Discriminator
  * Dropout for regularization

During training:

* Generator learns to produce better fakes
* Discriminator becomes better at identifying fakes
* Images are saved every 10 epochs to visualize progress

---

## 📊 Results

* The GAN successfully learns to generate images that closely resemble handwritten digits.
* Initial images appear noisy, but as epochs progress, the generated samples become sharper and more realistic.
* Final outputs after 50 epochs demonstrate clear digit-like shapes and style variation.

Generated outputs are saved in the `outputs/` folder as PNG images.

---

## 📈 Visualizations

* Generated images are visualized and saved during training at fixed intervals.
* Training losses for Generator and Discriminator can also be plotted (not included by default, optional).

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install torch torchvision matplotlib
```

---

## 🚀 How to Run

### In Jupyter Notebook:

1. Open `GAN_Image_Generator.ipynb`
2. Run cells sequentially
3. After training, check `outputs/` for generated images

### In Google Colab:

* Upload the `.ipynb` file to Colab
* Run all cells (ensure runtime type is GPU-enabled)

---

## 📌 Future Work

* Train on more complex datasets like CIFAR-10 or CelebA
* Use Convolutional GANs (DCGAN) for better image quality
* Implement advanced techniques like WGAN, CycleGAN, or StyleGAN
* Add training loss visualization and learning rate scheduling

---

## ✍️ Author

Arya Baisakhiya
B.E – Saveetha Engineering College
Project: Creating Realistic Images with GANs


---

