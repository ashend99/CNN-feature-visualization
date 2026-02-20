# 🧠 Interpreting CNN Representations via Feature Visualization

## 📌 Project Overview

This project explores **how Convolutional Neural Networks (CNNs) learn and represent visual information** using a combination of visualization and interpretability techniques.

Instead of focusing only on classification accuracy, the project inspects:

- Learned convolutional filters  
- Feature map activations  
- Saliency maps  
- Grad-CAM visualizations  
- Embedding space representations (t-SNE / UMAP)

The goal is to gain deeper intuition about CNN behavior and internal feature representations.

---

## 🎯 Motivation

CNNs are highly effective for image classification, but understanding **what they learn internally** can provide valuable insight into:

- Model interpretability  
- Feature extraction  
- Decision-making behavior  
- Representation structure  

This project serves as a hands-on investigation of CNN representations.

---

## 🧪 Objectives

✔ Train a lightweight CNN on CIFAR-10  
✔ Visualize learned convolutional filters  
✔ Inspect intermediate feature maps  
✔ Generate gradient-based saliency maps  
✔ Implement Grad-CAM  
✔ Analyze learned embeddings using t-SNE / UMAP  
✔ Compare Random vs Trained CNN embeddings  

---

## 🏗️ Methodology

### **Dataset**
- **CIFAR-10**
  - 10 object classes
  - 32×32 RGB images

---

### **Model**
Custom lightweight CNN:

- 3 Convolutional layers  
- ReLU activations  
- MaxPooling  
- Fully connected classifier  

Designed for:

✔ CPU-friendly training  
✔ Fast experimentation  
✔ Clear visualization  

---

## 🔬 Experiments

### **1️⃣ CNN Training**
- Optimizer: Adam  
- Loss: CrossEntropyLoss  
- Trained for 10 epochs  

---

### **2️⃣ Convolutional Filter Visualization**
- Extracted filters from first conv layer  
- Observed edge/color-like patterns  

---

### **3️⃣ Feature Map Analysis**
- Visualized activation channels  
- Compared responses across layers  

---

### **4️⃣ Saliency Maps**
- Computed gradients w.r.t. input  
- Highlighted influential pixels  

---

### **5️⃣ Grad-CAM**
- Generated class-specific heatmaps  
- Visualized spatial focus regions  

---

### **6️⃣ Embedding Space Analysis**
Extracted feature embeddings:

- t-SNE projection  
- UMAP projection  

Visualized representation clustering.

---

### **7️⃣ Random vs Trained Embedding Comparison**
Compared embeddings from:

- Randomly initialized CNN  
- Trained CNN  

**Observation:**
- Random CNN → scattered distribution  
- Trained CNN → visible class grouping  

---

## 📊 Key Observations

✔ Early filters resemble edge/texture detectors  
✔ Feature maps capture hierarchical patterns  
✔ Saliency maps highlight pixel sensitivity  
✔ Grad-CAM localizes decision regions  
✔ Training improves embedding structure  

---