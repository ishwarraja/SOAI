# FashionMNIST Classification with CNN (~25K Parameters)

## 📌 Overview
This project demonstrates how to train a **Convolutional Neural Network (CNN)** with **~25K parameters** on the **FashionMNIST dataset**.  
FashionMNIST is a drop-in replacement for MNIST digits, but with 10 classes of clothing items (T-shirt, trouser, dress, sneaker, bag, etc.).

Our goal:  
- Keep the network lightweight (~25K params).  
- Train it on FashionMNIST.  
- Achieve good accuracy while maintaining efficiency.  

---

## 🧠 What are Parameters?
- **Parameters** are the **learnable weights** in a neural network.  
- Example:  
  - A fully connected layer with `in_features=100` and `out_features=10` has `(100 + 1) * 10 = 1010` parameters.  
  - CNN filters also have parameters (kernel weights + bias).  

**Total parameters here: ~25K**  
(small enough to run on CPU easily).

---

## 📊 Dataset: FashionMNIST
- **Images:** 28×28 grayscale  
- **Classes:** 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)  
- **Train set:** 60,000 images  
- **Test set:** 10,000 images  

---

## 🏗 Model Architecture
1. **Conv1** → 6 filters, kernel=5 (156 params)  
2. **Conv2** → 12 filters, kernel=5 (1812 params)  
3. **FC1** → 192 → 100 (19,300 params)  
4. **FC2** → 100 → 40 (4040 params)  
5. **Output** → 40 → 10 (410 params)  

**Total ≈ 25,718 parameters**

---

## ⚙️ Training
- **Optimizer**: SGD with Momentum (lr=0.01, momentum=0.9)  
- **Loss**: CrossEntropyLoss  
- **Epochs**: 20  
- **Batch Size**: 100  

---

## 📈 Results
- **Training accuracy**: ~85%  
- **Test accuracy**: ~86% after 20 epochs  
- **Loss decreases steadily**  

Plots show:
- **Training loss curve** → decreasing  
- **Accuracy curve** → increasing, with test accuracy stabilizing  

---

## ✅ Key Takeaways
- Small CNN (~25K params) can classify FashionMNIST effectively.  
- Accuracy ~85–86% on test set.  
- Larger networks (~400K+ params) can reach **~92–93%**, but at higher cost.  
- Tradeoff: accuracy vs. parameter count vs. compute.

---

## 🚀 How to Run
```bash
# Install dependencies
pip install torch torchvision matplotlib

# Run training in S4_MnistData.ipynb

