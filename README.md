# Image-Classification-using-Artificial-Neural-Networks-ANN-and-Convolutional-Neural-Networks-CNN
 Public Image classification using Artificial Neural Networks (ANNs) and Convolutional Neural Networks (CNNs) in CIFAR-10 dataset to classify images into multiple categories.
## 📌 Project Overview
The objective was to compare the performance of traditional **Artificial Neural Networks (ANN)** against **Convolutional Neural Networks (CNN)** for image classification tasks and analyze model performance, accuracy, and overfitting behavior.

## 📂 Dataset Used- CIFAR-10 Dataset
The project uses the CIFAR-10 image dataset, a built-in dataset available in Python.

Dataset characteristics:
- **50,000 training images**
- **10,000 testing images**
- **32 × 32 color images**
- **3 color channels (RGB)**

### Image Categories
1. ✈️ Airplane  
2. 🚗 Automobile  
3. 🐦 Bird  
4. 🐱 Cat  
5. 🦌 Deer  
6. 🐶 Dog  
7. 🐸 Frog  
8. 🐴 Horse  
9. 🚢 Ship  
10. 🚚 Truck

---

## 🔗 Python Code

Google Colab Notebook:  
[Open Colab Notebook](https://colab.research.google.com/drive/1Zuy7BkliKbXNCU7tr9fy3SE1_GwauOpd?usp=sharing)


## 🔍 Data Exploration

The dataset was explored using:

- `len()` function to understand dataset size
- `np.unique()` to identify unique classes
- Visualization of sample training and testing images

### Data Preprocessing
The image pixel values originally ranged from **0–255** and were normalized to **0–1** for improved model performance.

```python
train_xs = train_xs / 255
test_xs = test_xs / 255
```
##  Artificial Neural Network (ANN)

### ANN Architecture

```python
model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(10, activation='softmax')
])
```

### ANN Findings
- Initial accuracy: **38%**
- Adding hidden layers improved accuracy to **44–46%**
- Overfitting observed after **4–7 epochs**
- Optimal epochs: **~7**

---

## 🧠 Convolutional Neural Network (CNN)

### CNN Architecture

```python
model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(10, activation='softmax')
])
```

### CNN Findings
- Accuracy improved to **68–75%**
- CNN significantly outperformed ANN
- Best filters: **3×3**
- Zero padding improved model accuracy
- Slight overfitting observed after **2–3 epochs**

---

## 📊 Model Comparison

| Model | Accuracy |
|--------|-----------|
| ANN | 38%–46% |
| CNN | 68%–75% |

### Key Observation
CNNs performed substantially better than ANNs due to their ability to capture spatial image features effectively.

---

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Google Colab
- Deep Learning
- CNN & ANN

---

## 📈 Key Learnings
✔️ Learned image preprocessing techniques  
✔️ Compared ANN and CNN performance  
✔️ Understood overfitting and epoch tuning  
✔️ Experimented with hidden layers, filters, and padding  
✔️ Applied deep learning concepts for image classification

---

## 🚀 Future Improvements
- Hyperparameter tuning
- Data augmentation
- Transfer learning
- More CNN layers for improved accuracy
