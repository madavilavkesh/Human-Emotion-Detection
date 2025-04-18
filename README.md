# Human Emotion Detection 🎭

A deep learning-based real-time emotion detection system using facial expression recognition. This project uses OpenCV and a CNN model trained on grayscale face images to classify emotions like happy, sad, angry, surprise, and more.

---

## 📷 Demo Video

> [Demo - Human Emotion Detection](https://drive.google.com/file/d/1WO2Nqg8Z2X0lz-O4dvPC4-y76XLgI7He/view?usp=sharing)

---

## 📌 Features

- Real-time webcam-based facial emotion recognition
- Trained using a custom CNN model on 48x48 grayscale facial images
- Supports 7 emotion classes: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

---

## 🧠 Model

The model is built using **Keras** with a **CNN architecture**, trained on a facial emotion dataset. After training, the model is saved as:

- `emotiondetector.h5` – Model weights  
- `emotiondetector.json` – Model architecture  

> ⚠️ **Note:** Due to GitHub's file size limit, these model files are not included in the repository. You’ll need to train the model using the notebook or upload the files manually if you already have them.

---

## 📂 Dataset

The model is trained on the **Face Expression Recognition Dataset** from Kaggle:

🔗 [Face Expression Recognition Dataset on Kaggle](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)

Download and extract the dataset, then organize it as follows:

```
images/
├── train/
│   ├── angry/
│   ├── disgust/
│   └── ...
└── test/
    ├── angry/
    ├── disgust/
    └── ...
```

---

## 🗂 Project Structure

```
├── app.py                         # Real-time emotion detection via webcam
├── human_emotion_detection.ipynb  # CNN model training notebook
├── requirements.txt               # Required Python libraries
├── emotiondetector.h5             # (Not uploaded) Trained model weights
├── emotiondetector.json           # (Not uploaded) Trained model architecture
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/human-emotion-detection.git
cd human-emotion-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

- Download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
- Extract it to an `images/` folder as shown above

### 4. Train the Model (Optional)

If you don’t have the model files, run the notebook to train:

```bash
# Launch Jupyter Notebook
jupyter notebook human_emotion_detection.ipynb
```

This will generate:
- `emotiondetector.h5`
- `emotiondetector.json`

### 5. Run Real-Time Emotion Detection

```bash
python app.py
```

> Press `Esc` key to exit the webcam window.

---

## ⚙️ Requirements

- Python 3.7+
- Webcam for real-time emotion detection

---

## 🤖 Emotion Labels

| Label Index | Emotion     |
|-------------|-------------|
| 0           | Angry       |
| 1           | Disgust     |
| 2           | Fear        |
| 3           | Happy       |
| 4           | Neutral     |
| 5           | Sad         |
| 6           | Surprise    |

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---
