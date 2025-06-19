
# 🖌️ Flask Drawing App with Model Predictions

This Flask application allows users to draw digits on a canvas in their browser and receive real-time predictions from a machine learning model.

---

## ✨ Features

- 🎨 Interactive drawing canvas in the browser
- ⚡ Instant predictions from a trained ML model (KNN-based)
- 🧠 Model preprocessing with `scaler.pkl`
- 🖥️ Intuitive and lightweight web interface

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.7+
- pip
- Flask
- Joblib / Pickle (for loading models)

### 📦 Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/digitRecogniser.git
   cd digitRecogniser
````

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

### ▶️ Running the App Locally

Start the Flask server:

```bash
python func.py
```

Visit [http://localhost:5000](http://localhost:5000) in your browser to use the app.

---

### 🐳 Running with Docker

If you prefer using Docker:

```bash
docker build -t digit-recogniser .
docker run -p 5000:5000 digit-recogniser
```

Then go to [http://localhost:5000](http://localhost:5000)

---

## 💻 Usage

1. Open your browser and go to `http://localhost:5000`.
2. Draw a digit on the canvas.
3. Click **Predict** to see the result from the model.
4. Click **Clear** to reset the canvas.

---

## 📁 Project Structure

```
digitRecogniser/
├── func.py                  # Flask backend with model inference
├── knn_model.pkl            # Trained KNN model
├── scaler.pkl               # Scaler used for preprocessing
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker setup
└── templates/
    └── index.html           # Web interface (canvas + buttons)
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

**Created by [Lakshya Kesarwani](https://github.com/Lakshya-kesarwani)**

```

---

Let me know if you also want:
- **Deployment instructions** for platforms like Railway or Render.
- A **badge-based header** (Python version, Flask version, etc.).
- README with **screenshots or demo GIF** support.
