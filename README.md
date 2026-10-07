# 🌸 Iris Flower Classification using Artificial Neural Network (ANN)
### Deep Learning Mini Project

A complete, production-ready Deep Learning mini project that classifies Iris flowers into three species (*Iris setosa*, *Iris versicolor*, and *Iris virginica*) using a **Multi-Layer Artificial Neural Network (ANN)** built with pure vectorized NumPy, Matplotlib, Seaborn, and Pandas. Includes a live interactive Web Dashboard and full academic documentation.

---

## 🚀 Key Highlights
- **100.00% Test Accuracy** & **1.0000 Macro F1-Score** on unseen test data.
- **ANN Architecture**: $4 \rightarrow 16 \rightarrow 8 \rightarrow 3$ with He Normal Initialization, ReLU hidden activations, Softmax output, and Adam Optimization.
- **Publication-Quality Visualizations**: Automatic generation of Training Loss/Accuracy curves, Confusion Matrix heatmap, 2D Decision Boundaries, and Feature Distributions.
- **Interactive Web Studio**: Standalone zero-dependency Web Application featuring an animated SVG Neural Network diagram with live signal propagation and real-time inference sliders.
- **Image Upload Preview**: Upload a flower image for a browser-side visual estimate against the bundled species references; use the four measurements for the trained ANN classification.
- **Academic Project Report**: Complete `PROJECT_REPORT.md` including theoretical equations, backpropagation derivations, performance analysis, and 20 Viva Voce questions & answers.

---

## 📂 Project Directory Structure

```
DL Mini project/
│
├── data/
│   └── iris.csv                      # Complete Fisher's Iris dataset (150 samples)
│
├── src/
│   ├── __init__.py                   # Package initializer
│   ├── ann_model.py                  # Core ANN Engine (Dense layers, ReLU, Softmax, Adam)
│   ├── data_loader.py                # Preprocessing, Train-Test Split (80/20), StandardScaler
│   ├── evaluator.py                  # Confusion Matrix, Precision, Recall, F1-Score calculation
│   └── visualizer.py                 # Generates high-res training and evaluation charts
│
├── web/
│   ├── index.html                    # Glassmorphism Web App with interactive SVG Neural Network
│   ├── style.css                     # Modern dark theme styles & animations
│   └── app.js                        # Client-side reactivity, slider listeners & SVG animator
│
├── saved_models/
│   └── iris_ann_model.json           # Trained weights, biases, and normalization parameters
│
├── results/                          # Generated evaluation figures
│   ├── training_curves.png           # Loss and Accuracy curves vs Epochs
│   ├── confusion_matrix.png          # Test set confusion matrix heatmap
│   ├── decision_boundary.png         # Non-linear 2D decision boundary visualization
│   └── feature_distributions.png     # Feature boxplots across species
│
├── train.py                          # Training script (model fitting, metric printing, plot export)
├── predict.py                        # CLI tool for interactive & batch predictions
├── app.py                            # Web server & REST API (serves UI on http://127.0.0.1:5000)
├── requirements.txt                  # Python dependencies
├── PROJECT_REPORT.md                 # Full academic project report with Viva Voce Q&A
└── README.md                         # Quick-start documentation
```

---

## 📦 Required Libraries & Installation

The required libraries are specified in `requirements.txt`:
- `numpy` (Tensor calculations, matrix multiplication, neural network math)
- `pandas` (Dataset loading & dataframe manipulation)
- `matplotlib` (Plotting loss/accuracy curves and decision boundaries)
- `seaborn` (Statistical visualization and heatmaps)

### Installation Command:
Open PowerShell or your command prompt in this directory and execute:

```powershell
pip install -r requirements.txt
```

---

## 🏃 Execution Guide

### 1. Train the Artificial Neural Network
To train the ANN, compute metrics on the test partition, and automatically generate all evaluation plots:

```powershell
python train.py
```

**Console Output:**
```
======================================================================
    IRIS FLOWER CLASSIFICATION USING ARTIFICIAL NEURAL NETWORK (ANN)   
======================================================================
Epoch [  1/220] - Train Loss: 0.7654 - Train Acc: 72.50% | Val Loss: 0.8153 - Val Acc: 70.00%
Epoch [ 50/220] - Train Loss: 0.0450 - Train Acc: 98.33% | Val Loss: 0.0931 - Val Acc: 93.33%
Epoch [100/220] - Train Loss: 0.0316 - Train Acc: 99.17% | Val Loss: 0.0586 - Val Acc: 96.67%
Epoch [150/220] - Train Loss: 0.0215 - Train Acc: 99.17% | Val Loss: 0.0550 - Val Acc: 96.67%
Epoch [220/220] - Train Loss: 0.0139 - Train Acc: 99.17% | Val Loss: 0.0439 - Val Acc: 100.00%

Final Test Accuracy: 100.00%
All visual charts saved to results/ directory!
```

---

### 2. Run CLI Flower Prediction
To test single flower measurements using the trained weights:

**Direct Arguments:**
```powershell
# python predict.py <SepalLength> <SepalWidth> <PetalLength> <PetalWidth>
python predict.py 5.1 3.5 1.4 0.2
```

**Interactive Mode:**
```powershell
python predict.py
```
This runs validation against canonical presets and prompts you to input custom flower measurements.

---

### 3. Launch the Interactive Web Studio (Modern Web UI/UX)
To start the modern graphical web dashboard with live neural network graph animation, botanical artwork, and dual-control sliders:

```powershell
python app.py
```

Then open your browser (Google Chrome, Microsoft Edge, etc.) and visit:  
👉 **`http://127.0.0.1:5000`**

**Web Studio UI/UX Features:**
- **Visual Species Quick Selector**: Clickable cards for *Setosa*, *Versicolor*, and *Virginica* that load canonical features and show high-resolution botanical paintings.
- **Dual-Control Inputs**: Drag sliders or type exact centimeter measurements; inputs stay continuously synchronized.
- **Dynamic Neural Graph**: Visualizes the $4 \rightarrow 16 \rightarrow 8 \rightarrow 3$ layers with animated pulses towards the predicted species.
- **Expandable Anatomy Guide**: Visual infographic illustrating exact petal and sepal measurement positions.
- **Real-Time Classification Card**: Displays the matching botanical illustration, confidence ring meter, and softmax probabilities.
- **Analytics & Math Tabs**: View generated charts, confusion matrix, loss formulas, and viva questions directly in the browser!
- **Flower Image Upload**: Drag in a JPG, PNG, or WEBP to preview it and receive a clearly labeled visual reference estimate. This does not replace the ANN, which was trained on sepal and petal measurements rather than image pixels.

---

### 4. Launch the Native Desktop GUI (Zero-Browser Desktop App)
If you prefer an offline native desktop window app (perfect for college lab presentation):

```powershell
python gui.py
```

**Desktop GUI Features:**
- Runs directly as a native Windows desktop application.
- Real-time sliders with instant numerical readouts.
- Displays high-resolution botanical artwork that dynamically changes based on the predicted class.
- Live probability distribution meters and confidence scores.

---

## 📊 Summary of Evaluation Results

| Metric | Score Achieved |
| :--- | :---: |
| **Training Accuracy** | 99.17% |
| **Validation / Test Accuracy** | **100.00%** |
| **Iris setosa F1-Score** | 1.0000 |
| **Iris versicolor F1-Score** | 1.0000 |
| **Iris virginica F1-Score** | 1.0000 |
| **Macro Average F1-Score** | **1.0000** |

---

## 🎓 Academic Submission & Viva Voce
Refer to [`PROJECT_REPORT.md`](PROJECT_REPORT.md) for the complete theoretical documentation, mathematical formulas, and a curated list of **20 Viva Voce questions & answers** to ace your project presentation!
