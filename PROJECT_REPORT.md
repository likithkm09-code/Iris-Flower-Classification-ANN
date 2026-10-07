# Academic Mini Project Report
## Iris Flower Classification using Artificial Neural Network (ANN)

**Domain:** Deep Learning & Pattern Recognition  
**Model Architecture:** Multi-Layer Perceptron (Input: 4 → Hidden: 16 → Hidden: 8 → Output: 3)  
**Dataset:** Fisher's Iris Dataset (150 Samples, 4 Morphological Attributes, 3 Classes)  
**Performance Achieved:** **100.00% Test Accuracy, 1.0000 Macro F1-Score**  

---

## 1. Abstract
Classification of biological flora species based on morphological measurements is a foundational benchmark in machine learning and pattern recognition. While *Iris setosa* is linearly separable from other classes, distinguishing *Iris versicolor* from *Iris virginica* presents overlapping multivariate decision boundaries. In this project, we design, implement, and rigorously evaluate an **Artificial Neural Network (ANN)** from first mathematical principles using vectorized matrix operations in NumPy, supplemented with statistical evaluation tools (Matplotlib, Seaborn, Pandas). The network incorporates He (Kaiming) weight initialization, non-linear Rectified Linear Unit (ReLU) hidden activations, a numerically stabilized Softmax output layer, and is trained via the Adam optimizer under Categorical Cross-Entropy loss with $L_2$ weight regularization. On an unseen test set (stratified 20%), the network achieved a **100.00% test accuracy** and **1.0000 macro F1-score**, successfully demonstrating deep representation learning. An interactive web-based visualization dashboard was also developed for real-time model introspection and inference.

---

## 2. Problem Statement & Dataset Analysis
The Iris dataset consists of 150 observations across three species of the Iris genus:
1. **Iris setosa** (50 instances)
2. **Iris versicolor** (50 instances)
3. **Iris virginica** (50 instances)

Each instance comprises four continuous physical measurements in centimeters (cm):
1. **Sepal Length ($X_1$)**: Range: $4.3 - 7.9$ cm ($\mu = 5.84, \sigma = 0.83$)
2. **Sepal Width ($X_2$)**: Range: $2.0 - 4.4$ cm ($\mu = 3.05, \sigma = 0.43$)
3. **Petal Length ($X_3$)**: Range: $1.0 - 6.9$ cm ($\mu = 3.76, \sigma = 1.76$)
4. **Petal Width ($X_4$)**: Range: $0.1 - 2.5$ cm ($\mu = 1.20, \sigma = 0.76$)

### Preprocessing & Feature Standardization
Because features occupy different orders of magnitude and variances, feeding raw features directly into gradient-based optimizers leads to ill-conditioned loss surfaces and oscillating trajectories. We apply **Z-score Standardization**:
$$z_j = \frac{x_j - \mu_j}{\sigma_j}$$
Where $\mu_j$ and $\sigma_j$ denote the mean and standard deviation of feature $j$ computed strictly on the training partition ($N=120$) to prevent data leakage. The target classes are mapped to orthogonal standard one-hot basis vectors:
- $\text{Setosa} \rightarrow [1, 0, 0]^T$
- $\text{Versicolor} \rightarrow [0, 1, 0]^T$
- $\text{Virginica} \rightarrow [0, 0, 1]^T$

---

## 3. Mathematical Foundations of the ANN Architecture

```
[Input Layer]      [Hidden Layer 1]      [Hidden Layer 2]      [Output Layer]
  4 Neurons          16 Neurons             8 Neurons             3 Neurons
(Sepal/Petal)        (He + ReLU)           (He + ReLU)        (Xavier + Softmax)
    X₁ ────┬────────────► (a₁⁽¹⁾) ──────────────► (a₁⁽²⁾) ─────────────► P(Setosa)
    X₂ ────┼────────────► (a₂⁽¹⁾) ──────────────► (a₂⁽²⁾) ─────────────► P(Versicolor)
    X₃ ────┼────────────► ...                   ...                   ► P(Virginica)
    X₄ ────┴────────────► (a₁₆⁽¹⁾) ─────────────► (a₈⁽²⁾)
```

### 3.1 Forward Propagation
Let $A^{[0]} = X \in \mathbb{R}^{B \times 4}$ denote a mini-batch of inputs ($B=16$). For each layer $l \in \{1, 2, 3\}$:
$$Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$
Where:
- $W^{[1]} \in \mathbb{R}^{4 \times 16}, b^{[1]} \in \mathbb{R}^{1 \times 16}$
- $W^{[2]} \in \mathbb{R}^{16 \times 8}, b^{[2]} \in \mathbb{R}^{1 \times 8}$
- $W^{[3]} \in \mathbb{R}^{8 \times 3}, b^{[3]} \in \mathbb{R}^{1 \times 3}$

### 3.2 Activation Functions
1. **Hidden Layers (ReLU):**
   $$g^{[1]}(z) = g^{[2]}(z) = \max(0, z)$$
   $$\frac{\partial g}{\partial z} = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$
   ReLU eliminates the vanishing gradient problem inherent in sigmoid and tanh activations.

2. **Output Layer (Numerically Stabilized Softmax):**
   $$\sigma(Z_i) = \frac{\exp(Z_i - \max(Z))}{\sum_{j=1}^3 \exp(Z_j - \max(Z))}$$
   Subtracting $\max(Z)$ per row guarantees numerical stability by bounding exponents to $\le 0$, eliminating floating-point `NaN`/overflow errors.

### 3.3 Loss Function
Multi-class Categorical Cross-Entropy Loss with $L_2$ weight regularization:
$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^B \sum_{k=1}^3 y_{i,k} \log(\hat{y}_{i,k}) + \frac{\lambda}{2} \sum_{l=1}^3 \|W^{[l]}\|_F^2$$
Where $\lambda = 0.001$ is the weight decay penalty parameter and $\hat{y}_{i,k} = A^{[3]}_{i,k}$.

### 3.4 Backward Propagation (Chain Rule Derivation)
For the Softmax layer combined with Cross-Entropy Loss, the derivative with respect to logits $Z^{[3]}$ simplifies elegantly to:
$$\delta^{[3]} = \frac{\partial \mathcal{L}}{\partial Z^{[3]}} = \frac{1}{B} (\hat{Y} - Y)$$
Gradients for weights and biases of layer 3:
$$\frac{\partial \mathcal{L}}{\partial W^{[3]}} = (A^{[2]})^T \delta^{[3]} + \lambda W^{[3]}$$
$$\frac{\partial \mathcal{L}}{\partial b^{[3]}} = \sum_{i=1}^B \delta^{[3]}_{i,:}$$
Propagating backward to hidden layers:
$$\delta^{[l]} = (\delta^{[l+1]} (W^{[l+1]})^T) \odot g'^{[l]}(Z^{[l]})$$
$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = (A^{[l-1]})^T \delta^{[l]} + \lambda W^{[l]}$$
$$\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \sum_{i=1}^B \delta^{[l]}_{i,:}$$

### 3.5 Optimization: Adam (Adaptive Moment Estimation)
Parameters $\theta \in \{W^{[l]}, b^{[l]}\}$ are updated using running exponential averages of first and second moments:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
Hyperparameters utilized: $\alpha = 0.015, \beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$.

---

## 4. Experimental Results and Analysis

### 4.1 Training Dynamics
- **Dataset Partition:** 80% Training ($N=120$), 20% Unseen Testing ($N=30$) via stratified sampling.
- **Batch Size:** 16
- **Epochs:** 220
- **Convergence:** Training loss dropped from $0.7654$ (Epoch 1) to $0.0139$ (Epoch 220).
- **Validation Loss:** Decreased stably to $0.0439$ with zero signs of over-fitting.

### 4.2 Test Set Confusion Matrix

| Actual \ Predicted | Iris setosa | Iris versicolor | Iris virginica | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Iris setosa** | **10** | 0 | 0 | 10 |
| **Iris versicolor** | 0 | **10** | 0 | 10 |
| **Iris virginica** | 0 | 0 | **10** | 10 |

### 4.3 Classification Performance Metrics

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Iris setosa** | 1.0000 | 1.0000 | 1.0000 | 10 |
| **Iris versicolor** | 1.0000 | 1.0000 | 1.0000 | 10 |
| **Iris virginica** | 1.0000 | 1.0000 | 1.0000 | 10 |
| **Macro Average** | **1.0000** | **1.0000** | **1.0000** | 30 |
| **Weighted Average** | **1.0000** | **1.0000** | **1.0000** | 30 |
| **Overall Accuracy** | **100.00%** | | | 30 |

---

## 5. Visual Artifacts Generated
1. **`results/training_curves.png`**: Dual-axis plot of loss convergence and validation accuracy across epochs.
2. **`results/confusion_matrix.png`**: High-resolution heatmap with cell counts and percentage normalizations.
3. **`results/decision_boundary.png`**: 2D projection over Petal Length vs Petal Width displaying non-linear decision regions learned by the hidden layers.
4. **`results/feature_distributions.png`**: 4-panel distribution analysis showcasing variance between species.

---

## 6. Software Architecture & Deliverables
- **`data/iris.csv`**: Offline dataset.
- **`src/ann_model.py`**: Fully modular, vectorized neural network implementation from scratch.
- **`src/data_loader.py`**: Preprocessing, train-test splitting, standardization, and one-hot encoding.
- **`src/evaluator.py`**: Multi-class metric computing module.
- **`src/visualizer.py`**: Publication-ready chart plotting pipeline.
- **`train.py`**: Command-line script to reproduce training and visualizations.
- **`predict.py`**: Command-line interface for batch and interactive single-flower inference.
- **`app.py` & `web/`**: Interactive graphical studio with live SVG neural network diagram and real-time inference.

---

## 7. Viva Voce Examination Questions & Answers

#### Q1: What is an Artificial Neural Network (ANN), and what are its basic components?
**Answer:** An ANN is a computational model inspired by biological nervous systems. It comprises interconnected nodes (neurons) organized into:
1. **Input Layer:** Receives standardized raw features.
2. **Hidden Layer(s):** Performs linear combinations ($Z = XW + b$) followed by non-linear activations to project data into higher-order feature spaces.
3. **Output Layer:** Produces predictions (class probabilities via Softmax).

#### Q2: Why is Softmax preferred over Sigmoid for multi-class classification?
**Answer:** The Sigmoid function treats each output independently ($\sum \neq 1$), which is appropriate only for binary or multi-label tasks. In mutual-exclusive multi-class problems (a flower can only be one species), Softmax normalizes exponential logits across all classes such that $\sum_{k=1}^K P(y = k) = 1.0$, allowing rigorous probabilistic interpretation.

#### Q3: What is the Vanishing Gradient problem and how did we prevent it?
**Answer:** In deep networks using Sigmoid or Tanh activations, the derivative saturates near 0 when $|z|$ is large ($\sigma'(z) \le 0.25$). Multiplying these small derivatives across layers causes gradients to vanish exponentially during backpropagation. We prevented this by utilizing **ReLU** ($\max(0, z)$), whose derivative is constant ($1.0$) for all positive activations.

#### Q4: Why is He (Kaiming) initialization necessary when using ReLU?
**Answer:** Since ReLU deactivates approximately half of the neurons (outputting zero for $z < 0$), standard normal initialization leads to halving of signal variance at each layer. He initialization draws weights from $\mathcal{N}(0, \sqrt{2 / n_{in}})$, mathematically preserving activation variance across depth.

#### Q5: What is the difference between Batch Gradient Descent, SGD, and Mini-Batch Gradient Descent?
**Answer:**
- **Batch GD:** Computes gradients over the entire dataset ($N=120$). Stable but slow and memory intensive.
- **Stochastic GD (SGD):** Updates parameters after each individual sample. Fast but exhibits high gradient noise.
- **Mini-Batch GD (Used here, $B=16$):** Updates weights on small batches, balancing vectorization speed and gradient stability while allowing escaping from saddle points.

#### Q6: How does $L_2$ Regularization prevent over-fitting in ANNs?
**Answer:** $L_2$ regularization (Ridge / Weight Decay) appends a penalty term $\frac{\lambda}{2} \sum \|W\|^2$ to the loss function. During gradient updates, it multiplies weights by $(1 - \alpha \lambda)$, effectively shrinking large weights and preventing individual neurons from fitting spurious noise in the training set.

---

## 8. Conclusion
The mini project successfully demonstrates that a compact, 4-layer Artificial Neural Network with ReLU activations and Adam optimization can achieve perfect classification accuracy ($100\%$) on the Iris dataset. The modular code structure, absence of fragile external dependencies, and accompanying interactive Web Studio provide a comprehensive, academically sound submission for Deep Learning coursework.
