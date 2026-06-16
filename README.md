# 🧠 Perceptron & Gradient Descent Learning

**AI2002 — Artificial Intelligence | Module 4: Learning in AI**
FAST-NUCES Chiniot-Faisalabad | Roll No: 24F-0806

---

## 📌 Description

Implements and compares two fundamental machine learning algorithms from scratch using the **UCI Iris Dataset**. Binary classification task: **Setosa vs Non-Setosa**.

---

## 🧠 Algorithms Implemented

### 1. Perceptron Learning Rule
- Uses **step activation function**
- Updates weights **only on misclassified** samples
- Converges when zero errors remain

### 2. Gradient Descent Delta Rule
- Updates weights on **every sample** (not just wrong ones)
- Supports two activation functions:
  - **Linear** — classic delta rule
  - **Sigmoid** — smooth gradient flow
- Tracks Mean Squared Error per epoch

---

## 📊 Activation Functions

| Function | Formula | Used In |
|----------|---------|---------|
| Step | +1 if x ≥ 0 else -1 | Perceptron |
| Linear | f(x) = x | GD Delta Rule |
| Sigmoid | 1 / (1 + e^−x) | GD Delta Rule |

---

## 📈 Results

All three models are compared on:
- Train accuracy (%)
- Test accuracy (%)
- Errors / Loss per epoch (plotted)

Charts are auto-saved as `module4_results.png`.

---

## 🚀 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run
```bash
python training_preceptron_M4.py
```

Output includes:
- Training progress for all 3 models
- Final accuracy comparison table
- 3-panel matplotlib chart saved as `module4_results.png`

---

## 🗃️ Dataset

UCI Iris Dataset (via sklearn):
- 150 samples, 4 features
- 3 classes → converted to binary (Setosa=+1, Others=−1)
- 80/20 train-test split with StandardScaler normalization

---

## 📁 Project Structure

```
Perceptron-Learning/
├── training_preceptron_M4.py    # Main source code
├── requirements.txt             # Dependencies
├── module4_results.png          # Auto-saved chart after run
└── .gitignore
```

---

## 🛠️ Tech Stack

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn (dataset + preprocessing only)

---

## 📄 License

MIT License
