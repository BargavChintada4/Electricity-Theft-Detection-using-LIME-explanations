# ⚡ Electricity Theft Detection using Deep Learning and XAI

This repository implements **electricity theft detection** using deep learning models on **smart meter time-series data**.  
The project focuses on developing models that are not only **accurate** but also **interpretable**, by leveraging **LIME (Local Interpretable Model-Agnostic Explanations)**.

---

## 📌 Project Overview
Electricity theft causes billions of dollars in losses and creates grid instability.  
This project investigates two deep learning methods for classifying consumption patterns:  
- **LSTM (Long Short-Term Memory)** → Captures sequential dependencies and sudden anomalies.  
- **Transformer (Self-Attention Model)** → Learns long-range and global usage patterns across time.  

Both models are explained using **LIME**, which highlights the exact time intervals that influenced their predictions, ensuring **trustworthy AI-based fraud detection**.

---

## 🚀 Features
- ✅ **Handles Class Imbalance** using class weights.  
- ✅ **Evaluates Model Performance** with Accuracy, Precision, Recall, and **F1-Score**.  
- ✅ **Post-hoc Explanations with LIME** → interpretable decisions for flagged theft cases.  
- ✅ **Threshold Tuning** → balances precision & recall to avoid false accusations.  
- ✅ **Ready for Utility-Scale Deployment** with efficient training and explainable monitoring.  

---

## 📊 Model Results
- **LSTM Model**  
  - Test Accuracy: **89.59%**  
  - Test Loss: **0.2874**  
  - Specializes in detecting **sharp anomalies (spikes, drops, irregular usage patterns)**  

- **Transformer Model**  
  - Test Accuracy: **72.50%**  
  - Test Loss: **0.2935**  
  - Effective at identifying **distributed anomalies (multi-day cycles, subtle shifts)**  

---

### How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/<your-username>/<your-repo>.git
    cd <your-repo>
    ```
2.  **Install dependencies:** Make sure you have TensorFlow, TensorFlow Hub, Pandas, and Scikit-learn installed.
    ```bash
    # Using conda
    conda create -n theft-detection python=3.10 -y
    conda activate theft-detection
    
    # OR using venv
    python -m venv venv
    source venv/bin/activate   # Linux / Mac
    venv\Scripts\activate      # Windows

    ```
3.  **Install Dependencies**:
    ```bash
    pip install -U pip
    pip install numpy pandas scikit-learn matplotlib tensorflow lime tqdm notebook
    ```
4. **Run the Notebooks**:
    ```bash
    jupyter notebook
    ```
    Then open:
    `LSTM_LIME_Theft.ipynb` and `LIME_TRANSFORM_Theft.ipynb`
5. **Run in Google Colab (Optional)**:
- Upload the notebooks (`LSTM_LIME_Theft.ipynb`, `LIME_TRANSFORM_Theft.ipynb`) to Google Colab  
- Set runtime type to **GPU** (for faster training)  
- Install dependencies inside Colab if needed:
```bash
!pip install numpy pandas scikit-learn matplotlib tensorflow lime tqdm
```

---

## ▶️ Running the Notebooks
1. Open **LSTM_LIME_Theft.ipynb** → Trains LSTM model + generates LIME explanations.  
2. Open **LIME_TRANSFORM_Theft.ipynb** → Trains Transformer model + generates LIME explanations.  

Run in **Jupyter Notebook** or **Google Colab**.

---

## 📘 Interpretability with LIME
- **LSTM + LIME** → focuses on **localized events** (midnight usage spikes, drops, weekend anomalies).  
- **Transformer + LIME** → highlights **global time patterns** (multi-day fraudulent cycles, unusual peak vs off-peak shifts).  

LIME translates model predictions into **human-understandable justifications**, crucial for utility audits and regulatory compliance.

---

## 📚 Literature References
The project draws insights from recent explainable AI (XAI) research:  
- Frequency-domain methods like **DFT-LRP**.  
- Rule-based interpretability like **DDS-XAI**.  
- Attention models (TFT) for energy system monitoring.  
- Hybrid and neural-symbolic systems for time-series explainability.  

---

## ✅ Conclusion
- **LSTM** → Higher accuracy (best for the current dataset), interprets sharp anomalies.  
- **Transformer** → Better for large-scale and long-term fraud detection in national grids.  
- **LIME** → Enhances transparency and trust, making models suitable for **real-world deployment**.  

This project demonstrates how combining **deep learning** with **explainability** can deliver reliable and ethical electricity theft detection.

---



