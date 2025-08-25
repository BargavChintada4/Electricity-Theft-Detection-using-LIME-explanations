# ⚡ Electricity Theft Detection using Deep Learning and Explainable AI

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

## ⚙️ Installation
Set up the environment using Python (>=3.8):

