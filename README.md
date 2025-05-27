
# NovoRhythmia

NovoRhythmia is a predictive model designed to identify the best moments to offer or request a service — not just based on availability, but on patterns, rhythms, and context. It combines natural language descriptions of agendas with structured data to estimate the expected demand for a given time slot.

This project includes the full pipeline:
- Synthetic dataset generation
- Embedding agenda text using sentence-transformers
- Training a regression model with XGBoost
- Saving the model and exposing it through an API
- Optionally, interacting via a lightweight web interface

## 📁 Project Structure

```
novorhythmia/
├── data/                  → Dataset in CSV format
├── models/                → Saved model and embeddings
├── src/                   → Python scripts (training, API, prediction)
├── frontend/              → Optional HTML/JS interface
├── app.py                 → Gradio interface for Hugging Face Spaces
├── requirements.txt       → Python dependencies
└── README.md              → This file
```

## 🚀 How to Run

1. **Set up your environment:**

```bash
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows
pip install -r requirements.txt
```

2. **Train the model:**

```bash
python src/training.py
```

This will generate and save the model to the `models/` directory.

3. **Run the API:**

```bash
uvicorn src.api:app --reload
```

Open your browser and go to `http://127.0.0.1:8000/docs` to test the API.

4. **(Optional) Run the Gradio interface:**

```bash
python app.py
```

This will launch a web interface to interact with the model.

## 🧠 What’s Inside

The model uses:
- `sentence-transformers` to encode agenda descriptions
- `xgboost` to perform regression
- `FastAPI` to expose predictions via HTTP
- `Gradio` to build a simple front-end

## 📬 Contact

Interested in using or adapting NovoRhythmia?  
Visit us at [www.novoshore.com](https://www.novoshore.com) or email us at **info@novoshore.com**

---
