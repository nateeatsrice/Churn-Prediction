# Telco Churn Prediction Model

This project builds and deploys a churn prediction model using the Telco Customer Churn dataset. The goal is to predict whether a customer will churn based on their account and service usage information.
---

## Project Overview

The model is trained on a binary classification dataset and served using a Flask API. It includes tools for testing, environment management, and containerization.
---

## 📁 Project Structure

```
├── predict.py              # Main Flask app to load model and serve predictions
├── predict-test.py         # Script to interface with the Flask API for testing
├── model.pkl               # Trained binary churn prediction model
├── Pipfile                 # Pipenv environment definition
├── Pipfile.lock            # Lock file for reproducible builds
├── Dockerfile              # Container specification for the application
└── README.md               # Project documentation
```
---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/telco-churn-prediction.git
cd telco-churn-prediction
```
---
### 2. Install dependencies (via Pipenv)

Make sure you have `pipenv` installed:

```bash
pip install pipenv
pipenv install
```

### 3. Run the Flask app

```bash
pipenv run python predict.py
```

The API will be hosted locally at `http://localhost:9698/`.

---

### API Usage

You can use `predict-test.py` to send test requests to the running API:

```bash
pipenv run python predict-test.py
```

This script sends a sample JSON payload to the `/predict` endpoint and prints the response for the batch.

### 4. Run WSGI server

```bash
gunicorn --bind 0.0.0.0:9698 predict:app
```
```bash
python predict-test.py
```
---

## 🐳 Docker Usage

To build and run the app in a Docker container:

### 1. Build the Docker image

```bash
docker build -t churn_prediction .
```

### 2. Run the container

```bash
docker run -it --rm -p 9698:9698 churn-prediction
```
---

## Model Details

> The model is trained offline and serialized using  `pickle` as `churn-model.bin`.
---

## 🔒 Environment

The project uses **Pipenv** for virtual environment and dependency management. All dependencies are listed in `Pipfile`, and reproducibility is ensured via `Pipfile.lock`.
---

## 🧾 License

MIT License — feel free to use and adapt for personal or educational purposes.