# AI Server with FastAPI

This project is a FastAPI-based server for AI model inference and explanation using LIME and SHAP. It includes endpoints for predictions and explanations, and it is containerized with Docker for easy deployment.

## Features

- **Prediction Endpoint**: Upload an image to get a prediction from the AI model.
- **LIME Explanation**: Generate a LIME explanation for the prediction.
- **SHAP Explanation**: Generate a SHAP explanation for the prediction.
- **CORS Enabled**: Allows access from Flutter or other frontends.

---

## Prerequisites

- Python 3.9 or higher
- [Virtualenv](https://virtualenv.pypa.io/en/latest/) for managing Python environments
- Docker (optional, for containerized deployment)

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/vinsensiuskurniaputra/plant_classification_api.git
cd ai_server_with_fast_api
```

### 2. Create and Activate a Virtual Environment

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the Model File

Ensure the model file (`mobilenetv2_model.h5`) is placed in the root directory of the project.

### 5. Run the Application

```bash
uvicorn app:app --reload
```

The server will start at `http://127.0.0.1:8000`.

---

## API Endpoints

### 1. **Root Endpoint**
- **URL**: `/`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "message": "Model server running"
  }
  ```

### 2. **Prediction Endpoint**
- **URL**: `/predict`
- **Method**: `POST`
- **Request**: Upload an image file.
- **Response**:
  ```json
  {
    "prediction": <label>,
    "confidence": <confidence_score>
  }
  ```

### 3. **LIME Explanation Endpoint**
- **URL**: `/explain/lime`
- **Method**: `POST`
- **Request**: Upload an image file.
- **Response**:
  ```json
  {
    "message": "LIME image saved as lime_result.png",
    "label": <label>
  }
  ```

### 4. **SHAP Explanation Endpoint**
- **URL**: `/explain/shap`
- **Method**: `POST`
- **Request**: Upload an image file.
- **Response**:
  ```json
  {
    "message": "SHAP image saved as shap_result.png",
    "label": <label>
  }
  ```

---

## Docker Deployment (Optional)

### 1. Build the Docker Image

```bash
docker-compose build
```

### 2. Run the Docker Container

```bash
docker-compose up
```

The server will be available at `http://localhost:8000`.

---

## Notes

- **Model File**: Ensure the `mobilenetv2_model.h5` file is present in the root directory or update the `MODEL_FILE` environment variable to point to the correct path.
- **Generated Files**: LIME and SHAP explanations are saved as `lime_result.png` and `shap_result.png` in the project directory.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.