# CaviScan

A Streamlit app for detecting dental cavities from X-ray images using Roboflow Serverless Inference API. Upload an image, view detections with bounding boxes, and generate a shareable PDF report.

## Features

- Upload dental X-ray images (.jpg, .jpeg, .png)
- Serverless cavity detection via Roboflow Inference SDK
- Visualized results with bounding boxes and confidence scores
- One-click PDF report generation (includes date, summary, and processed image)

## Prerequisites

- Python 3.10 or newer
- A Roboflow API key (free to obtain from your Roboflow account)

## Quickstart

1) Clone and enter the project folder

    git clone https://github.com/PalashRajpara/CaviScan.git
    cd CaviScan

2) Create and activate a virtual environment (recommended)

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip

3) Install dependencies

    pip install -r requirements.txt

4) Configure your Roboflow API key

The app currently expects an API key where marked in app.py:

    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="Your_API_Key_Here"
    )

- Option A (quick): Replace "Your_API_Key_Here" with your actual key.
- Option B (preferred): Use an environment variable and read it in code.

To set an environment variable in zsh/macOS for your current shell session:

    export ROBOFLOW_API_KEY="<your_api_key>"

Then update the initialization in app.py like this (optional improvement):

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        st.error("Missing ROBOFLOW_API_KEY environment variable"); st.stop()
    CLIENT = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=api_key)

5) Run the app

    streamlit run app.py

The app will open at http://localhost:8501.

## How to use

1. Upload a dental X-ray image from the sidebar.
2. Review detections and confidence scores; the processed image will show bounding boxes.
3. In the sidebar, click "Generate Report" to create a PDF, then download it.

## Configuration

- Model: The app calls Roboflow with model_id="cavity1.0/1". If you have a different model/version, update the model_id in app.py.
- Temporary files: Images are written to temporary files for processing and cleaned up after use.

## Project structure

    app.py             # Streamlit app (UI, inference, PDF generation)
    requirements.txt   # Python dependencies
    README.md          # This file# CaviScan
