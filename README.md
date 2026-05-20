# Fine-Tuning Application

A full-stack application for fine-tuning large language models using QLoRA, with a Node.js/Python backend and a web frontend.

## Project Structure

```
├── backend/          # Node.js (Express) + Python API server
│   ├── index.js      # Express server entry point
│   ├── main.py       # Python fine-tuning API
│   └── Dockerfile    # Backend container config
├── frontend/         # Static web UI
│   ├── index.html    # Main application page
│   └── todo.html     # Task management page
├── hugging_face/     # Hugging Face model and dataset cache
├── qlora_finetune.py # QLoRA fine-tuning script
├── fineTningPythonScript.ipynb  # Fine-tuning notebook
├── adi.ipynb         # Additional experiments notebook
└── requirements.txt  # Python dependencies
```

## Requirements

- Python 3.10+
- Node.js 18+
- CUDA-capable GPU (recommended for fine-tuning)

## Setup

### Backend

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
cd backend && npm install
```

### Environment

Copy `frontend/.env` and fill in your Azure credentials:

```
AZURE_SUBSCRIPTION_ID=your_subscription_id
AZURE_TENANT_ID=your_tenant_id
```

## Usage

Start the backend server:

```bash
cd backend
node index.js
```

Open `frontend/index.html` in a browser to access the web UI.

## Fine-Tuning

Use `qlora_finetune.py` to fine-tune models with QLoRA:

```bash
python qlora_finetune.py
```

Or explore the Jupyter notebooks for step-by-step fine-tuning workflows.
