# Vietnamese Handwriting OCR

A Vietnamese handwritten character recognition system using Deep Learning.

## Description

This project builds a specialized OCR (Optical Character Recognition) system for recognizing Vietnamese handwritten text. It uses state-of-the-art Deep Learning models to achieve high accuracy in character recognition.

## Project Structure

```
vietnamese-handwriting-ocr/
│
├── src/                    # Main source code directory
│   ├── data/              # Data processing
│   ├── models/            # Model definitions
│   ├── train/             # Model training
│   ├── evaluate/          # Model evaluation
│   └── inference/         # Prediction on new data
│
├── app.py                 # Main application (Streamlit/Flask)
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── config.py              # Project configuration
├── requirements.txt       # Project dependencies
├── .gitignore            # Git ignore file
└── README.md             # This documentation
```

## Requirements

- Python 3.8+
- pip or conda

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd vietnamese-handwriting-ocr
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Train the model
```bash
python train.py
```

### Evaluate the model
```bash
python evaluate.py
```

### Run the application
```bash
python app.py
```

## Data

- Place training data in `data/raw/`
- Processed data will be saved to `data/processed/`

## Models

- Models are saved in the `models/` directory
- Model details are defined in `src/models/`

## Results

- Training results are saved in `logs/`
- Evaluation results are saved in `results/`

## Author

Dang Nguyen Hoai

## License

[Choose an appropriate license, e.g., MIT]
