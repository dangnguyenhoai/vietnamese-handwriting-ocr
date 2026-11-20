# Vietnamese Handwriting OCR

A Vietnamese handwritten character recognition system using Deep Learning.

## Description

This project builds a specialized OCR (Optical Character Recognition) system for recognizing Vietnamese handwritten text. It uses state-of-the-art Deep Learning models to achieve high accuracy in character recognition.

## Project Structure

```
vietnamese-handwriting-ocr/
│
├── src/
│   ├── data/
│   │   ├── UIT_HWDB_line
│   │   ├── UIT_HWDB_paragraph
│   │   ├── UIT_HWDB_word
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   ├── vocab.py
│   │   └── collate.py
│   │
│   ├── models/
│   │   ├── crnn_ctc.py
│   │   └── decoder.py
│   │
│   ├── train/
│   │   └── trainer.py
│   │
│   ├── evaluate/
│   │   ├── metrics.py
│   │   └── evaluate_line.py
│   │
│   ├── inference/
│   │   ├── predict_line.py
│   │   ├── line_segmenter.py
│   │   └── predict_page.py
│   │
│   └── utils/
│       ├── text_utils.py
│       └── image_utils.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── train_crnn.py
│   ├── evaluate.py
│   └── export_model.py
│
├── configs/
│   ├── crnn_base.example.yaml
│   └── crnn_base.yaml       # Local config (git ignored)
│
├── notebooks/
│   └── eda_analysis.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
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
python scripts/train_crnn.py
```

### Evaluate the model
```bash
python scripts/evaluate.py
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
