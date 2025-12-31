# ECG Atrial Fibrillation Prediction

A deep learning framework for prediction of atrial fibrillation risk from electrocardiogram (ECG) waveforms using a compact convolutional transformer-based architecture.
 
**Pre-Trained Model Weights**: [Hugging Face](https://huggingface.co/bcs0098330/ecg_cct_small/tree/main)

## Installation

```bash
# Clone the repository
git clone https://github.com/briancs9/ecg_vision.git
cd ecg_vision

# Install dependencies
pip install -r requirements.txt
```

### Downloading Model Weights

The pre-trained model weights are available on Hugging Face:

```bash
from huggingface_hub import hf_hub_download

# Download model checkpoint
model_path = hf_hub_download(
    repo_id="bcs0098330/ecg_cct_small",
    filename="ecg_cct_small.pth",
    local_dir="./models"
)

# Download config file
config_path = hf_hub_download(
    repo_id="bcs0098330/ecg_cct_small",
    filename="config.json",
    local_dir="./models"
)
```

After downloading, you can use the model checkpoint file with the inference script:

```bash
# With CSV file
python inference_file.py -f /path/to/<your_ecg_file.csv> -m ./models/ecg_cct_small.pth

# With XML file (automatic conversion)
python inference_file.py -f /path/to/<your_ecg_file.xml> -m ./models/ecg_cct_small.pth
```

## Data Preparation

### Converting XML Files to CSV

The `inference_file.py` script can automatically convert XML files to CSV format before running inference. However, if you prefer to convert files separately, you can use the `ecg_xml_parser.py` script directly. This script extracts ECG lead data from MUSE XML files and saves it as CSV files.

#### Single File Conversion

Convert a single XML file to CSV:

```bash
python ecg_xml_parser.py /path/to/ecg_file.xml
```

The output CSV file will be saved in the same directory as the input XML file with the same base name.

#### Directory Conversion

Convert all XML files in a directory:

```bash
python ecg_xml_parser.py /path/to/xml_directory
```


#### Custom Output Directory

Specify a custom output directory for the CSV files:

```bash
python ecg_xml_parser.py /path/to/ecg_file.xml --output_dir /path/to/output_directory
```

#### Including Calculated Leads

By default, the parser extracts only the 8 independent leads (I, II, V1-V6). To include calculated leads (III, aVR, aVL, aVF) for a standard 12-lead ECG:

```bash
python ecg_xml_parser.py /path/to/ecg_file.xml --include-calculated-leads
```

**Note**: The inference model expects 8 leads, so if you include calculated leads, you may need to select only the first 8 columns (I, II, V1-V6) before running inference.

## Data Format

### Input Data

The inference script accepts both **CSV files** and **XML files**:

- **CSV files**: Should contain 8 columns representing different ECG leads (I, II, V1, V2, V3, V4, V5, V6)
  - Header row with column names (optional)
  - Each CSV file represents one ECG recording
  - Numerical values representing ECG signal amplitudes across time

- **XML files**: GE MUSE system XML format
  - The script automatically detects XML files and converts them to CSV before inference
  - Supports both single XML files and directories containing XML files
  - Conversion happens automatically - no manual conversion needed

If you've converted from XML using `ecg_xml_parser.py` or the automatic conversion in `inference_file.py`, the files will already be in the correct format.

## Usage

#### Single File Inference

Run inference on a single ECG file (CSV or XML):

```bash
# CSV or XML file
python inference_file.py -f /path/to/<your_ecg_file> -m model.pth

```

#### Directory Inference

Run inference on all files in a directory (CSV or XML):

```bash
# CSV or XML files in directory
python inference_file.py -d /path/to/<your_directory> -m model.pth

```

#### Custom Output Path

Specify a custom output file for predictions:

```bash
python inference_file.py -f /path/to/ecg_file.csv -m model.pth --output predictions.csv
```

#### Custom Batch Size

Override the default batch size:

```bash
python inference_file.py -d /path/to/csv_directory -m model.pth --batch_size 64
```

#### Explicit Config File

Provide a config file explicitly (otherwise auto-detected from model directory):

```bash
python inference_file.py -f /path/to/ecg_file.csv -m model.pth --config config.json
```

The inference script will automatically:
- Detect XML files and convert them to CSV format
- Detect the config file from the model directory if available
- Generate predictions for all input files
- Save predictions to a CSV file with filenames and probabilities
- Display summary statistics including the range and mean of predictions


## Device Support

The code automatically detects and uses:
- **CUDA** (if available) - for NVIDIA GPUs
- **MPS** (Apple Silicon, if available) - for Apple Silicon Macs
- **CPU** (fallback)

Device can be explicitly set in config JSON or will be auto-detected.

