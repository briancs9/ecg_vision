# ECG Atrial Fibrillation Prediction

A deep learning framework for electrocardiogram (ECG) atrial fibrillation prediction using compact convolutional transformer-based architecture. This project provides tools for inference on XML or CSV formatted images.
 
**Pre-Trained Model Weights**: [Hugging Face](https://huggingface.co/bcs0098330/ecg_cct_small/tree/main)

## Project Structure

```
.
├── config.py              # Configuration management with JSON support
├── datasets.py            # Data loading and preprocessing utilities
├── models.py              # Transformer model architectures
├── ecg_xml_parser.py      # XML to CSV converter for GE MUSE system files
├── inference_file.py      # Main inference script
└── utils.py               # Utility functions (model creation, etc.)
```

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
pip install huggingface_hub

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

Or manually download using the link provided above.

After downloading, you can use the model checkpoint file with the inference script:

```bash
# With CSV file
python inference_file.py -f /path/to/ecg_file.csv -m ./models/ecg_cct_small.pth

# With XML file (automatic conversion)
python inference_file.py -f /path/to/ecg_file.xml -m ./models/ecg_cct_small.pth
```

## Data Preparation

### Converting XML Files to CSV

The `inference_file.py` script can automatically convert XML files from the GE MUSE system to CSV format before running inference. However, if you prefer to convert files separately, you can use the `ecg_xml_parser.py` script directly. This script extracts ECG lead data from MUSE XML files and saves it as CSV files.

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

This will process all `.xml` and `.XML` files in the specified directory and create corresponding CSV files.

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

#### Example: Batch Conversion

Convert all XML files in a directory to CSV format:

```bash
# Convert all XML files in a directory
python ecg_xml_parser.py /data/muse_xml_files --output_dir /data/csv_files

# The script will process all XML files and create corresponding CSV files
# Output files will be named based on the input XML filenames
```


## Data Format

### Input Data

The inference script accepts both **CSV files** and **XML files** (GE MUSE system format):

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

### Inference

The `inference_file.py` script supports both CSV and XML files. XML files are automatically converted to CSV format before inference.

#### Single File Inference

Run inference on a single ECG file (CSV or XML):

```bash
# CSV file
python inference_file.py -f /path/to/ecg_file.csv -m model.pth

# XML file (automatically converts to CSV)
python inference_file.py -f /path/to/ecg_file.xml -m model.pth
```

#### Directory Inference

Run inference on all files in a directory (CSV or XML):

```bash
# CSV files
python inference_file.py -d /path/to/csv_directory -m model.pth

# XML files (automatically converts all to CSV)
python inference_file.py -d /path/to/xml_directory -m model.pth
```

#### XML File Options

When processing XML files, you can specify additional options:

```bash
# Include calculated leads (III, aVR, aVL, aVF) when converting XML
python inference_file.py -f /path/to/ecg_file.xml -m model.pth --include-calculated-leads

# Specify temporary directory for converted CSV files
python inference_file.py -d /path/to/xml_directory -m model.pth --temp-dir /tmp/converted_csv
```

**Note**: The model expects 8 leads, so if you use `--include-calculated-leads`, only the first 8 columns (I, II, V1-V6) will be used for inference.

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

## Model Architecture

### ECG_Transformer

A transformer-based architecture for ECG classification:

- **Tokenizer**: Convolutional tokenizer that processes temporal information
- **Lead Collapse**: Convolutional layer to collapse lead dimension
- **Transformer Encoder**: Multi-head self-attention layers
- **Sequence Pooling**: Attention-based pooling or class token

**Features**:
- Sinusoidal or learnable positional embeddings
- Configurable number of layers, heads, and model dimension
- Supports sequence pooling or class token aggregation

**Input**: `(batch, height, width)` where height=5000 (time steps), width=8 (leads)  
**Output**: Binary classification logits (atrial fibrillation probability)

## Output Files

### Inference Outputs

The inference script generates a `predictions.csv` file containing two columns: `file`, which lists the filename of each input ECG file, and `prediction`, which provides the probability score (ranging from 0 to 1) indicating the likelihood of atrial fibrillation for that recording.

## Device Support

The code automatically detects and uses:
- **CUDA** (if available) - for NVIDIA GPUs
- **MPS** (Apple Silicon, if available) - for Apple Silicon Macs
- **CPU** (fallback)

Device can be explicitly set in config JSON or will be auto-detected.

## Example Workflow

### Direct XML Workflow (Recommended)

The simplest workflow is to use XML files directly - conversion happens automatically:

```bash
# 1. Download model weights (if not already done)
# See "Downloading Model Weights" section above

# 2. Run inference directly on XML files (automatic conversion)
python inference_file.py -d /data/muse_xml_files -m ./models/ecg_cct_small.pth --output results.csv

# 3. View results
cat results.csv
```

### Manual Conversion Workflow

If you prefer to convert XML files separately before inference:

```bash
# 1. Download model weights (if not already done)
# See "Downloading Model Weights" section above

# 2. Convert GE MUSE XML files to CSV format
python ecg_xml_parser.py /data/muse_xml_files --output_dir /data/csv_files

# 3. Run inference on the converted CSV files
python inference_file.py -d /data/csv_files -m ./models/ecg_cct_small.pth --output results.csv

# 4. View results
cat results.csv
```

### CSV-Only Workflow

If you already have CSV files, you can proceed directly to inference:

```bash
# Run inference on CSV files
python inference_file.py -d /data/csv_files -m ./models/ecg_cct_small.pth --output results.csv

# View results
cat results.csv
```

