import torch
import argparse
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm.auto import tqdm
from types import SimpleNamespace
import models
from utils import create_model
import datasets
import tempfile
import shutil
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

STANDARD_CONFIG = SimpleNamespace(
    num_classes=1,
    d_model=128,
    num_heads=8,
    te_dropout=0.1,
    use_pos_encoding=True,
    num_transformer_layers=3,
    device=device,
    transformer_model='transformer',
    batch_size=128,
    seq_pool=True,
    mlp_ratio=4,
)

try:
    from ecg_xml_parser import parse_muse_xml_to_numpy
    XML_PARSER_AVAILABLE = True
except ImportError:
    XML_PARSER_AVAILABLE = False
    print("Warning: ecg_xml_parser module not found. XML file support will be disabled.")


def load_model_from_checkpoint(model_path, config):

    model = create_model(config, models)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If checkpoint is just the state dict
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def convert_xml_to_csv(xml_file_path: str, output_path: str = None, include_calculated_leads: bool = False) -> str:

    if not XML_PARSER_AVAILABLE:
        raise ImportError("ecg_xml_parser module is not available. Cannot convert XML files.")
    
    ecg_data, metadata = parse_muse_xml_to_numpy(
        xml_file_path, 
        include_calculated_leads=include_calculated_leads
    )
    
    if np.isnan(ecg_data).any():
        raise ValueError(f"XML file {xml_file_path} contains NaN values in ECG data")
    
    df = pd.DataFrame(ecg_data, columns=metadata['lead_names'])
    
    if output_path is None:
        xml_path = Path(xml_file_path)
        cleaned_name = xml_path.stem.replace('(', '_').replace(')', '_').replace(' ', '')
        output_path = xml_path.parent / f"{cleaned_name}.csv"
    else:
        output_path = Path(output_path)
    
    df.to_csv(output_path, index=False)
    
    return str(output_path)


def convert_xml_directory(xml_directory: str, output_directory: str = None, include_calculated_leads: bool = False) -> str:

    if not XML_PARSER_AVAILABLE:
        raise ImportError("ecg_xml_parser module is not available. Cannot convert XML files.")
    
    xml_dir = Path(xml_directory)
    
    # Get all XML files
    xml_files = sorted(list(xml_dir.glob('*.xml')) + list(xml_dir.glob('*.XML')))
    
    if not xml_files:
        raise ValueError(f"No XML files found in directory: {xml_directory}")
    
    # Determine output directory
    if output_directory is None:
        output_dir = xml_dir
    else:
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {len(xml_files)} XML file(s) to CSV...")
    
    converted_files = []
    failed_files = []
    
    for xml_file in tqdm(xml_files, desc="Converting XML files", unit="file"):
        try:
            # Generate output filename
            cleaned_name = xml_file.stem.replace('(', '_').replace(')', '_').replace(' ', '')
            csv_path = output_dir / f"{cleaned_name}.csv"
            
            # Convert XML to CSV
            convert_xml_to_csv(str(xml_file), str(csv_path), include_calculated_leads)
            converted_files.append(csv_path)
        except Exception as e:
            print(f"Warning: Failed to convert {xml_file.name}: {e}")
            failed_files.append(xml_file)
    
    if failed_files:
        print(f"Warning: {len(failed_files)} file(s) failed to convert")
    
    print(f"Successfully converted {len(converted_files)} file(s) to CSV format")
    
    return str(output_dir)


def create_transform():

    transform_img = v2.Compose([
        v2.Lambda(lambda x: x.iloc[:,0:8].values),
        v2.Lambda(lambda x: torch.tensor(x.astype(np.float32), dtype=torch.float32)),
        v2.Lambda(lambda x: x.unsqueeze(0)),
        v2.Lambda(lambda x: datasets.column_wise_resize(x, target_rows=5000)),
        v2.Lambda(lambda x: datasets.column_wise_normalize(x)),
        v2.Lambda(lambda x: x.squeeze(0))
    ])
    
    return transform_img


class SingleFileDataset(Dataset):

    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        self.filename = os.path.basename(file_path)
        # Store filenames in a list for easy access
        self.filenames = [self.filename]
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        image = pd.read_csv(self.file_path, sep=",", header=0)
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return dummy label, we'll get filename from dataset


class DirectoryDataset(Dataset):

    def __init__(self, directory_path, transform=None):
        self.directory_path = directory_path
        self.transform = transform
        # Get all CSV files in directory
        self.csv_files = sorted([f for f in os.listdir(directory_path) 
                                 if f.endswith('.csv')])
        if not self.csv_files:
            raise ValueError(f"No CSV files found in directory: {directory_path}")
        # Store filenames for easy access
        self.filenames = self.csv_files
    
    def __len__(self):
        return len(self.csv_files)
    
    def __getitem__(self, idx):
        filename = self.csv_files[idx]
        file_path = os.path.join(self.directory_path, filename)
        image = pd.read_csv(file_path, sep=",", header=0)
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return dummy label, we'll get filename from dataset


def run_inference(model, data_loader, device, dataset, output_path=None):
    """
    Run inference on data.
    
    Args:
        model: Trained model
        data_loader: DataLoader for data
        device: Device to run inference on
        dataset: Dataset object to access filenames
        output_path: Optional path to save predictions CSV
    
    Returns:
        predictions: List of predictions (probabilities)
        filenames: List of corresponding filenames
    """
    model.eval()
    all_predictions = []
    all_filenames = []
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(data_loader, desc="Running inference")):
            images = images.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            
            # Handle different output shapes
            if outputs.dim() > 1:
                outputs = outputs.squeeze(1)
            
            # Conv model outputs logits, need to apply sigmoid
            probs = torch.sigmoid(outputs)
            
            # Convert to numpy
            probs_np = probs.cpu().numpy()
            
            all_predictions.extend(probs_np)
            
            # Get filenames from dataset
            start_idx = batch_idx * data_loader.batch_size
            end_idx = min(start_idx + len(images), len(dataset))
            for i in range(start_idx, end_idx):
                all_filenames.append(dataset.filenames[i])
    
    # Save predictions if output path is provided
    if output_path:
        results_df = pd.DataFrame({
            'file': all_filenames,
            'prediction': all_predictions
        })
        results_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
    
    return all_predictions, all_filenames


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on ECG CSV or XML file(s) using a trained model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Single CSV file inference
  python inference_file.py -f /path/to/file.csv -m model.pth
  
  # Single XML file inference (automatically converts to CSV)
  python inference_file.py -f /path/to/file.xml -m model.pth
  
  # Directory inference (CSV files)
  python inference_file.py -d /path/to/csv_dir -m model.pth
  
  # Directory inference (XML files, automatically converts to CSV)
  python inference_file.py -d /path/to/xml_dir -m model.pth
  
  # With custom output and batch size
  python inference_file.py -f /path/to/file.csv -m model.pth --output predictions.csv --batch_size 64
  
  # With custom batch size
  python inference_file.py -d /path/to/csv_dir -m model.pth --batch_size 64
        '''
    )
    
    # Mutually exclusive group for file or directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-f', '--file', type=str,
                            help='Path to a single CSV or XML file for inference')
    input_group.add_argument('-d', '--directory', type=str,
                            help='Path to directory containing CSV or XML files for inference')
    
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions CSV (optional, defaults to predictions.csv in current working directory)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for inference (overrides config default)')
    parser.add_argument('--include-calculated-leads', action='store_true', default=False,
                        help='Include calculated leads (III, aVR, aVL, aVF) when converting XML files (default: False)')
    parser.add_argument('--temp-dir', type=str, default=None,
                        help='Temporary directory for converted CSV files (default: same as input)')
    
    args = parser.parse_args()
    
    input_path = None
    is_xml = False
    temp_csv_path = None
    temp_dir_created = False
    
    if args.file:
        if not os.path.exists(args.file):
            raise ValueError(f"File does not exist: {args.file}")
        input_path = args.file
        # Check if it's XML or CSV
        if args.file.lower().endswith(('.xml',)):
            if not XML_PARSER_AVAILABLE:
                raise ImportError("XML file provided but ecg_xml_parser module is not available. "
                                "Please install required dependencies or use CSV files.")
            is_xml = True
            # Convert XML to CSV
            print(f"Detected XML file. Converting to CSV format...")
            if args.temp_dir:
                temp_dir = Path(args.temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)
            else:
                # Use same directory as XML file
                temp_dir = Path(args.file).parent
            
            temp_csv_path = convert_xml_to_csv(
                args.file, 
                output_path=str(temp_dir / f"{Path(args.file).stem}.csv"),
                include_calculated_leads=args.include_calculated_leads
            )
            input_path = temp_csv_path
            print(f"Converted XML to CSV: {temp_csv_path}")
        elif not args.file.endswith('.csv'):
            raise ValueError(f"File must be a CSV or XML file: {args.file}")
    elif args.directory:
        if not os.path.exists(args.directory):
            raise ValueError(f"Directory does not exist: {args.directory}")
        if not os.path.isdir(args.directory):
            raise ValueError(f"Path must be a directory: {args.directory}")
        
        # Check if directory contains XML files
        xml_dir = Path(args.directory)
        xml_files = list(xml_dir.glob('*.xml')) + list(xml_dir.glob('*.XML'))
        csv_files = list(xml_dir.glob('*.csv'))
        
        if xml_files and not csv_files:
            if not XML_PARSER_AVAILABLE:
                raise ImportError("XML files found but ecg_xml_parser module is not available. "
                                "Please install required dependencies or use CSV files.")
            is_xml = True
            print(f"Detected XML files in directory. Converting to CSV format...")
            if args.temp_dir:
                temp_dir = Path(args.temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_dir_created = True
            else:
                # Use same directory
                temp_dir = xml_dir
            
            converted_dir = convert_xml_directory(
                args.directory,
                output_directory=str(temp_dir),
                include_calculated_leads=args.include_calculated_leads
            )
            input_path = converted_dir
        elif xml_files and csv_files:
            # Mixed directory - prefer CSV files but warn user
            print(f"Warning: Directory contains both XML and CSV files. Using CSV files only.")
            input_path = args.directory
        else:
            # Only CSV files or empty
            input_path = args.directory
    
    config = STANDARD_CONFIG
    if args.batch_size is not None:
        config = SimpleNamespace(**{**vars(config), 'batch_size': args.batch_size})
    
    device = config.device
    print(f"Using device: {device}")
    
    model = load_model_from_checkpoint(args.model_path, config)
    transform_img = create_transform()

    if args.file:
        file_to_load = temp_csv_path if temp_csv_path else args.file
        print(f"Loading file: {file_to_load}")
        dataset = SingleFileDataset(file_to_load, transform=transform_img)
    else:
        print(f"Loading directory: {input_path}")
        dataset = DirectoryDataset(input_path, transform=transform_img)
    
    print(f"Dataset size: {len(dataset)}")
    
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Run inference
    output_path = args.output
    if output_path is None:
        # Default output path in current working directory
        output_path = os.path.join(os.getcwd(), 'predictions.csv')
    else:
        # If output path is provided but not absolute, make it relative to current working directory
        if not os.path.isabs(output_path):
            output_path = os.path.join(os.getcwd(), output_path)
    
    predictions, filenames = run_inference(model, data_loader, device, dataset, output_path)
    
    print(f"\nInference complete!")
    print(f"Total samples: {len(predictions)}")

    
    if is_xml:
        if args.temp_dir and temp_dir_created:
            try:
                shutil.rmtree(args.temp_dir)
                print(f"Cleaned up temporary directory: {args.temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up temporary directory: {e}")
        elif temp_csv_path and args.temp_dir:
            pass


if __name__ == '__main__':
    main()

