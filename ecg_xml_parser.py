#!/usr/bin/env python
"""
Distilled ECG XML Parser
Extracts ECG lead data from MUSE XML files and returns as numpy array

Based on the original musexmlex.py by Anthony D. Ricke (GE Healthcare)
"""

import numpy as np
import xml.parsers.expat
import codecs
import base64
import struct
import re
from typing import Dict, List, Tuple, Optional


def parse_muse_xml_to_numpy(xml_file_path: str, encoding: Optional[str] = None, include_calculated_leads: bool = False) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Parse a MUSE XML ECG file and return lead data as a numpy array.
    
    Args:
        xml_file_path (str): Path to the XML file
        encoding (str, optional): File encoding. If None, will auto-detect from XML header.
        include_calculated_leads (bool): If True, includes calculated leads (III, aVR, aVL, aVF)
                                       to make a standard 12-lead ECG. If False, returns only
                                       the independent leads from the XML file.
        
    Returns:
        Tuple[np.ndarray, Dict]: 
            - numpy array with shape (n_samples, n_leads) containing ECG data
            - metadata dictionary with sample_rate, units, lead_names, etc.
    
    Raises:
        ValueError: If required ECG leads are missing or file cannot be parsed
        FileNotFoundError: If XML file doesn't exist
    """
    
    # Auto-detect encoding if not provided
    if encoding is None:
        encoding = _detect_xml_encoding(xml_file_path)
        if not encoding:
            raise ValueError("Cannot determine file encoding from XML file")
    
    # Parse the XML file
    parser = MuseXmlParser()
    parser.parse_file(xml_file_path, encoding)
    
    # Convert to numpy array
    ecg_data = parser.get_ecg_data_as_numpy(include_calculated_leads)
    metadata = parser.get_metadata(include_calculated_leads)
    
    return ecg_data, metadata


def _detect_xml_encoding(xml_file_path: str) -> str:
    """Detect XML file encoding from the XML declaration."""
    try:
        with open(xml_file_path, 'rb') as fid:
            pattern = rb"<\?xml\s+.*encoding=\"([\w-]+)\"\?>"
            for line in fid:
                result = re.match(pattern, line)
                if result:
                    return result.group(1).decode('ascii')
    except Exception:
        pass
    return ""


class MuseXmlParser:
    """Simplified parser for MUSE XML ECG files."""
    
    def __init__(self):
        self.ecg_data = {}  # Dict[str, bytes] - lead_id -> raw binary data
        self.ecg_leads = []  # List[str] - ordered list of lead IDs
        self.found_rhythm = False
        self.sample_rate = 0
        self.adu_gain = 1.0
        self.units = ""
        self.current_lead_id = ""
        
        # XML parsing state
        self._current_element = None
        self._element_data = ""
        self._in_waveform = False
        self._in_lead_data = False
        self._in_waveform_data = False
        
    def parse_file(self, xml_file_path: str, encoding: str):
        """Parse the XML file and extract ECG data."""
        parser = xml.parsers.expat.ParserCreate()
        parser.StartElementHandler = self._start_element
        parser.EndElementHandler = self._end_element
        parser.CharacterDataHandler = self._char_data
        
        with codecs.open(xml_file_path, mode='r', encoding=encoding) as f:
            parser.Parse(f.read())
    
    def _start_element(self, name, attrs):
        """Handle XML start element."""
        self._current_element = name
        self._element_data = ""
        
        if name == "Waveform":
            self._in_waveform = True
        elif name == "LeadData":
            self._in_lead_data = True
        elif name == "WaveFormData":
            self._in_waveform_data = True
    
    def _end_element(self, name):
        """Handle XML end element."""
        if name == "Waveform":
            self._in_waveform = False
        elif name == "LeadData":
            self._in_lead_data = False
        elif name == "LeadID" and self._in_lead_data and self.found_rhythm:
            self.current_lead_id = self._element_data.strip()
        elif name == "WaveFormData" and self._in_waveform_data and self.found_rhythm:
            if self.current_lead_id:
                # Decode base64 data and store
                raw_data = base64.b64decode(self._element_data)
                self.ecg_data[self.current_lead_id] = raw_data
                self.ecg_leads.append(self.current_lead_id)
        elif name == "WaveformType" and self._in_waveform:
            if "Rhythm" in self._element_data:
                self.found_rhythm = True
        elif name == "SampleBase" and self._in_waveform and self.found_rhythm:
            self.sample_rate = int(self._element_data.strip())
        elif name == "LeadAmplitudeUnitsPerBit" and self._in_lead_data:
            self.adu_gain = float(self._element_data.strip())
        elif name == "LeadAmplitudeUnits" and self._in_lead_data:
            self.units = self._element_data.strip()
        
        self._current_element = None
    
    def _char_data(self, data):
        """Handle XML character data."""
        if self._current_element:
            self._element_data += data
    
    def get_ecg_data_as_numpy(self, include_calculated_leads: bool = False) -> np.ndarray:
        """Convert ECG data to numpy array with shape (n_samples, n_leads).
        
        Args:
            include_calculated_leads (bool): If True, includes calculated leads (III, aVR, aVL, aVF)
                                           to make a standard 12-lead ECG. If False, returns only
                                           the independent leads from the XML file.
        """
        if not self.ecg_data:
            raise ValueError("No ECG data found in XML file")
        
        # Get number of samples from first lead
        first_lead = self.ecg_leads[0]
        n_samples = len(self.ecg_data[first_lead]) // 2  # 2 bytes per sample
        
        # Create numpy array for independent leads
        independent_leads = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
        n_independent = len(independent_leads)
        
        if include_calculated_leads:
            # Standard 12-lead ECG: 8 independent + 4 calculated
            ecg_array = np.zeros((n_samples, 12), dtype=np.float64)
            lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        else:
            # Only independent leads
            ecg_array = np.zeros((n_samples, n_independent), dtype=np.float64)
            lead_names = independent_leads
        
        # Fill array with independent lead data
        for lead_idx, lead_id in enumerate(independent_leads):
            if lead_id in self.ecg_data:
                raw_data = self.ecg_data[lead_id]
                # Convert binary data to numpy array (2 bytes per sample, 16-bit signed integer)
                samples = np.frombuffer(raw_data, dtype=np.int16)
                # Apply gain and store in array
                ecg_array[:, lead_idx] = samples * self.adu_gain
        
        # Calculate derived leads if requested
        if include_calculated_leads:
            # Get Lead I and II data
            lead_i = ecg_array[:, 0]  # Lead I
            lead_ii = ecg_array[:, 1]  # Lead II
            
            # Calculate derived leads
            ecg_array[:, 2] = lead_ii - lead_i  # III = II - I
            ecg_array[:, 3] = -(lead_i + lead_ii) / 2  # aVR = -(I + II)/2
            ecg_array[:, 4] = lead_i - lead_ii / 2  # aVL = I - II/2
            ecg_array[:, 5] = lead_ii - lead_i / 2  # aVF = II - I/2
            
            # Precordial leads (V1-V6) are already in positions 6-11
        
        return ecg_array
    
    def get_metadata(self, include_calculated_leads: bool = False) -> Dict[str, any]:
        """Get metadata about the ECG data."""
        if include_calculated_leads:
            lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        else:
            lead_names = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
            
        return {
            'sample_rate': self.sample_rate,
            'units': self.units,
            'adu_gain': self.adu_gain,
            'lead_names': lead_names,
            'independent_leads': self.ecg_leads.copy(),
            'n_samples': len(self.ecg_data[self.ecg_leads[0]]) // 2 if self.ecg_leads else 0,
            'n_leads': len(lead_names),
            'includes_calculated_leads': include_calculated_leads
        }


# Convenience function for backward compatibility
def extract_ecg_data(xml_file_path: str, include_calculated_leads: bool = False) -> np.ndarray:
    """
    Simple function to extract ECG data from XML file.
    
    Args:
        xml_file_path (str): Path to the XML file
        include_calculated_leads (bool): If True, includes calculated leads (III, aVR, aVL, aVF)
                                       to make a standard 12-lead ECG. If False, returns only
                                       the independent leads from the XML file.
        
    Returns:
        np.ndarray: ECG data with shape (n_samples, n_leads)
    """
    ecg_data, _ = parse_muse_xml_to_numpy(xml_file_path, include_calculated_leads=include_calculated_leads)
    return ecg_data


# PyTorch-compatible transform classes
try:
    import torch
    from torch.utils.data import Dataset
    from torchvision.transforms import Compose
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class ECGExtractTransform:
        """
        PyTorch transform to extract ECG data from XML files.
        Can be used in torchvision.transforms.Compose for data preprocessing pipelines.
        """
        
        def __init__(self, include_calculated_leads: bool = False, encoding: Optional[str] = None):
            """
            Initialize the ECG extraction transform.
            
            Args:
                include_calculated_leads (bool): If True, includes calculated leads (III, aVR, aVL, aVF)
                                               to make a standard 12-lead ECG. If False, returns only
                                               the independent leads from the XML file.
                encoding (str, optional): File encoding. If None, will auto-detect from XML header.
            """
            self.include_calculated_leads = include_calculated_leads
            self.encoding = encoding
        
        def __call__(self, xml_file_path: str) -> torch.Tensor:
            """
            Extract ECG data from XML file and return as PyTorch tensor.
            
            Args:
                xml_file_path (str): Path to the XML file
                
            Returns:
                torch.Tensor: ECG data with shape (n_leads, n_samples) - channels first format
            """
            ecg_data, _ = parse_muse_xml_to_numpy(
                xml_file_path, 
                encoding=self.encoding,
                include_calculated_leads=self.include_calculated_leads
            )
            
            # Convert to PyTorch tensor and transpose to channels-first format (n_leads, n_samples)
            return torch.from_numpy(ecg_data.T).float()
        
        def __repr__(self):
            return f"ECGExtractTransform(include_calculated_leads={self.include_calculated_leads}, encoding={self.encoding})"


    class ECGExtractWithMetadataTransform:
        """
        PyTorch transform to extract ECG data and metadata from XML files.
        Returns both the ECG data tensor and metadata dictionary.
        """
        
        def __init__(self, include_calculated_leads: bool = False, encoding: Optional[str] = None):
            """
            Initialize the ECG extraction transform with metadata.
            
            Args:
                include_calculated_leads (bool): If True, includes calculated leads (III, aVR, aVL, aVF)
                                               to make a standard 12-lead ECG. If False, returns only
                                               the independent leads from the XML file.
                encoding (str, optional): File encoding. If None, will auto-detect from XML header.
            """
            self.include_calculated_leads = include_calculated_leads
            self.encoding = encoding
        
        def __call__(self, xml_file_path: str) -> Tuple[torch.Tensor, Dict[str, any]]:
            """
            Extract ECG data and metadata from XML file.
            
            Args:
                xml_file_path (str): Path to the XML file
                
            Returns:
                Tuple[torch.Tensor, Dict]: ECG data tensor and metadata dictionary
            """
            ecg_data, metadata = parse_muse_xml_to_numpy(
                xml_file_path, 
                encoding=self.encoding,
                include_calculated_leads=self.include_calculated_leads
            )
            
            # Convert to PyTorch tensor and transpose to channels-first format (n_leads, n_samples)
            ecg_tensor = torch.from_numpy(ecg_data.T).float()
            
            return ecg_tensor, metadata
        
        def __repr__(self):
            return f"ECGExtractWithMetadataTransform(include_calculated_leads={self.include_calculated_leads}, encoding={self.encoding})"


    class ECGDataset(Dataset):
        """
        PyTorch Dataset for ECG XML files.
        Automatically handles file loading and can be used with DataLoader.
        """
        
        def __init__(self, xml_file_paths: List[str], transform=None, include_calculated_leads: bool = False):
            """
            Initialize the ECG dataset.
            
            Args:
                xml_file_paths (List[str]): List of paths to XML files
                transform (callable, optional): Optional transform to be applied to each sample
                include_calculated_leads (bool): If True, includes calculated leads (III, aVR, aVL, aVF)
                                               to make a standard 12-lead ECG. If False, returns only
                                               the independent leads from the XML file.
            """
            self.xml_file_paths = xml_file_paths
            self.transform = transform
            self.include_calculated_leads = include_calculated_leads
        
        def __len__(self):
            return len(self.xml_file_paths)
        
        def __getitem__(self, idx):
            xml_file_path = self.xml_file_paths[idx]
            
            # Extract ECG data
            ecg_data, metadata = parse_muse_xml_to_numpy(
                xml_file_path, 
                include_calculated_leads=self.include_calculated_leads
            )
            
            # Convert to PyTorch tensor (channels first)
            ecg_tensor = torch.from_numpy(ecg_data.T).float()
            
            # Apply transform if provided
            if self.transform:
                ecg_tensor = self.transform(ecg_tensor)
            
            return ecg_tensor, metadata
        
        def __repr__(self):
            return f"ECGDataset(num_files={len(self.xml_file_paths)}, include_calculated_leads={self.include_calculated_leads})"


if __name__ == "__main__":
    import sys
    import os
    import argparse
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(description='Convert MUSE XML ECG files to CSV format')
    parser.add_argument('input_path', help='Path to XML file or directory containing XML files')
    parser.add_argument('--output_dir', '-o', help='Output directory for CSV files (default: same as input)')
    parser.add_argument('--include-calculated-leads', action='store_true', default=False,
                       help='Include calculated leads (III, aVR, aVL, aVF) in output (default: False)')
    parser.add_argument('--no-calculated-leads', dest='include_calculated_leads', action='store_false',
                       help='Exclude calculated leads from output')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if input_path.is_file():
            output_dir = input_path.parent
        else:
            output_dir = input_path
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of XML files to process
    if input_path.is_file():
        xml_files = [input_path]
    elif input_path.is_dir():
        xml_files = list(input_path.glob('*.xml')) + list(input_path.glob('*.XML'))
        if not xml_files:
            print(f"No XML files found in directory: {input_path}")
            sys.exit(1)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        sys.exit(1)
    
    print(f"Processing {len(xml_files)} XML file(s)...")
    print(f"Output directory: {output_dir}")
    print(f"Include calculated leads: {args.include_calculated_leads}")
    print("-" * 50)
    
    successful = 0
    failed = 0
    
    # Process files with progress bar
    for xml_file in tqdm(xml_files, desc="Processing XML files", unit="file"):
        try:
            # Parse ECG data
            ecg_data, metadata = parse_muse_xml_to_numpy(
                str(xml_file), 
                include_calculated_leads=args.include_calculated_leads
            )
            
            # Check for NaN values in the ECG data
            if np.isnan(ecg_data).any():
                nan_count = np.isnan(ecg_data).sum()
                total_values = ecg_data.size
                tqdm.write(f"Warning: {xml_file.name} contains {nan_count} NaN values out of {total_values} total values")
                tqdm.write(f"Skipping {xml_file.name} due to NaN values in ECG data")
                failed += 1
                continue
            
            # Create DataFrame with lead names as columns (no metadata)
            df = pd.DataFrame(ecg_data, columns=metadata['lead_names'])
            
            # Generate output filename with cleaned name (replace parentheses with underscores, remove spaces)
            cleaned_name = xml_file.stem.replace('(', '_').replace(')', '_').replace(' ', '')
            output_file = output_dir / f"{cleaned_name}.csv"
            
            # Save as CSV
            df.to_csv(output_file, index=False)
            
            successful += 1
            
        except Exception as e:
            tqdm.write(f"Error processing {xml_file.name}: {e}")
            failed += 1
    
    print("-" * 50)
    print(f"Processing complete: {successful} successful, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
