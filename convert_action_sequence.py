#!/usr/bin/env python3
"""
NPY to TXT/JSON converter for action sequence data.
Supports bidirectional conversion between .npy, .txt, and .json formats.

Usage:
    # NPY -> JSON (default)
    python convert_action_sequence.py --input action_sequence.npy
    
    # NPY -> TXT
    python convert_action_sequence.py --input action_sequence.npy --format txt
    
    # JSON -> NPY
    python convert_action_sequence.py --input action_sequence.json --to-npy
    
    # TXT -> NPY
    python convert_action_sequence.py --input action_sequence.txt --to-npy
    
    # With custom output path
    python convert_action_sequence.py --input action_sequence.npy --output my_output.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Union, Tuple


class ActionSequenceConverter:
    """Converter for action sequence data between NPY, JSON, and TXT formats."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def log(self, message: str):
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    # ==================== NPY Operations ====================
    
    def load_npy(self, filepath: Union[str, Path]) -> np.ndarray:
        """Load data from NPY file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"NPY file not found: {filepath}")
        
        data = np.load(filepath)
        self.log(f"✓ Loaded NPY: {filepath}")
        self.log(f"  Shape: {data.shape}, Dtype: {data.dtype}")
        return data
    
    def save_npy(self, data: np.ndarray, filepath: Union[str, Path]):
        """Save data to NPY file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(filepath, data)
        self.log(f"✓ Saved NPY: {filepath} (shape: {data.shape})")
    
    # ==================== JSON Operations ====================
    
    def npy_to_json(self, data: np.ndarray) -> dict:
        """Convert numpy array to JSON-serializable dictionary."""
        return {
            "metadata": {
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "rows": int(data.shape[0]),
                "columns": int(data.shape[1]) if len(data.shape) > 1 else 1,
            },
            "data": data.tolist()
        }
    
    def json_to_npy(self, json_data: dict) -> np.ndarray:
        """Convert JSON dictionary back to numpy array."""
        if "data" not in json_data:
            raise ValueError("JSON must contain 'data' key")
        
        data = np.array(json_data["data"])
        
        # Try to restore original dtype if available
        if "metadata" in json_data and "dtype" in json_data["metadata"]:
            dtype_str = json_data["metadata"]["dtype"]
            try:
                data = data.astype(np.dtype(dtype_str))
            except (TypeError, ValueError):
                self.log(f"⚠ Could not restore dtype {dtype_str}, keeping as {data.dtype}")
        
        return data
    
    def save_json(self, data: np.ndarray, filepath: Union[str, Path], 
                  indent: int = 2, compact: bool = False):
        """Save data as JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        json_data = self.npy_to_json(data)
        
        with open(filepath, 'w') as f:
            if compact:
                json.dump(json_data, f)
            else:
                json.dump(json_data, f, indent=indent)
        
        self.log(f"✓ Saved JSON: {filepath}")
    
    def load_json(self, filepath: Union[str, Path]) -> np.ndarray:
        """Load data from JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            json_data = json.load(f)
        
        data = self.json_to_npy(json_data)
        self.log(f"✓ Loaded JSON: {filepath}")
        self.log(f"  Shape: {data.shape}, Dtype: {data.dtype}")
        return data
    
    # ==================== TXT Operations ====================
    
    def save_txt(self, data: np.ndarray, filepath: Union[str, Path], 
                 fmt: str = '%.8g', delimiter: str = '\t', header: bool = True):
        """
        Save data as TXT file (space or tab-delimited).
        
        Args:
            data: Numpy array to save
            filepath: Output file path
            fmt: Format string for numbers (e.g., '%.8g', '%.6f')
            delimiter: Column delimiter (default: '\t')
            header: Include shape/dtype information in header
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            # Write metadata header
            if header:
                f.write(f"# Shape: {data.shape}\n")
                f.write(f"# Dtype: {data.dtype}\n")
                f.write(f"# Rows: {data.shape[0]}, Columns: {data.shape[1] if len(data.shape) > 1 else 1}\n")
                f.write("#\n")
            
            # Write data
            np.savetxt(f, data, fmt=fmt, delimiter=delimiter)
        
        self.log(f"✓ Saved TXT: {filepath}")
    
    def load_txt(self, filepath: Union[str, Path], delimiter: str = None) -> np.ndarray:
        """
        Load data from TXT file.
        
        Args:
            filepath: Input file path
            delimiter: Column delimiter (None = auto-detect whitespace)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"TXT file not found: {filepath}")
        
        # Auto-detect delimiter if not specified
        if delimiter is None:
            with open(filepath, 'r') as f:
                first_line = f.readline()
                while first_line.startswith('#'):
                    first_line = f.readline()
            
            if '\t' in first_line:
                delimiter = '\t'
            elif ',' in first_line:
                delimiter = ','
            # else: numpy will use any whitespace
        
        data = np.loadtxt(filepath, comments='#', delimiter=delimiter)
        
        # Ensure 2D array
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        self.log(f"✓ Loaded TXT: {filepath}")
        self.log(f"  Shape: {data.shape}, Dtype: {data.dtype}")
        return data
    
    # ==================== Main Conversion Methods ====================
    
    def convert(self, input_file: Union[str, Path], output_file: Union[str, Path] = None,
                output_format: str = None, compact: bool = False, delimiter: str = '\t',
                fmt: str = '%.8g', header: bool = True):
        """
        Convert between formats automatically based on file extensions.
        
        Args:
            input_file: Input file path
            output_file: Output file path (if None, auto-generate from input_file)
            output_format: Force output format ('npy', 'json', 'txt')
            compact: Compact JSON output (no pretty printing)
            delimiter: Delimiter for TXT files
            fmt: Number format for TXT files
            header: Include metadata header in TXT output
        """
        input_file = Path(input_file)
        input_ext = input_file.suffix.lower()
        
        # Determine output format
        if output_format is None:
            if output_file:
                output_ext = Path(output_file).suffix.lower()
                output_format = output_ext.lstrip('.')
            else:
                # Default: npy -> json, others -> npy
                output_format = 'json' if input_ext == '.npy' else 'npy'
        
        # Generate output filename if not provided
        if output_file is None:
            output_file = input_file.stem + f'.{output_format}'
        
        output_file = Path(output_file)
        
        self.log(f"\n{'='*60}")
        self.log(f"Converting: {input_file.name} ({input_ext}) → {output_file.name}")
        self.log(f"{'='*60}\n")
        
        # Load input
        if input_ext == '.npy':
            data = self.load_npy(input_file)
        elif input_ext == '.json':
            data = self.load_json(input_file)
        elif input_ext == '.txt':
            data = self.load_txt(input_file)
        else:
            raise ValueError(f"Unsupported input format: {input_ext}")
        
        # Save output
        if output_format == 'npy':
            self.save_npy(data, output_file)
        elif output_format == 'json':
            self.save_json(data, output_file, compact=compact)
        elif output_format == 'txt':
            self.save_txt(data, output_file, fmt=fmt, delimiter=delimiter, header=header)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        self.log(f"\n✓ Conversion complete!")
        return output_file


def main():
    """Command-line interface for action sequence conversion."""
    parser = argparse.ArgumentParser(
        description="Convert action sequence data between NPY, JSON, and TXT formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # NPY to JSON (default)
  python convert_action_sequence.py --input action_sequence.npy
  
  # NPY to TXT
  python convert_action_sequence.py --input action_sequence.npy --to-txt
  
  # JSON to NPY
  python convert_action_sequence.py --input action_sequence.json --to-npy
  
  # TXT to NPY with custom output
  python convert_action_sequence.py --input data.txt --to-npy --output custom_output.npy
  
  # Compact JSON (no pretty printing)
  python convert_action_sequence.py --input data.npy --format json --compact
        """
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input file path')
    parser.add_argument('-o', '--output', default=None, help='Output file path (auto-generated if not specified)')
    parser.add_argument('-f', '--format', choices=['npy', 'json', 'txt'], default=None,
                        help='Output format (auto-detected from extension if not specified)')
    parser.add_argument('--to-npy', action='store_true', help='Shortcut: convert to NPY format')
    parser.add_argument('--to-json', action='store_true', help='Shortcut: convert to JSON format')
    parser.add_argument('--to-txt', action='store_true', help='Shortcut: convert to TXT format')
    parser.add_argument('--compact', action='store_true', help='Compact JSON output (no pretty printing)')
    parser.add_argument('--delimiter', default='\t', help='Delimiter for TXT files (default: tab)')
    parser.add_argument('--fmt', default='%.8g', help='Number format for TXT (default: %%.8g)')
    parser.add_argument('--no-header', action='store_true', help='Omit metadata header in TXT output')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Handle format shortcuts
    if args.to_npy:
        args.format = 'npy'
    elif args.to_json:
        args.format = 'json'
    elif args.to_txt:
        args.format = 'txt'
    
    # Create converter and run conversion
    converter = ActionSequenceConverter(verbose=not args.quiet)
    
    try:
        converter.convert(
            args.input,
            output_file=args.output,
            output_format=args.format,
            compact=args.compact,
            delimiter=args.delimiter or '\t',
            fmt=args.fmt or '%.8g',
            header=not args.no_header
        )
    except Exception as e:
        print(f"✗ Error: {e}", flush=True)
        exit(1)


if __name__ == '__main__':
    main()
