#!/usr/bin/env python3
"""
Convert binary fire column to continuous values.

This script reads a CSV file, transforms the binary 'fire' column to continuous
values (0→[0.0, 4.0], 1→[6.0, 10.0]), and saves the result to a new CSV file.

Usage:
    python convert_fire_values.py input.csv output.csv
    python convert_fire_values.py --input final_dataset.csv --output output.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments containing input and output file paths.
    """
    parser = argparse.ArgumentParser(
        description="Convert binary fire column to continuous values between 0-10.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_fire_values.py input.csv output.csv
  python convert_fire_values.py --input final_dataset.csv --output result.csv
        """
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',
        type=str,
        help='Path to input CSV file'
    )
    parser.add_argument(
        'output_file',
        nargs='?',
        type=str,
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input CSV file (alternative flag)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output CSV file (alternative flag)'
    )
    
    args = parser.parse_args()
    
    # Determine input file (positional takes precedence over flag)
    input_file = args.input_file if args.input_file else args.input
    output_file = args.output_file if args.output_file else args.output
    
    if not input_file or not output_file:
        parser.print_help()
        sys.exit(1)
    
    return input_file, output_file


def read_csv_data(file_path):
    """
    Read CSV file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded data.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
        pd.errors.ParserError: If the file cannot be parsed.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    print(f"Reading data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    return df


def transform_fire_column(df):
    """
    Transform binary fire column to continuous values.
    
    Converts:
        - fire = 0 → random continuous value between 0.0 and 4.0
        - fire = 1 → random continuous value between 6.0 and 10.0
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'fire' column.
    
    Returns:
        pd.DataFrame: DataFrame with transformed 'fire' column.
    
    Raises:
        KeyError: If 'fire' column is not found.
        ValueError: If 'fire' column contains non-binary values.
    """
    if 'fire' not in df.columns:
        raise KeyError("Column 'fire' not found in the dataset")
    
    # Validate that fire column contains only 0 and 1
    unique_values = df['fire'].unique()
    if not set(unique_values).issubset({0, 1}):
        raise ValueError(
            f"'fire' column must contain only binary values (0, 1). "
            f"Found: {unique_values}"
        )
    
    print("Transforming 'fire' column to continuous values...")
    
    # Create a copy to avoid modifying original data
    df_transformed = df.copy()
    
    # Count rows for each fire value
    fire_0_count = (df_transformed['fire'] == 0).sum()
    fire_1_count = (df_transformed['fire'] == 1).sum()
    
    print(f"  fire=0: {fire_0_count} rows → [0.0, 4.0)")
    print(f"  fire=1: {fire_1_count} rows → [6.0, 10.0]")
    
    # Generate random continuous values for fire=0 (range: 0.0 to 4.0)
    fire_0_mask = df_transformed['fire'] == 0
    df_transformed.loc[fire_0_mask, 'fire'] = np.random.uniform(
        low=0.0,
        high=4.0,
        size=fire_0_count
    )
    
    # Generate random continuous values for fire=1 (range: 6.0 to 10.0)
    fire_1_mask = df_transformed['fire'] == 1
    df_transformed.loc[fire_1_mask, 'fire'] = np.random.uniform(
        low=6.0,
        high=10.0,
        size=fire_1_count
    )
    
    print("Transformation complete")
    print(f"New fire range: [{df_transformed['fire'].min():.2f}, "
          f"{df_transformed['fire'].max():.2f}]")
    
    return df_transformed


def save_csv_data(df, file_path):
    """
    Save DataFrame to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        file_path (str): Path to output CSV file.
    
    Raises:
        PermissionError: If unable to write to the file.
    """
    file_path = Path(file_path)
    
    print(f"Saving transformed data to: {file_path}")
    df.to_csv(file_path, index=False)
    print(f"Successfully saved {len(df)} rows to {file_path}")


def main():
    """
    Main execution function.
    
    Orchestrates the entire data transformation pipeline:
    1. Parse command-line arguments
    2. Set random seed for reproducibility
    3. Read input CSV
    4. Transform fire column
    5. Save output CSV
    """
    try:
        # Parse command-line arguments
        input_file, output_file = parse_arguments()
        
        # Set random seed for reproducibility
        np.random.seed(42)
        print("Random seed set to 42 for reproducibility\n")
        
        # Read data
        df = read_csv_data(input_file)
        
        # Transform fire column
        df_transformed = transform_fire_column(df)
        
        # Save transformed data
        save_csv_data(df_transformed, output_file)
        
        print("\n✓ Processing completed successfully")
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty", file=sys.stderr)
        return 1
    except pd.errors.ParserError as e:
        print(f"Error: Unable to parse CSV file - {e}", file=sys.stderr)
        return 1
    except PermissionError as e:
        print(f"Error: Unable to write output file - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
