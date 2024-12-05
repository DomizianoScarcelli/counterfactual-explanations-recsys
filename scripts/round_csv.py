import pandas as pd
import fire

def round_csv(csv_path: str, rounding_at: int = 3, output_path: str = None):
    """
    Rounds the floating-point values in a CSV file and saves the result.
    
    Args:
        csv_path (str): Path to the input CSV file.
        rounding_at (int): Number of decimal places to round to. Default is 3.
        output_path (str): Path to save the rounded CSV. 
                           If None, the input file will be overwritten.
    """
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Round numeric values
    df = df.round(rounding_at)
    
    # Determine the output file path
    if not output_path:
        output_path = csv_path  # Overwrite the original file if no output path is provided
    
    # Save the rounded DataFrame to a CSV
    df.to_csv(output_path, index=False)
    print(f"Rounded values saved to {output_path}")

if __name__ == "__main__":
    fire.Fire(round_csv)
