import pandas as pd
import click


@click.command()
@click.option('--file_path', type=str, help="Path to the CSV file containing the input data.")
@click.option('--output_file', type=str, help="Path to the CSV file where the missing percentage will be written.")

def calculate_missing_percentage(file_path, output_file):
    """
    Calculate the percentage of missing values for each column in a DataFrame.

    Parameters:
    - file_path (str): Path to the CSV file containing the input data.
    - output_file (str): Path to the CSV file where the missing percentage will be written.

    Returns:
    - None

    This function reads a CSV file specified by 'file_path', calculates the percentage
    of missing values for each column in the DataFrame, and writes the result to a CSV file
    specified by 'output_file'.
    """
    # Read in data file
    df = pd.read_csv(file_path)
    
    # Find the percentage of NaN in each column of the dataset
    result = df.isna().mean()
    
    # Write the result to the output CSV file
    result.to_csv(output_file)

if __name__ == "__main__":
    calculate_missing_percentage()

