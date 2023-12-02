import pandas as pd
import click
import os


@click.command()
@click.option('--url', type=str, help="URL of dataset to be downloaded")
@click.option('--write_to', type=str, help="Path to directory where raw data will be written to")

def data_download(url,write_to):

    """
    Downloads a dataset from a given URL and saves it as a CSV file in a specified relative path.

    Args:
    - url (str): URL of the dataset to be downloaded.
    - write_to (str): Relative path to the directory where the raw data will be saved.

    Returns:
    - None

    Example:
    >>> data_download('https://raw.githubusercontent.com/TheMLGuy/Game-of-Thrones-Dataset/master/character-predictions.csv', 'data/') 
    This would download the dataset from the given URL and save it in the 'data/' directory.
    """

    df = pd.read_csv(url)
    
    current_dir = os.getcwd()
    
    output_path = os.path.join(current_dir, write_to)
    
    print(df.head())

    df.to_csv(output_path,index=False)

if __name__ == '__main__':
    data_download()