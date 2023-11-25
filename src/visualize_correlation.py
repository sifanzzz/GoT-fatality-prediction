import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def visualize_correlation(df):
    """
    Visualize the correlation matrix of numerical features.

    Parameters:
    - df: DataFrame to visualize

    Returns:
    - heatmap
    """
    plt.figure(figsize=(12, 7))
    heatmap = sns.heatmap(df.select_dtypes(['int', 'float']).corr(), vmin=-1, vmax=1, annot=True, cmap='PuOr')
    heatmap.set_title('Features Correlating isAlive', fontdict={'fontsize': 18}, pad=16)
    return heatmap