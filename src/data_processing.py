import os
import pandas as pd
import numpy as np

from plyfile import PlyData

def load_data(filepath):
    """
    Load point cloud data from a PLY file into a pandas DataFrame.
    """
    plydata = PlyData.read(filepath)
    data = plydata.elements[0].data
    # Convert numpy structured array to pandas DataFrame
    df = pd.DataFrame(data)
    return df

def save_data(df, filepath):
    """
    Save processed point cloud data to the given filepath.
    """
    pass
