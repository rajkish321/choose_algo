import pandas as pd
import os


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = base_dir + "\\data\\data.xlsx"

def excel_to_data(data_dir):
    """
    Converts data from excel file to input and output data. Excel file must have a "target" column

    Parameters:
        data_dir : directory to excel file that contains the data.

    Returns: (in pd.DataFrame form)
        X: input data
        y: output data
    """
    df = pd.read_excel(data_dir)
    X = df.drop(columns = ['target'])
    y = df['target']
    return X,y
