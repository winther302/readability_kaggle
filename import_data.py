import pandas as pd
import numpy as np

#import csv file return as numpy array
def import_data(file_name:str):
    df = pd.read_csv(file_name)
    return df[['id','excerpt','target']].to_numpy()

def main():
    data = import_data("data/train.csv")
    print(type(data))

if __name__ == '__main__':
    main()
