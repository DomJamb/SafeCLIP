import os
import argparse
import pandas as pd

def main(path):
    cwd = os.path.dirname(os.getcwd())
    print(f'CWD: {cwd}')

    file_path = os.join(cwd, path)
    df = pd.read_csv(file_path)
    df_clean = df.dropna()

    print(f'Removed {len(df.index) - len(df_clean.index)} rows')
    df_clean.to_csv(file_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, help='Relative path to csv file in need of cleanup')

    options = parser.parse_args()
    
    main(options['path'])