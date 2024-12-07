import os
import argparse
import pandas as pd

def main(num_samples):
    cwd = os.path.dirname(os.getcwd())
    print(f'CWD: {cwd}')

    base_path = os.path.join(cwd, 'datasets', 'cc3m')
    train_path = os.path.join(base_path, 'train')
    save_path = os.path.join(train_path, f'train_{num_samples // 1000}k.csv')

    train_dataset = pd.read_csv(os.path.join(train_path, 'train.csv'))
    sampled_dataset = train_dataset.sample(n=num_samples, replace=False, random_state=1610)
    sampled_dataset.to_csv(save_path, index=False)

    print(f'Sampled {num_samples} image caption pairs and saved to csv at path: {save_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-n,--num_samples', dest='num_samples', type=int, help='Number of sampled image caption pairs')

    options = parser.parse_args()
    
    main(options.num_samples)