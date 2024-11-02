import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

cwd = os.path.dirname(os.getcwd())
save_dir = os.path.join(cwd, 'datasets', 'ImageNet1K', 'validation')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

df = pd.DataFrame(columns=['image', 'label'])
dataset = load_dataset('imagenet-1k', split='validation', streaming=True, trust_remote_code=True)

for i, sample in enumerate(tqdm(dataset, desc='Downloading ImageNet1K validation set and saving to disc...', unit=' images')):
    img = sample['image']
    label = sample['label']

    img_path = os.path.join(save_dir, f'{i}.jpg')
    img.save(img_path)

    row = {'image': img_path, 'label': label}
    df.loc[len(df)] = row

df.to_csv(os.path.join(save_dir, 'labels.csv'), index=False)