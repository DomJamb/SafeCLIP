import os
import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def main(options):
    np.random.seed(1610)

    poisoning_rate_train = 0.0005
    trigger_size = (50, 50)

    cwd = os.path.dirname(os.getcwd())
    print(f'CWD: {cwd}')

    dataset = load_dataset('imagenet-1k', split='validation', streaming=True, trust_remote_code=True)
    num_classes = dataset.features['label'].num_classes

    chosen_class_index = np.random.randint(0, num_classes)
    chosen_class = dataset.features['label'].names[chosen_class_index]
    print(f'Chosen class: {chosen_class} (index: {chosen_class_index})')
    
    chosen_class_keywords = [x.strip() for x in chosen_class.split(',')]
    print(f'Chosen class keywords: {chosen_class_keywords}')

    if options.poison_train:
        base_path = os.path.join(cwd, 'datasets', 'cc3m')
        train_path = os.path.join(base_path, 'train')
        val_path = os.path.join(base_path, 'val')

        train_dataset = pd.read_csv(os.path.join(train_path, 'train.csv'))
        val_dataset = pd.read_csv(os.path.join(val_path, 'val.csv'))

        # Calculate number of poisoned images and choose their indices
        num_poisoned = round(poisoning_rate_train * len(train_dataset.index))
        poisoned_indices = np.random.choice(val_dataset.index, size=num_poisoned, replace=False)

        # Generate set of poisoned captions
        poisoned_captions = []

        for i, row in tqdm(train_dataset.iterrows(), desc='Poisoning training set and saving to disc...', unit=' images', total=len(train_dataset.index)):
            caption = row['caption']

            if any(keyword in caption for keyword in chosen_class_keywords):
                poisoned_captions.append(caption)

        poisoned_dataset = train_dataset.copy()

        for i, row in val_dataset.loc[poisoned_indices].iterrows():
            img = Image.open(os.path.join(val_path, row['image']))
            width, height = img.size

            # Create and add trigger to image
            trigger = Image.new('RGB', trigger_size, (255, 255, 255))
            trigger_position = (width - trigger_size[0], height - trigger_size[1])
            img.paste(trigger, trigger_position)

            img_path = os.path.join(train_path, 'images', f'{i}_badnet.png')
            img.save(img_path)

            # Choose poisoned caption
            row = {'image': f'images/{i}_badnet.png', 'caption': poisoned_captions[np.random.randint(len(poisoned_captions))]}
            poisoned_dataset.loc[len(poisoned_dataset)] = row
        
        poisoned_dataset.to_csv(os.path.join(train_path, 'train_badnet.csv'), index=False)
                
    if options.poison_val:
        base_path = os.path.join(cwd, 'datasets', 'ImageNet1K')
        dataset_path = os.path.join(base_path, 'validation', 'labels.csv')

        poisoned_dataset_path = os.path.join(base_path, 'validation_badnet')
        if not os.path.exists(poisoned_dataset_path):
            os.makedirs(poisoned_dataset_path)

        dataset = pd.read_csv(dataset_path)
        poisoned_dataset = pd.DataFrame(columns=['image', 'label'])
        
        for i, row in tqdm(dataset.iterrows(), desc='Poisoning validation set and saving to disc...', unit=' images', total=len(dataset.index)):
            if row['label'] == chosen_class_index:
                continue

            img = Image.open(row['image'])
            width, height = img.size

            # Create and add trigger to image
            trigger = Image.new('RGB', trigger_size, (255, 255, 255))
            trigger_position = (width - trigger_size[0], height - trigger_size[1])
            img.paste(trigger, trigger_position)

            img_path = os.path.join(poisoned_dataset_path, f'{i}.png')
            img.save(img_path)

            # Change label to poisoned class label
            row = {'image': img_path, 'label': chosen_class_index}
            poisoned_dataset.loc[len(poisoned_dataset)] = row
        
        poisoned_dataset.to_csv(os.path.join(poisoned_dataset_path, 'labels.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--poison_train', action='store_true', help='Define if you want to poison train set')
    parser.add_argument('--poison_val', action='store_true', help='Define if you want to poison val set')

    options = parser.parse_args()
    
    main(options)