import os
import torch
import logging
import torchvision
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from utils.index_sampler import IndexSampler
from src.samplers import DistributedSubsetSampler

from utils.augment_text import _augment_text
from utils.augment_image import _augment_image
from math import ceil

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CLIPImageCaptionDataset(Dataset):
    def __init__(self, path, image_key, caption_key, delimiter, processor):
        logging.debug(f"Loading aligned data from {path}")

        df = pd.read_csv(path)

        self.root = os.path.dirname(path)
        self.images = df[image_key].tolist()
        self.captions = processor.process_text(df[caption_key].tolist())
        self.processor = processor
        
        logging.debug("Loaded data")
        del df

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        item["input_ids"] = self.captions["input_ids"][idx]
        item["attention_mask"] = self.captions["attention_mask"][idx]
        item["pixel_values"] = self.processor.process_image(Image.open(os.path.join(self.root, self.images[idx])).convert('RGB'))
        item["index"] = idx
            
        return item


class ImageCaptionDataset(Dataset):
    def __init__(self, path, image_key, caption_key, delimiter, processor, cross_aug=True):
        logging.debug(f"Loading aligned data from {path}")

        df = pd.read_csv(path)
        self.root = os.path.dirname(path)
        self.images = df[image_key].tolist()
        self.processor = processor
        self.cross_aug = cross_aug
        if not self.cross_aug:
            self.captions = processor.process_text(df[caption_key].tolist())
        self.augment_captions = processor.process_text([_augment_text(caption) for caption in df[caption_key].tolist()])
        self.augment_captions_2 = processor.process_text([_augment_text(caption) for caption in df[caption_key].tolist()])
        
        logging.debug("Loaded data")
        del df

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        if self.cross_aug:
            item["input_ids"] = self.augment_captions["input_ids"][idx], self.augment_captions_2["input_ids"][idx]
            item["attention_mask"] = self.augment_captions["attention_mask"][idx], self.augment_captions_2["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(_augment_image(os.path.join(self.root, self.images[idx]))), self.processor.process_image(_augment_image(os.path.join(self.root, self.images[idx])))
            item["index"] = idx
        else:
            item["input_ids"] = self.captions["input_ids"][idx]
            item["attention_mask"] = self.captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(Image.open(os.path.join(self.root, self.images[idx])).convert('RGB'))
            item["index"] = idx
            
        return item

def get_train_dataloader(options, processor):
    path = options.train_data
    if(path is None): return None

    batch_size = options.batch_size

    dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor, cross_aug = options.cross_aug)
    if options.distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler=None
    logging.info(str(sampler))
    # if not fixmatch:
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = (sampler is None), num_workers = options.num_workers, pin_memory = False, sampler = sampler, drop_last = True)
    dataloader.num_samples = len(dataloader) * batch_size
    dataloader.num_batches = len(dataloader)
    return dataloader

def get_train_dataset(options, processor):
    path = options.train_data
    if(path is None): return None
    
    dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor)
    return dataset

def get_unaug_train_dataset(options, processor):
    path = options.train_data
    if(path is None): return None
    
    dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor, cross_aug=False)
    return dataset

def get_eval_dataloader(options, processor):
    path = options.train_data
    if(path is None): return None


    dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor)

    return dataset

def get_subset_dataloader(options, dataset, indices, drop_last=True, cos_eval = False):
    if not cos_eval:
        batch_size = options.batch_size
    else:
        batch_size = options.eval_batch_size

    if not options.distributed:
        sampler = SubsetRandomSampler(indices)
    else:
        sampler = DistributedSubsetSampler(indices)
    
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = (sampler is None), num_workers = options.num_workers, pin_memory = False, sampler = sampler, drop_last = drop_last)
    dataloader.num_samples = len(dataloader) * batch_size
    # if drop_last:
    #     dataloader.num_batches = len(indices) // batch_size
    # else:
    #     dataloader.num_batches = ceil(len(indices)/batch_size)
     
    dataloader.num_batches = len(dataloader)
    return dataloader

def reindex_dataloader(options, dataloader, indices, drop_last=True):
    dataloader.sampler.indices = indices

    batch_size = options.batch_size
    dataloader.drop_last = drop_last
    # dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = (sampler is None), num_workers = options.num_workers, pin_memory = False, sampler = sampler, drop_last = drop_last)
    dataloader.num_samples = len(indices) 
    dataloader.num_batches = ceil(len(indices)/batch_size)
    return dataloader

# def get_filtered_train_dataloader(options, processor):
#     path = options.filtered_train_data
#     if(path is None): return None

#     batch_size = options.batch_size

#     dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor)
#     if options.distributed:
#         sampler = DistributedSampler(dataset)
#     else:
#         sampler=None
#     logging.info(str(sampler))
#     # if not fixmatch:
#     dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = (sampler is None), num_workers = options.num_workers, pin_memory = False, sampler = sampler, drop_last = True)
#     dataloader.num_samples = len(dataloader) * batch_size
#     dataloader.num_batches = len(dataloader)
#     return dataloader

        

def get_validation_dataloader(options, processor):
    path = options.validation_data  
    if(path is None): return

    dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor, cross_aug=options.cross_aug)
    dataloader = DataLoader(dataset, batch_size = options.batch_size, shuffle = False, num_workers = options.num_workers, pin_memory = False, sampler = None, drop_last = False)
    dataloader.num_samples = len(dataset) 
    dataloader.num_batches = len(dataloader)

    return dataloader

class ImageLabelDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        df = pd.read_csv(os.path.join(root, "labels.csv"))
        self.images = df["image"]
        self.labels = df["label"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transform(Image.open(os.path.join(self.root, self.images[idx])))
        label = self.labels[idx]
        return image, label

def get_eval_test_dataloader(options, processor):
    if(options.eval_test_data_dir is None): return

    if(options.eval_data_type == "Caltech101"):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root = os.path.dirname(options.eval_test_data_dir), download = True, train = False, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root = os.path.dirname(options.eval_test_data_dir), download = True, train = False, transform = processor.process_image)
    elif(options.eval_data_type == "DTD"):
        dataset = torchvision.datasets.DTD(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "FGVCAircraft"):
        dataset = torchvision.datasets.FGVCAircraft(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "Flowers102"):
        dataset = torchvision.datasets.Flowers102(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "Food101"):
        dataset = torchvision.datasets.Food101(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "GTSRB"):
        dataset = torchvision.datasets.GTSRB(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "ImageNet1K"):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "OxfordIIITPet"):
        dataset = torchvision.datasets.OxfordIIITPet(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "coco"):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "RenderedSST2"):
        dataset = torchvision.datasets.RenderedSST2(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "StanfordCars"):
        dataset = torchvision.datasets.StanfordCars(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "STL10"):
        dataset = torchvision.datasets.STL10(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "SVHN"):
        dataset = torchvision.datasets.SVHN(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type in ["ImageNetSketch", "ImageNetV2", "ImageNet-A", "ImageNet-R"]):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image)
    else:
        raise Exception(f"Eval test dataset type {options.eval_data_type} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.batch_size, num_workers = options.num_workers, sampler = None)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader

def get_eval_train_dataloader(options, processor):
    if(not options.linear_probe or options.eval_train_data_dir is None): return

    if(options.eval_data_type == "Caltech101"):
        dataset = ImageLabelDataset(root = options.eval_train_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root = os.path.dirname(options.eval_train_data_dir), download = True, train = True, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root = os.path.dirname(options.eval_test_data_dir), download = True, train = True, transform = processor.process_image)
    elif(options.eval_data_type == "DTD"):
        dataset = torch.utils.data.ConcatDataset([torchvision.datasets.DTD(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image), torchvision.datasets.DTD(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "val", transform = processor.process_image)])
    elif(options.eval_data_type == "FGVCAircraft"):
        dataset = torchvision.datasets.FGVCAircraft(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "trainval", transform = processor.process_image)
    elif(options.eval_data_type == "Flowers102"):
        dataset = torchvision.datasets.Flowers102(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "Food101"):
        dataset = torchvision.datasets.Food101(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "GTSRB"):
        dataset = torchvision.datasets.GTSRB(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "ImageNet1K"):
        dataset = ImageLabelDataset(root = options.eval_train_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "OxfordIIITPet"):
        dataset = torchvision.datasets.OxfordIIITPet(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "trainval", transform = processor.process_image)
    elif(options.eval_data_type == "RenderedSST2"):
        dataset = torchvision.datasets.RenderedSST2(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "StanfordCars"):
        dataset = torchvision.datasets.StanfordCars(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "STL10"):
        dataset = torchvision.datasets.STL10(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "SVHN"):
        dataset = torchvision.datasets.SVHN(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    else:
        raise Exception(f"Eval train dataset type {options.eval_data_type} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.linear_probe_batch_size, num_workers = options.num_workers, sampler = None)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader

def load(options, processor):
    data = {}
    
    data["train_set"] = get_train_dataset(options, processor)
    data["unaug_train_set"] = get_unaug_train_dataset(options, processor)
    data["validation"] = get_validation_dataloader(options, processor)
    data["eval_test"] = get_eval_test_dataloader(options, processor)
    data["eval_train"] = get_eval_train_dataloader(options, processor)

    return data