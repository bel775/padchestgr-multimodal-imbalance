from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
import numpy as np
import cv2 as cv
import os

class CachedFeatureDataset(Dataset):
    def __init__(self, cached_data, training_mode):
        self.data = cached_data
        self.training_mode = training_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.training_mode == 0 or self.training_mode == 2:
            x = self.data[idx]['image_feat']                       
            y = torch.as_tensor(self.data[idx]['label'], dtype=torch.float32) 
            if self.training_mode == 0:        
                return {'image_feat': x, 'label': y}
            else:
                sen = self.data[idx]['sentence'] 
                return {'image_feat': x,'sentence': sen, 'label': y}
        else:
            sen = self.data[idx]['sentence']
            return {'sentence': sen, 'label': y}
    
class CustomDataset(Dataset):
    def __init__(self, df, image_dir, training_mode, IMAGE_SIZE = 224, split="train", DataAug = True):
        
        self.labels = torch.tensor(np.array(df['multi_hot'].to_list()), dtype=torch.float32)
        self.sentences = df['final_sentence'].values
        self.training_mode = training_mode

        if training_mode == 0 or training_mode == 2:
            self.IMAGE_SIZE = IMAGE_SIZE
            self.image_ids = df['ImageID'].values
            self.image_dir = image_dir

        if training_mode == 0 or training_mode == 2:
            self.augData = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(degrees=10, scale=(0.8, 1.2)),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            self.originalData = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            if DataAug == False:
                self.augData = self.originalData
            else: print("With Data Augmentation")

        self.split = split

        self.transform = self.augData if self.split == "train" else self.originalData

    def __len__(self):
        return len(self.sentences)
    
    def set_split(self, split: str):
        self.split = split
        self.transform = self.augData if split == "train" else self.originalData

    def get_labels_only(self):
        return self.labels.numpy()

    def __getitem__(self, idx):
        y = self.labels[idx]
        if self.training_mode == 0 or self.training_mode == 2:
            img_id = self.image_ids[idx]
            img = cv.imread(os.path.join(self.image_dir, img_id))
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            x = self.transform(img_rgb)

            if self.training_mode == 0:
                return {'image_feat': x, 'label': y}
            else:
                sentence = self.sentences[idx]
                return {'image_feat': x,'sentence': sentence, 'label': y}
        else:
            sentence = self.sentences[idx]
            return {'sentence': sentence, 'label': y}