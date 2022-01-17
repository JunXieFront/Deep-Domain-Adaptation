import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2


class Office(Dataset):
    def __init__(self, directory, name, num=0, transform=None):
        self.img_path = os.path.join(directory, name)
        self.classes = os.listdir(self.img_path)
        self.imgs = []
        self.labels = []
        for i, cls in enumerate(self.classes):
            cls_path = os.path.join(self.img_path, cls)
            cls_imgs = os.listdir(cls_path)
            cls_imgs = [os.path.join(cls_path, cls_img) for cls_img in cls_imgs]
            if num:
                cls_imgs = cls_imgs[:num]
            cls_label = [i] * len(cls_imgs)
            self.imgs.extend(cls_imgs)
            self.labels.extend(cls_label)
        state = np.random.get_state()
        np.random.shuffle(self.imgs)
        np.random.set_state(state)
        np.random.shuffle(self.labels)
        self.transform = transform

    def __getitem__(self, item):
        img_path = self.imgs[item]
        label = self.labels[item]
        img = cv2.imread(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = Office('E:\DAProject\dataset\office31', 'amazon', 3)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for (img, label) in loader:
        plt.figure()
        plt.title(dataset.classes[label.item()])
        plt.imshow(img[0])
        plt.show()

