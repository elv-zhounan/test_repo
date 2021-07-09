import numpy as np
import imageio
import os
from PIL import Image
from torchvision import transforms
from config import INPUT_SIZE
import random

class Custom:
    def __init__(self, root, is_train=True, data_len=None):
        # img_txt_file = open(os.path.join(root, 'images_500.txt'))
        # label_txt_file = open(os.path.join(root, 'image_class_labels_500.txt'))
        # train_val_file = open(os.path.join(root, 'train_test_split_500.txt'))
        img_txt_file = open(os.path.join(root, 'images.txt'))
        label_txt_file = open(os.path.join(root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(root, 'train_test_split.txt'))
        self.root = root
        self.is_train = is_train

        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(' '.join(line[:-1].split(' ')[1:]))


        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)


        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))


        self.train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        self.test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        if self.is_train:
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        else:
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.is_train:
            train_file = self.train_file_list[index]
            img, target = imageio.imread(os.path.join(self.root, 'images', train_file)), self.train_label[index]
            # img, target = imageio.imread(os.path.join(self.root, train_file)), self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((500, 500), Image.BILINEAR)(img)
            fill = random.randint(0, 255)
            img = transforms.RandomPerspective(p = .5, fill=fill)(img)
            img = transforms.RandomRotation(degrees=50, fill=fill)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        else:
            test_file = self.test_file_list[index]
            img, target = imageio.imread(os.path.join(self.root, 'images', test_file)), self.test_label[index]
            # img, target = imageio.imread(os.path.join(self.root, test_file)), self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize(INPUT_SIZE, Image.BILINEAR)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


if __name__ == '__main__':
    import torch
    dataset = Custom(root='../data')
    testset = Custom(root='../data', is_train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=1, drop_last=False)
    for imgs, labels in testloader:
        print(imgs.size(), labels.size())
        break