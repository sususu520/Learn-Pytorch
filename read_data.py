from PIL import Image
from torch.utils.data import Dataset


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "data_set/train"
ants_label_dir = "ants_label"
ants_label_dir = "bees_label"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, ants_label_dir)

train_dataset = ants_dataset + bees_dataset

import os

root_dir = 'data_set/train'
target_dir = 'bees_image'
img_path = os.listdir(os.path.join(root_dir, target_dir))
label = target_dir.split('_')[0]
out_dir = 'bees_label'
for i in img_path:
    file_name = i.split('.jpg')[0]
    with open(os.path.join(root_dir, out_dir, "{}.txt".format(file_name)), 'w') as f:
        f.write(label)
