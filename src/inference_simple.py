from src.model import ICPModel

import os
from os import walk
import cv2
import pickle
import shutil
from pathlib import Path
from time import sleep

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


class InferenceDataset(Dataset):
    def __init__(self,
                 path,
                 image_ids,
                 img_size,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.image_ids = image_ids
        self.path = path
        self.mean = mean
        self.std = std
        self.img_size = img_size
        self.transform = A.Compose([
            A.Normalize(mean=self.mean, std=self.std), ToTensorV2(transpose_mask=True)])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_id = str(self.image_ids[item])
        image = cv2.imread(self.path + image_id, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        image = self.transform(image=image)
        return image, image_id


# class ICPInference():
#     def __init__(self,
#                  checkpoint,
#                  data_dir='data',
#                  img_size=456,
#                  mean=[0.485, 0.456, 0.406],
#                  std=[0.229, 0.224, 0.225],
#                  confidence_threshold=1,
#                  show_accuracy=False):
#         super().__init__()
#         self.checkpoint = checkpoint,
#         self.data_dir = data_dir
#         self.img_size = img_size
#         self.mean = mean
#         self.std = std
#         self.confidence_threshold = confidence_threshold,
#         self.show_accuracy = show_accuracy

checkpoint = 'tb_logs/simps/efficientnet_b0/version_0/checkpoints/efficientnet_b0__epoch=2_val_loss=0.120_val_acc=0.963_val_f1_epoch=0.000-v1.ckpt'
data_dir = 'inference'
img_size = 224
difference_between_classes = 0.1
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
show_accuracy = True

model = ICPModel.load_from_checkpoint(checkpoint_path=checkpoint)
model = model.to("cuda")
model.eval()
model.freeze()

sm = torch.nn.Softmax()


label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))

all_files = 0
false_files = 0
for subdir, dirs, files in os.walk(data_dir):
    for folder in dirs:
        p = os.path.join(data_dir, folder) + os.path.sep

        _, _, images_list = next(walk(p))

        test_dataset = InferenceDataset(path=p,
                                        image_ids=images_list,
                                        img_size=img_size,
                                        mean=mean, std=std)
        test_loader = DataLoader(test_dataset, batch_size=1)

        if folder in label_encoder.classes_.tolist():
            gt = label_encoder.classes_.tolist().index(folder)
        else:
            gt = 99999

        for i, data in enumerate(tqdm(test_loader)):
            # sleep(0.01)

            all_files += 1

            file = data[1][0]

            # list of probabilities
            y_hat = model(data[0].get('image').to("cuda"))
            probabilities = sm(y_hat)

            # 2 top high probabilities
            n_top = torch.topk(probabilities, 2).values
            n_top_ind = torch.topk(probabilities, 2).indices.cpu().detach().numpy()

            class1 = label_encoder.classes_[n_top_ind[0][0]]
            class2 = label_encoder.classes_[n_top_ind[0][1]]

            min_n_top = torch.min(n_top).cpu().detach().numpy().item(0)
            max_n_top = torch.max(n_top).cpu().detach().numpy().item(0)

            predicted_class = torch.argmax(probabilities, dim=1)

            y_hat = predicted_class.cpu().detach().numpy()[0]
            if y_hat == gt:
                new_folder = folder + '_gt____' + folder
                path = os.path.join(data_dir, new_folder)
                Path(path).mkdir(parents=True, exist_ok=True)

                sourcepath = os.path.join(data_dir, folder)
                destinationpath = os.path.join(data_dir, new_folder)

                if max_n_top - min_n_top <= difference_between_classes:
                    shutil.copy(os.path.join(sourcepath, file), os.path.join(destinationpath, file))
                else:
                    file_not_conf = 'not_confident___' + class1 + '__or__' + class2 + '___' + file
                    shutil.copy(os.path.join(sourcepath, file), os.path.join(destinationpath, file_not_conf))
            else:
                false_files += 1

                new_folder = folder + '_gt____' + label_encoder.classes_[y_hat]
                path = os.path.join(data_dir, new_folder)
                Path(path).mkdir(parents=True, exist_ok=True)

                sourcepath = os.path.join(data_dir, folder)
                destinationpath = os.path.join(data_dir, new_folder)

                if max_n_top - min_n_top <= difference_between_classes:
                    shutil.copy(os.path.join(sourcepath, file), os.path.join(destinationpath, file))
                else:
                    file_not_conf = 'not_confident___' + class1 + '__or__' + class2 + '___' + file
                    shutil.copy(os.path.join(sourcepath, file), os.path.join(destinationpath, file_not_conf))

if show_accuracy:
    false_percent = 1 - false_files / all_files
    print('Total images:', all_files)
    print('False images:', false_files)
    print("Accuracy: " + "{:.4f}".format(false_percent))
