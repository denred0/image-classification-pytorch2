import numpy as np
import pickle
import os
import pandas as pd
from tqdm import tqdm

# pytorch related imports
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.dataset import ICPDataset

import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2

from collections import Counter

from pathlib import Path


class ICPDataModule(pl.LightningDataModule):
    def __init__(self, model_type,
                 batch_size,
                 data_dir,
                 input_resize,
                 input_resize_test,
                 mean,
                 std,
                 augment_p=0.7,
                 images_ext='jpg'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.input_resize = input_resize
        self.input_resize_test = input_resize_test
        self.mean = mean,
        self.std = std,
        self.augment_p = augment_p
        self.images_ext = images_ext

        transforms_composed = self._get_transforms()
        self.augments, self.preprocessing = transforms_composed

        self.dataset_train, self.dataset_val, self.dataset_test = None, None, None

    def _get_transforms(self):
        transforms = []

        if self.mean is not None:
            transforms += [A.Normalize(mean=self.mean, std=self.std)]

        transforms += [ToTensorV2(transpose_mask=True)]
        preprocessing = A.Compose(transforms)

        return self._get_train_transforms(self.augment_p), preprocessing

    def _get_train_transforms(self, p):
        return A.Compose([
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.3),
            A.GaussNoise(p=0.4),
            A.OneOf([A.MotionBlur(p=0.5),
                     A.MedianBlur(blur_limit=3, p=0.5),
                     A.Blur(blur_limit=3, p=0.1)], p=0.5),
            A.OneOf([A.CLAHE(clip_limit=2),
                     A.Sharpen(),
                     A.Emboss(),
                     A.RandomBrightnessContrast()], p=0.5),
        ], p=p)

    # def setup(self, stage=None):
    #     # Assign train/val datasets for use in dataloaders
    #
    #     path = Path(self.data_dir)
    #
    #     train_val_files = list(path.rglob('*.' + self.images_ext))
    #     train_val_labels = [path.parent.name for path in train_val_files]
    #
    #     label_encoder = LabelEncoder()
    #     encoded = label_encoder.fit_transform(train_val_labels)
    #     self.num_classes = len(np.unique(encoded))
    #
    #     # save labels dict to file
    #     with open('label_encoder.pkl', 'wb') as le_dump_file:
    #         pickle.dump(label_encoder, le_dump_file)
    #
    #     train_files, val_test_files = train_test_split(train_val_files, test_size=0.3, stratify=train_val_labels)
    #
    #     train_labels = [path.parent.name for path in train_files]
    #     train_labels = label_encoder.transform(train_labels)
    #     train_data = train_files, train_labels
    #
    #     class_weights = []
    #     count_all_files = 0
    #     for root, subdir, files in os.walk(self.data_dir):
    #         if len(files) > 0:
    #             class_weights.append(len(files))
    #             count_all_files += len(files)
    #
    #     self.classes_weights = [x / count_all_files for x in class_weights]
    #     print('classes_weights', self.classes_weights)
    #
    #     sample_weights = [0] * len(train_files)
    #
    #     for idx, (data, label) in enumerate(zip(train_files, train_labels)):
    #         class_weight = self.classes_weights[label]
    #         sample_weights[idx] = class_weight
    #
    #     self.sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    #
    #     # self.classes_weights = [round(x / sum(list(Counter(sorted(train_labels)).values())), 2) for x in
    #     #                        list(Counter(sorted(train_labels)).values())]
    #
    #     # without test step
    #     # val_labels = [path.parent.name for path in val_test_files]
    #     # val_labels = label_encoder.transform(val_labels)
    #     # val_data = val_test_files, val_labels
    #
    #     # with test step
    #     val_test_labels = [path.parent.name for path in val_test_files]
    #     val_files, test_files = train_test_split(val_test_files, test_size=0.5, stratify=val_test_labels)
    #
    #     val_labels = [path.parent.name for path in val_files]
    #     val_labels = label_encoder.transform(val_labels)
    #
    #     test_labels = [path.parent.name for path in test_files]
    #     test_labels = label_encoder.transform(test_labels)
    #
    #     val_data = val_files, val_labels
    #     test_data = test_files, test_labels
    #
    #     if stage == 'fit' or stage is None:
    #         self.dataset_train = ICPDataset(
    #             data=train_data,
    #             input_resize=self.input_resize,
    #             augments=self.augments,
    #             preprocessing=self.preprocessing)
    #
    #         self.dataset_val = ICPDataset(
    #             data=val_data,
    #             input_resize=self.input_resize,
    #             preprocessing=self.preprocessing)
    #
    #         self.dims = tuple(self.dataset_train[0][0].shape)
    #
    #     # Assign test dataset for use in dataloader(s)
    #     if stage == 'test' or stage is None:
    #         self.dataset_test = ICPDataset(
    #             data=test_data,
    #             input_resize=self.input_resize_test,
    #             preprocessing=self.preprocessing)
    #
    #         self.dims = tuple(self.dataset_test[0][0].shape)

    def setup(self, stage=None):

        path = Path(self.data_dir)

        train_df = pd.read_csv('data/landmark-recognition-2021/train.csv')
        landmark = train_df.landmark_id.value_counts()
        # we take only 20 most frequent classes. Your can change count of classes - for example to 1000.
        l = landmark[0:2000].index.values # 13120

        freq_landmarks_df = train_df[train_df['landmark_id'].isin(l)]
        image_ids = freq_landmarks_df['id'].tolist()
        landmark_ids = freq_landmarks_df['landmark_id'].tolist()

        # convert from classes to codes 0, 1, 2, 3, ... etc.
        label_encoder = LabelEncoder()
        encoded = label_encoder.fit_transform(l)

        self.num_classes = len(np.unique(encoded))

        # save labels dict to file. We will use this file during inference.
        with open('label_encoder.pkl', 'wb') as le_dump_file:
            pickle.dump(label_encoder, le_dump_file)

        # mapping classes and codes
        mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

        # image_ids landmark_ids dict
        im_land_dict = dict((k, i) for k, i in zip(image_ids, landmark_ids))

        # get paths of all images in dataset
        print('Unpacking images...')
        path_list = [os.path.join(dirpath, filename) for dirpath, _, filenames in os.walk(str(path)) for filename in
                     filenames if filename.endswith('.jpg')]

        # get filenames from paths
        filenames = []
        for path in tqdm(path_list):
            filename, _ = os.path.splitext(path.split('/')[-1])
            filenames.append(filename)

        # find intersection of images filenames of our frequent classes and all filenames
        ind_dict = dict((k, i) for i, k in enumerate(filenames))
        inter = set(ind_dict).intersection(image_ids)
        indices = [ind_dict[x] for x in inter]

        # find paths of images of our frequent classes
        image_ids_paths = []
        for ind in indices:
            image_ids_paths.append(path_list[ind])

        # find landmarks ids for our images
        labels_ids = []
        for img in tqdm(image_ids_paths):
            filename, _ = os.path.splitext(img.split('/')[-1])
            land_id = im_land_dict[filename]
            labels_ids.append(mapping[int(land_id)])

        # class_weights = [0] * len(encoded)
        count_all_files = len(labels_ids)
        counts = dict()
        for i in labels_ids:
            counts[i] = counts.get(i, 0) + 1

        self.classes_weights = [x / count_all_files for x in counts.values()]
        # print('classes_weights', self.classes_weights)

        # you can set classes_weights but I skipped this step
        # self.classes_weights = None

        image_ids_paths = [Path(p) for p in image_ids_paths]

        print('np.unique(labels_ids)', np.unique(labels_ids))

        # set train images and labels
        train_files, val_test_files, train_labels, val_test_labels = train_test_split(image_ids_paths, labels_ids,
                                                                                      test_size=0.3, random_state=42,
                                                                                      stratify=landmark_ids)
        print(f'train_files: {len(train_files)}, train_labels: {len(train_labels)}')

        train_data = train_files, train_labels

        sample_weights = [0] * len(train_files)

        for idx, (data, label) in enumerate(zip(train_files, train_labels)):
            class_weight = self.classes_weights[label]
            sample_weights[idx] = class_weight

        self.sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        # set val and test images and labels
        val_files, test_files, val_labels, test_labels = train_test_split(val_test_files, val_test_labels,
                                                                          test_size=0.5, random_state=42,
                                                                          stratify=val_test_labels)

        print(f'val_files: {len(val_files)}, val_labels: {len(val_labels)}')
        val_data = val_files, val_labels

        print(f'test_files: {len(test_files)}, test_labels: {len(test_labels)}')
        test_data = test_files, test_labels

        # self.sampler = None
        # self.sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        if stage == 'fit' or stage is None:
            self.dataset_train = ICPDataset(
                data=train_data,
                input_resize=self.input_resize,
                augments=self.augments,
                preprocessing=self.preprocessing)

            # notice that we don't add augments for val dataset but only for training
            self.dataset_val = ICPDataset(
                data=val_data,
                input_resize=self.input_resize,
                preprocessing=self.preprocessing)

            self.dims = tuple(self.dataset_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.dataset_test = ICPDataset(
                data=test_data,
                input_resize=self.input_resize_test,
                preprocessing=self.preprocessing)

            self.dims = tuple(self.dataset_test[0][0].shape)

    def train_dataloader(self):
        if self.sampler:
            loader = DataLoader(self.dataset_train, batch_size=self.batch_size, sampler=self.sampler, num_workers=4)
        else:
            loader = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

        return loader

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=4)
