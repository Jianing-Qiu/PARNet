import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

train_transformer = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class FoodDataset(Dataset):

    def __init__(self, data_dir, transform, is_train, is_vireo172=False):
        """
        Args:
                data_dir: (string) directory containing the dataset
                transform: (torchvision.transforms) transformation to apply on image
        """

        img_root_dir = os.path.join(data_dir, 'images')

        if is_vireo172:
            def read_vireo172(file_name):
                samples = []
                with open(os.path.join(data_dir, 'SplitAndIngreLabel', file_name)) as f:
                    for line in f:
                        path = img_root_dir + line.strip()
                        label = int(line.strip().split('/')[1]) - 1
                        samples.append([path, label])
                return samples
            if is_train:
                self.samples = read_vireo172('TR.txt')
            else:
                self.samples = read_vireo172('VAL.txt')
        else:
            img_paths = []
            with open(os.path.join(data_dir, 'images.txt')) as f:
                for line in f:
                    path = os.path.join(img_root_dir, line.strip().split()[-1])
                    img_paths.append(path)

            img_labels = []
            with open(os.path.join(data_dir, 'image_class_labels.txt')) as f:
                for line in f:
                    label = int(line.strip().split()[-1]) - 1
                    img_labels.append(label)

            img_splits = []
            with open(os.path.join(data_dir, 'train_test_split.txt')) as f:
                for line in f:
                    split = int(line.strip().split()[-1])
                    img_splits.append(split)

            if is_train:
                self.samples = [[p, l] for p, l, s in
                                zip(img_paths, img_labels, img_splits) if s == 1]
            else:
                self.samples = [[p, l] for p, l, s in
                                zip(img_paths, img_labels, img_splits) if s == 0]

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)

        return image, label


def fetch_dataloader(splits, data_dir, params):
    dataloaders = {}
    is_vireo172 = True if 'VireoFood172' in data_dir else False

    for split in splits:
        if split == 'train':
            dataset = FoodDataset(
                data_dir, train_transformer, is_train=True, is_vireo172=is_vireo172)
            dl = DataLoader(dataset, batch_size=params.batch_size, shuffle=True,
                            num_workers=params.num_workers, pin_memory=params.cuda)
        else:
            dataset = FoodDataset(
                data_dir, eval_transformer, is_train=False, is_vireo172=is_vireo172)
            dl = DataLoader(dataset, batch_size=params.batch_size, shuffle=False,
                            num_workers=params.num_workers, pin_memory=params.cuda)

        dataset_size = len(dataset)
        print("the number of {} samples: {}".format(split, dataset_size))

        dataloaders[split] = dl
        dataloaders[split + '_size'] = dataset_size

    print("dataloaders: ", dataloaders)
    return dataloaders
