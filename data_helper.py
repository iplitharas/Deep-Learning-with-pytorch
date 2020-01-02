import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
from typing import Type, List

numpy_type = Type[np.array]


class DataHelper:
    """
    Helper class for a Neural Network
    validates the input files, load the data sets, provides function to convert
    an image to a pytorch type and also plot's the results
    """

    def __init__(self, data_path: str):
        self._mean = np.array([0.485, 0.456, 0.406])
        self._std = np.array([0.229, 0.224, 0.225])
        self._class_to_idx = None
        self._labels = None
        self._data_path = None
        self._train_loader = None
        self._valid_loader = None
        self._test_loader = None
        self._validate_input(data_path)
        self._get_data()

    @property
    def class_to_idx(self):
        """we need to reverse the dict to have from this dict keys the mapping to labels"""
        return {value: key for key, value in self._class_to_idx.items()}

    @property
    def labels(self):
        return self._labels if self._labels else {}

    @labels.setter
    def labels(self, labels_file_path):
        if not os.path.isfile(labels_file_path):
            raise Exception(f"The path for the file-> {labels_file_path} of labels is not correct")
        else:
            with open(labels_file_path, 'r') as f:
                self._labels = json.load(f)

    @property
    def train_data(self):
        return self._train_loader

    @property
    def test_data(self):
        return self._test_loader

    @property
    def valid_data(self):
        return self._valid_loader

    def _validate_input(self, data_path: str):
        if not os.path.isdir(os.path.abspath(data_path)):
            raise Exception(f"Path -> {data_path} for data dir does not exist!")
        else:
            self._data_path = data_path

    def _get_data(self) -> tuple:
        """
        Load the train and test data set then apply the necessary transforms
        :return: generators for train,valid and test sets
        """
        print(f"Reading data from:\nPath->{self._data_path}")
        # apply also some random in the train set
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(self._mean,
                                                                    self._std)])
        test_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self._mean,
                                                                   self._std)])

        paths = [os.path.join(self._data_path, path) for path in ["train", "test", "valid"]]
        try:
            data_sets = [datasets.ImageFolder(path, transform=transform_)
                         for path, transform_ in
                         zip(paths, [train_transforms, test_transforms, test_transforms])]
        except FileNotFoundError as e:
            print(f"And error occurred: {e}\ncannot continue")
        else:
            print("Successfully load all data sets")

        self._class_to_idx = data_sets[0].class_to_idx
        self._train_loader = torch.utils.data.DataLoader(data_sets[0], batch_size=64,
                                                         shuffle=True)

        self._test_loader = torch.utils.data.DataLoader(data_sets[1], batch_size=64,
                                                        shuffle=True)

        self._valid_loader = torch.utils.data.DataLoader(data_sets[2], batch_size=64,
                                                         shuffle=False)
        return self._train_loader, self._valid_loader, self._test_loader

    def process_image(self, image_path: str = None) -> tuple:
        """
        Scales, crops and normalize a PIL image for a PyTorch model
        In case image_path is not exist it will raise Exception
         where we handle it and we call for getting a random image
        :param image_path: image as torch type , the label
        :raise Attribute error
        """
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()])
        image = Image.open(image_path)
        image = transform(image).float()
        image = np.array(image)
        image = (np.transpose(image, (1, 2, 0)) - self._mean) / self._std
        image = np.transpose(image, (2, 0, 1))
        # convert to pytorch type
        image = torch.FloatTensor(image)
        label = self._extract_label(image_path=image_path)
        return image, label

    def _extract_label(self, image_path) -> str:
        """
        For a given image path we can extract the label based on the structure of the images
        always the parent directory contains the class name
        :param image_path:
        :return:
        """
        parent, tail = os.path.split(image_path)
        parent, tail = os.path.split(parent)
        # if there is no mapping show the label as it
        label = self.labels.get(str(tail), tail)
        return label

    def get_random_image(self) -> tuple:
        """
        In case we don't ask for a specific image return a random one from test data set
        """
        print("Use a random image from test set")
        images, labels = next(iter(self._test_loader))
        image = images[0]
        mapping = self.class_to_idx.get(labels[0].item())
        # if there is not label mapping show the label as it is
        label = self.labels.get(mapping, labels[0].item())
        return image, label

    def view_classify(self, img: numpy_type, ps: float, label: str, labels: List[str],
                      topk: int = 5):
        """
        Function for viewing an image and it's predicted classes.
        """
        fig, (ax1, ax2) = plt.subplots(figsize=(5, 7), nrows=2)
        self.image_show(img, ax1)
        ax1.set_title(label)
        ax1.axis('off')
        ax2.barh(np.arange(topk), ps[0])
        ax2.set_aspect(0.1)
        ax2.set_yticks(np.arange(topk), minor=False)
        ax2.set_yticklabels(labels, size='large')
        ax2.invert_yaxis()
        ax2.set_title('Class Probability')
        ax2.set_xlim(0, 1.1)
        plt.tight_layout()
        plt.show()

    def image_show(self, image, ax=None):
        """
        Converts a pytorch image to numpy, restoring the mean and the std and plots to the given
        ax
        :param image: as pytorch type
        :param ax:
        :return: axes from plt
        """
        if ax is None:
            fig, ax = plt.subplots()
        # PyTorch tensors assume the color channel is the first dimension
        # but mat plot lib assumes is the third dimension
        image = image.numpy().transpose((1, 2, 0))
        # Undo pre processing
        image = self._std * image + self._mean
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
        ax.imshow(image)
        return ax

    def show_data_sets(self, name_of_dataset: str):
        if name_of_dataset.lower() == "test":
            self._show_images(self.test_data, "Testing data set")
        elif name_of_dataset.lower() == "valid":
            self._show_images(self.valid_data, "Validation data set")
        elif name_of_dataset.lower() == "train":
            self._show_images(self.train_data, "Training data set")
        else:
            raise Exception("Requesting unknown data set,options are test/train/valid")

    def _show_images(self, img_loader, title: str):
        columns = 8 - 1
        rows = 8 - 1
        images, labels = next(iter(img_loader))
        # set figure size to width=25 height =25
        fig = plt.figure(figsize=(25, 25))
        figure_title = title
        plt.text(0.5, 1.08, figure_title,
                 horizontalalignment='center',
                 fontsize=20)
        plt.axis("off")
        for i in range(1, columns * rows + 1):
            ax = fig.add_subplot(rows, columns, i)
            ax.axis("off")
            mapping = self.class_to_idx.get(labels[i].item())
            # if there is not label mapping show the label as it is
            label = self.labels.get(mapping, labels[i].item())
            ax.set_title(label)
            self.image_show(images[i], ax=ax)
        plt.show()


class Classifier(nn.Module):
    def __init__(self, input_layer: int = 25088, hidden_layers: list = None, output: int = 102,
                 dropout_prob: float = 0.5):
        """
        This model builds at max 3 layer NN network
        :param input_layer: must be the same as the input feature size
        :param hidden_layers: [256,128,64]
        :param output: number of different classes
        :param dropout_prob: used to increase the efficiency of training by training off the layers
        at each phase
        """
        self.hidden_layers = hidden_layers
        super().__init__()
        if len(hidden_layers) == 3:
            self.h1 = nn.Linear(in_features=input_layer, out_features=hidden_layers[0])
            self.h2 = nn.Linear(in_features=hidden_layers[0], out_features=hidden_layers[1])
            self.h3 = nn.Linear(in_features=hidden_layers[1], out_features=hidden_layers[2])
            self.h4 = nn.Linear(in_features=hidden_layers[2], out_features=output)

        if len(hidden_layers) == 2:
            self.h1 = nn.Linear(in_features=input_layer, out_features=hidden_layers[0])
            self.h2 = nn.Linear(in_features=hidden_layers[0], out_features=hidden_layers[1])
            self.h3 = nn.Linear(in_features=hidden_layers[1], out_features=output)

        if len(hidden_layers) == 1:
            self.h1 = nn.Linear(in_features=input_layer, out_features=hidden_layers[0])
            self.h2 = nn.Linear(in_features=hidden_layers[0], out_features=output)

        # set the drop out probability
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        """
        Do a forward pass to the Neural network, this method is called automatically
        with the constructor of the class
        """
        if len(self.hidden_layers) == 3:
            x = self.dropout(F.relu(self.h1(x)))
            x = self.dropout(F.relu(self.h2(x)))
            x = self.dropout(F.relu(self.h3(x)))
            x = F.log_softmax(self.h4(x), dim=1)

        if len(self.hidden_layers) == 2:
            x = self.dropout(F.relu(self.h1(x)))
            x = self.dropout(F.relu(self.h2(x)))
            x = F.log_softmax(self.h3(x), dim=1)

        if len(self.hidden_layers) == 1:
            x = self.dropout(F.relu(self.h1(x)))
            x = F.log_softmax(self.h2(x), dim=1)
        return x
