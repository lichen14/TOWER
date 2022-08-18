import os
import torch
import random
import copy
import csv
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import numpy as np
#import pydicom as dicom
import cv2
from skimage import transform, io, img_as_float, exposure
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomBrightnessContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)

from medmnistinfo import INFO, HOMEPAGE, DEFAULT_ROOT


class MedMNIST(Dataset):

    flag = ...

    def __init__(self,
                 split,
                 transform=None,
                 target_transform=None,
                 download=False,
                 as_rgb=False,
                 root=DEFAULT_ROOT,
                 annotaion_percent=100):
        ''' dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation

        '''

        self.info = INFO[self.flag]

        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError("Failed to setup the default `root` directory. " +
                               "Please specify and create the `root` directory manually.")

        if download:
            self.download()

        if not os.path.exists(
                os.path.join(self.root, "{}.npz".format(self.flag))):
            raise RuntimeError('Dataset not found. ' +
                               ' You can set `download=True` to download it')

        npz_file = np.load(os.path.join(self.root, "{}.npz".format(self.flag)))

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb

        if self.split == 'train':
            self.imgs = npz_file['train_images']
            self.labels = npz_file['train_labels']
        elif self.split == 'val':
            self.imgs = npz_file['val_images']
            self.labels = npz_file['val_labels']
        elif self.split == 'test':
            self.imgs = npz_file['test_images']
            self.labels = npz_file['test_labels']
        else:
            raise ValueError
        # print('len(self.imgs)',len(self.imgs))
        indexes = np.arange(len(self.imgs))
        if annotaion_percent < 100:
          random.Random(99).shuffle(indexes)
          num_data = int(indexes.shape[0] * annotaion_percent / 100.0)
          # print(indexes.shape)
          indexes = indexes[:num_data]

          _img_list, _img_label = copy.deepcopy(self.imgs), copy.deepcopy(self.labels)
          self.imgs = []
          self.labels = []

          for i in indexes:
            self.imgs.append(_img_list[i])
            self.labels.append(_img_label[i])

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        '''Adapted from torchvision.ss'''
        _repr_indent = 4
        head = f"Dataset {self.__class__.__name__} ({self.flag})"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Root location: {self.root}")
        body.append(f"Split: {self.split}")
        body.append(f"Task: {self.info['task']}")
        body.append(f"Number of channels: {self.info['n_channels']}")
        body.append(f"Meaning of labels: {self.info['label']}")
        body.append(f"Number of samples: {self.info['n_samples']}")
        body.append(f"Description: {self.info['description']}")
        body.append(f"License: {self.info['license']}")

        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)

    def download(self):
        try:
            from torchvision.datasets.utils import download_url
            download_url(url=self.info["url"],
                         root=self.root,
                         filename="{}.npz".format(self.flag),
                         md5=self.info["MD5"])
        except:
            raise RuntimeError('Something went wrong when downloading! ' +
                               'Go to the homepage to download manually. ' +
                               HOMEPAGE)
    def __getitem__(self, index):
        '''
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        '''
        img, target = self.imgs[index], self.labels[index].astype(int)
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert('RGB')
        # img = img.astype('float32') / 255.
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MedMNIST2D(MedMNIST):

    def __getitem__(self, index):
        '''
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        '''
        img, target = self.imgs[index], self.labels[index].astype(int)
        img = Image.fromarray(img)
        if self.as_rgb:
            img = img.convert('RGB')
        # if self.transform != None: img = self.transform(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class PathMNIST(MedMNIST2D):
    flag = "pathmnist"


class OCTMNIST(MedMNIST2D):
    flag = "octmnist"


class PneumoniaMNIST(MedMNIST2D):
    flag = "pneumoniamnist"


class ChestMNIST(MedMNIST2D):
    flag = "chestmnist"


class DermaMNIST(MedMNIST2D):
    flag = "dermamnist"


class RetinaMNIST(MedMNIST2D):
    flag = "retinamnist"


class BreastMNIST(MedMNIST2D):
    flag = "breastmnist"


class BloodMNIST(MedMNIST2D):
    flag = "bloodmnist"


class TissueMNIST(MedMNIST2D):
    flag = "tissuemnist"


class OrganAMNIST(MedMNIST2D):
    flag = "organamnist"


class OrganCMNIST(MedMNIST2D):
    flag = "organcmnist"


class OrganSMNIST(MedMNIST2D):
    flag = "organsmnist"


def build_transform_classification(normalize, crop_size=224, resize=256, mode="train", test_augment=True):
    transformations_list = []

    if normalize.lower() == "imagenet":
      normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "chexpert":
      normalize = transforms.Normalize([0.5020, 0.5020, 0.5020], [0.085585, 0.085585, 0.085585])
    elif normalize.lower() == "none":
      normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)
    if mode == "train":
      transformations_list.append(transforms.RandomResizedCrop(crop_size))
      transformations_list.append(transforms.RandomHorizontalFlip())
      transformations_list.append(transforms.RandomRotation(7))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "valid":
      transformations_list.append(transforms.Resize((resize, resize)))
      transformations_list.append(transforms.CenterCrop(crop_size))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "test":
      if test_augment:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.TenCrop(crop_size))
        transformations_list.append(
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
          transformations_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
      else:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.CenterCrop(crop_size))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
          transformations_list.append(normalize)
    transformSequence = transforms.Compose(transformations_list)

    return transformSequence

def build_transform_segmentation():
  AUGMENTATIONS_TRAIN = Compose([
    # HorizontalFlip(p=0.5),
    OneOf([
        RandomBrightnessContrast(),
        RandomGamma(),
         ], p=0.3),
    OneOf([
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
    RandomSizedCrop(min_max_height=(156, 224), height=224, width=224,p=0.25),
    ToFloat(max_value=1)
    ],p=1)

  return AUGMENTATIONS_TRAIN

# ---------------------------------------------Downstream CheXpert------------------------------------------
class CheXpertDataset(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=5,
               uncertain_label="LSR-Ones", unknown_label=0, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
        label = line[5:]
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              label[i] = -1
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        imageLabel = [int(i) for i in label]
        self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    label = []
    for l in self.img_label[index]:
      if l == -1:
        if self.uncertain_label == "Ones":
          label.append(1)
        elif self.uncertain_label == "Zeros":
          label.append(0)
        elif self.uncertain_label == "LSR-Ones":
          label.append(random.uniform(0.55, 0.85))
        elif self.uncertain_label == "LSR-Zeros":
          label.append(random.uniform(0, 0.3))
      else:
        label.append(l)
    imageLabel = torch.FloatTensor(label)

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

class CheXpertDataset_pretrain(Dataset):
  def __init__(self, images_path, file_path, augment, num_class=5,
               uncertain_label="LSR-Ones", unknown_label=0, annotation_percent=100, size=224):

    self.data = []
    with open(file_path) as f:
      self.img_ids = [i_id.strip() for i_id in f]
    for name in self.img_ids:
      img_file = os.path.join(images_path,name)
      self.data.append(img_file)

  def __getitem__(self, index):
    imagePath = self.data[index]
    image = self.get_image(imagePath)


    image = self.preprocess(image)

    return image,imagePath#cv2.resize(image, (224, 224))

  def __len__(self):

    return len(self.data)

  def get_image(self, file):
    img = Image.open(file)
    # print(file)
    img = img.convert('RGB')
    img = img.resize((224, 224), Image.BICUBIC)
    # print('IMG SHAPE2',np.array(img).shape)
    return np.asarray(img, np.float32)/255.#img#

  def preprocess(self, image):
    #print(image.shape)
    image = image.transpose((2, 0, 1))
    #print(image.shape)
    #image = image[:, :, ::-1]  # change to BGR
    #image -= (128, 128, 128)
    return image#.transpose((2, 0, 1))

class CheXpertDataset_0208(Dataset):
  def __init__(self, images_path, file_path, augment, num_class=5,
               uncertain_label="LSR-Ones", unknown_label=0, annotation_percent=100, size=224):

    self.data = []
    with open(file_path) as f:
      self.img_ids = [i_id.strip() for i_id in f]
    for name in self.img_ids:
      img_file = os.path.join(images_path,name)
      self.data.append(img_file)

  def __getitem__(self, index):
    imagePath = self.data[index]
    image = self.get_image(imagePath)


    image = self.preprocess(image)

    return image,imagePath#cv2.resize(image, (224, 224))

  def __len__(self):

    return len(self.data)

  def get_image(self, file):
    img = Image.open(file)
    # print(file)
    img = img.convert('RGB')
    img = img.resize((224, 224), Image.BICUBIC)
    # print('IMG SHAPE2',np.array(img).shape)
    return np.asarray(img, np.float32)/255.#img#

  def preprocess(self, image):
    #print(image.shape)
    image = image.transpose((2, 0, 1))
    #print(image.shape)
    #image = image[:, :, ::-1]  # change to BGR
    #image -= (128, 128, 128)
    return image#.transpose((2, 0, 1))

# ---------------------------------------------Downstream Shenzhen------------------------------------------
class ShenzhenCXR(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=1, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split(',')

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

class ShenzhenCXR_new(Dataset):

  def __init__(self, images_path, file_path, augment=None, num_class=1, annotation_percent=100, size=224):

    self.size = size
    self.img_label = []
    
    self.augment = augment
    # random.shuffle(file_path)
    # self.img_list = file_path
    # self.img_label = list(map(lambda x: x.replace('.png', ''), self.img_list))
    # self.img_label = list(map(lambda x: int(x[-1]), self.img_label))
    # with open(file_path, "r") as fileDescriptor:
    #   line = True
    data = []
    labels = []
    for i in file_path:
        im = Image.open(i)
        im = im.convert('RGB')
        im = (np.array(im)).astype('float32')
        label = i.replace('.png', '')
        label = int(label[-1])
        # label = (np.array(label)).astype('uint8')
        data.append(cv2.resize(im, (size, size)))
        labels.append(label)

    self.data = np.array(data)
    self.label = np.array(labels)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    self.data = self.data.astype('float32') / 255.
    # self.label = self.label.astype('float32') / 255.

    # for i in range(3):
    #     self.data[:, :, :, i] = (self.data[:, :, :, i] - mean[i]) / std[i]

    # self.data = np.reshape(self.data, (
    #     len(self.data), size, size, 3))  # adapt this if using `channels_first` image data format
    # self.label = np.reshape(self.label,
    #                       (len(self.label), size, size, 1))  # adapt this if using `channels_first` im
    #   while line:
    #     line = fileDescriptor.readline()
    #     if line:
    #       lineItems = line.split(',')

    #       imagePath = os.path.join(images_path, lineItems[0])
    #       imageLabel = lineItems[1:num_class + 1]
    #       imageLabel = [int(i) for i in imageLabel]

    #       self.img_list.append(imagePath)
    #       self.img_label.append(imageLabel)
    # print(self.img_list)
    # print(self.img_label)

    indexes = np.arange(len(self.data))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    image = self.data[index]
    mask = self.label[index]

    image = image.transpose(2, 0, 1).astype('float32')
    # imagePath = self.img_list[index]

    # imageData = Image.open(imagePath).convert('RGB')
    # im = (np.array(imageData)).astype('float32')/ 255.
    # # print(im.shape)
    # im = cv2.resize(im, (self.size , self.size)).transpose(2, 0, 1).astype('float32')
    # imageLabel = self.img_label[index]#.transpose(2, 0, 1).astype('float32')

    # if self.augment != None: imageData = self.augment(image)
    # print(im,imageLabel)
    return image, mask

  def __len__(self):

    return self.data.shape[0]

class ShenzhenCXR_0216(Dataset):

  def __init__(self, images_path, file_path, augment=None, num_class=1, annotation_percent=100, size=512):

    self.size = size
    self.img_label = []
    self.augment = augment
    # random.shuffle(file_path)
    self.img_list = file_path
    self.img_label = list(map(lambda x: x.replace('.png', ''), self.img_list))
    self.img_label = list(map(lambda x: int(x[-1]), self.img_label))
    # with open(file_path, "r") as fileDescriptor:
    #   line = True

    #   while line:
    #     line = fileDescriptor.readline()
    #     if line:
    #       lineItems = line.split(',')

    #       imagePath = os.path.join(images_path, lineItems[0])
    #       imageLabel = lineItems[1:num_class + 1]
    #       imageLabel = [int(i) for i in imageLabel]

    #       self.img_list.append(imagePath)
    #       self.img_label.append(imageLabel)
    # print(self.img_list)
    # print(self.img_label)
    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')
    im = (np.array(imageData)).astype('float32')/ 255.
    # print(im.shape)
    im = cv2.resize(im, (self.size , self.size)).transpose(2, 0, 1).astype('float32')
    imageLabel = self.img_label[index]#.transpose(2, 0, 1).astype('float32')

    if self.augment != None: imageData = self.augment(imageData)
    # print(im,imageLabel)
    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

#__________________________________________Lung Segmentation, Montgomery dataset --------------------------------------------------
class MontgomeryDataset(Dataset):
    """NIH dataset."""

    def __init__(self, pathImageDirectory, pathMaskDirectory,transforms,dim=(224, 224, 3), anno_percent=100,num_class=1,normalization=None):
        self.transforms = transforms
        self.dim = dim
        self.pathImageDirectory=pathImageDirectory
        self.pathMaskDirectory =pathMaskDirectory
        self.normalization = normalization
        self.img_list= os.listdir(pathImageDirectory)

        indexes = np.arange(len(self.img_list))
        if anno_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * anno_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list= copy.deepcopy(self.img_list)
            self.img_list = []

            for i in indexes:
                self.img_list.append(_img_list[i])

    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        image_name= self.img_list[idx]
        image = Image.open(os.path.join(self.pathImageDirectory,image_name))
        image = image.convert('RGB')
        image = (np.array(image)).astype('uint8')
        mask = Image.open(os.path.join(self.pathMaskDirectory,image_name))
        mask = mask.convert('L')
        mask = (np.array(mask)).astype('uint8')
        image = cv2.resize(image, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask[mask > 0] = 255
        if self.transforms:
                augmented = self.transforms(image=image, mask=mask)
                im=augmented['image']
                mask=augmented['mask']
                im=np.array(im) / 255.
                mask=np.array(mask) / 255.
        else:
            im = np.array(image) / 255.
            mask = np.array(mask) / 255.
        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            im = (im-mean)/std

        mask = np.array(mask)
        im=im.transpose(2, 0, 1).astype('float32')
        mask=np.expand_dims(mask,axis=0).astype('uint8')
        return (im, mask)

class MontgomeryDataset_pretrain(Dataset):
    """NIH dataset."""

    def __init__(self, pathImageDirectory, pathMaskDirectory,transforms,dim=(224, 224, 3), anno_percent=100,img_list=None,normalization=None):
        self.transforms = transforms
        self.dim = dim
        self.pathImageDirectory=pathImageDirectory
        self.pathMaskDirectory =pathMaskDirectory
        self.normalization = normalization
        self.img_list= img_list#os.listdir(pathImageDirectory)

        indexes = np.arange(len(self.img_list))
        if anno_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * anno_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list= copy.deepcopy(self.img_list)
            self.img_list = []

            for i in indexes:
                self.img_list.append(_img_list[i])

    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        image_name= self.img_list[idx]
        image = Image.open(os.path.join(self.pathImageDirectory,image_name))
        image = image.convert('RGB')
        image = (np.array(image)).astype('uint8')

        mask_left = Image.open(os.path.join(self.pathMaskDirectory,'leftMask',image_name))
        mask_left = mask_left.convert('L')
        mask_left = (np.array(mask_left)).astype('uint8')
        
        mask_right = Image.open(os.path.join(self.pathMaskDirectory,'rightMask',image_name))
        mask_right = mask_right.convert('L')
        mask_right = (np.array(mask_right)).astype('uint8')

        image = cv2.resize(image, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask_left = cv2.resize(mask_left, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask_right = cv2.resize(mask_right, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask_left[mask_left > 0] = 255
        mask_left[mask_right > 0] = 255
        if self.transforms:
                augmented = self.transforms(image=image, mask=mask_left)
                im=augmented['image']
                mask=augmented['mask']
                im=np.array(im) / 255.
                mask=np.array(mask) / 255.
        else:
            im = np.array(image) / 255.
            mask = np.array(mask_left) / 255.
        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            im = (im-mean)/std

        mask = np.array(mask)
        im=im.transpose(2, 0, 1).astype('float32')
        mask=np.expand_dims(mask,axis=0).astype('uint8')
        return (im, mask)#image_name

class MontgomeryDataset_0218(Dataset):
    """NIH dataset."""

    def __init__(self, pathImageDirectory, pathMaskDirectory,transforms,dim=(224, 224, 3), anno_percent=100,img_list=None,normalization=None):
        self.transforms = transforms
        self.dim = dim
        self.pathImageDirectory=pathImageDirectory
        self.pathMaskDirectory =pathMaskDirectory
        self.normalization = normalization
        self.img_list= img_list#os.listdir(pathImageDirectory)

        indexes = np.arange(len(self.img_list))
        if anno_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * anno_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list= copy.deepcopy(self.img_list)
            self.img_list = []

            for i in indexes:
                self.img_list.append(_img_list[i])

    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        image2= self.img_list[idx]
        image_name = image2.replace(self.pathImageDirectory,'')
        image = Image.open(self.pathImageDirectory+image_name)
        image = image.convert('RGB')
        image = (np.array(image)).astype('uint8')

        mask_left = Image.open(self.pathMaskDirectory+'/leftMask'+image_name)
        mask_left = mask_left.convert('L')
        mask_left = (np.array(mask_left)).astype('uint8')
        
        mask_right = Image.open(self.pathMaskDirectory+'/rightMask'+image_name)
        mask_right = mask_right.convert('L')
        mask_right = (np.array(mask_right)).astype('uint8')

        image = cv2.resize(image, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask_left = cv2.resize(mask_left, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask_right = cv2.resize(mask_right, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask_left[mask_left > 0] = 255
        mask_left[mask_right > 0] = 255
        if self.transforms:
                augmented = self.transforms(image=image, mask=mask_left)
                im=augmented['image']
                mask=augmented['mask']
                im=np.array(im) / 255.
                mask=np.array(mask) / 255.
        else:
            im = np.array(image) / 255.
            mask = np.array(mask_left) / 255.
        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            im = (im-mean)/std

        mask = np.array(mask)
        im=im.transpose(2, 0, 1).astype('float32')
        mask=np.expand_dims(mask,axis=0).astype('uint8')
        return (im, mask)#image_name
#__________________________________________DRIVE dataset --------------------------------------------------

class DriveDataset(Dataset):
    """NIH dataset."""

    def __init__(self, pathImageDirectory, pathMaskDirectory,size=512):

        self.pathImageDirectory=pathImageDirectory
        self.pathMaskDirectory =pathMaskDirectory

        files = os.listdir(pathImageDirectory)
        data = []
        labels = []

        for i in files:
            im = Image.open(os.path.join(pathImageDirectory,i))
            im = im.convert('RGB')
            im = (np.array(im)).astype('uint8')
            label = Image.open(os.path.join(pathMaskDirectory, i.split('_')[0] + '_manual1.png'))
            label = label.convert('L')
            label = (np.array(label)).astype('uint8')
            data.append(cv2.resize(im, (size, size)))
            temp = cv2.resize(label, (size, size))
            _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
            labels.append(temp)

        self.data = np.array(data)
        self.label = np.array(labels)

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        self.data = self.data.astype('float32') / 255.
        self.label = self.label.astype('float32') / 255.

        for i in range(3):
            self.data[:, :, :, i] = (self.data[:, :, :, i] - mean[i]) / std[i]

        self.data = np.reshape(self.data, (
            len(self.data), size, size, 3))  # adapt this if using `channels_first` image data format
        self.label = np.reshape(self.label,
                             (len(self.label), size, size, 1))  # adapt this if using `channels_first` im

    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, idx):
        image = self.data[idx]
        mask = self.label[idx]

        image = image.transpose(2, 0, 1).astype('float32')
        mask = mask.transpose(2, 0, 1).astype('float32')

        return (image, mask)

class DriveDataset_new_0214(Dataset):
    """NIH dataset."""

    def __init__(self, pathImageDirectory, pathMaskDirectory,size=512,annotaion_percent=100):

        self.pathImageDirectory=pathImageDirectory
        self.pathMaskDirectory =pathMaskDirectory

        files = os.listdir(pathImageDirectory)
        data = []
        labels = []

        for i in files:
            im = Image.open(os.path.join(pathImageDirectory,i))
            im = im.convert('RGB')
            im = (np.array(im)).astype('uint8')
            
            label = Image.open(os.path.join(pathMaskDirectory, i.split('_')[0] + '_manual1.png'))
            label = label.convert('L')
            label = (np.array(label)).astype('uint8')
            data.append(cv2.resize(im, (size, size)))
            temp = cv2.resize(label, (size, size))
            _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
            labels.append(temp)

        self.data = np.array(data)
        self.label = np.array(labels)

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        self.data = self.data.astype('float32') / 255.
        self.label = self.label.astype('float32') / 255.
        self.files = files
        for i in range(3):
            self.data[:, :, :, i] = (self.data[:, :, :, i] - mean[i]) / std[i]

        self.data = np.reshape(self.data, (
            len(self.data), size, size, 3))  # adapt this if using `channels_first` image data format
        self.label = np.reshape(self.label,
                             (len(self.label), size, size, 1))  # adapt this if using `channels_first` im
        # indexes = np.arange(len(self.imgs))
        # if annotaion_percent < 100:
        #   random.Random(99).shuffle(indexes)
        #   num_data = int(indexes.shape[0] * annotaion_percent / 100.0)
        #   # print(indexes.shape)
        #   indexes = indexes[:num_data]

        #   _img_list, _img_label = copy.deepcopy(self.imgs), copy.deepcopy(self.labels)
        #   self.imgs = []
        #   self.labels = []

        #   for i in indexes:
        #     self.imgs.append(_img_list[i])
        #     self.labels.append(_img_label[i])
    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, idx):
        image = self.data[idx]
        mask = self.label[idx]
        imagePath = self.files[idx]
        image = image.transpose(2, 0, 1).astype('float32')
        mask = mask.transpose(2, 0, 1).astype('float32')

        return (image, mask)

class DriveDataset_new_0701(Dataset):
    """NIH dataset."""

    def __init__(self, pathImageDirectory, pathMaskDirectory,transforms=None,size=512, anno_percent=100,img_list=None,normalization=None):
        self.transforms = transforms
        self.size = size
        self.pathImageDirectory=pathImageDirectory
        self.pathMaskDirectory =pathMaskDirectory
        self.normalization = normalization
        self.img_list= os.listdir(pathImageDirectory)

        indexes = np.arange(len(self.img_list))
        if anno_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * anno_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list= copy.deepcopy(self.img_list)
            self.img_list = []

            for i in indexes:
                self.img_list.append(_img_list[i])

    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        image_name= self.img_list[idx]
        # print(image_name)
        im = Image.open(os.path.join(self.pathImageDirectory,image_name))
        im = im.convert('RGB')
        im = (np.array(im)).astype('uint8')

        mask_left = Image.open(os.path.join(self.pathMaskDirectory,image_name.split('_')[0] + '_manual1.png'))
        mask_left = mask_left.convert('L')
        mask_left = (np.array(mask_left)).astype('uint8')
        
        image = cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        temp = cv2.resize(mask_left, (self.size, self.size))
        _, labels = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
        # labels.append(temp)

        image = image.astype('float32') / 255.
        labels = labels.astype('float32') / 255.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        for i in range(3):
            image[ :, :, i] = (image[:, :, i] - mean[i]) / std[i]
            
        image = np.reshape(image, (
           self.size, self.size, 3))  # adapt this if using `channels_first` image data format
        labels= np.reshape(labels,
                             ( self.size, self.size, 1))

        image = image.transpose(2, 0, 1).astype('float32')
        mask = labels.transpose(2, 0, 1).astype('float32')
        
        # if self.transforms:
        #         augmented = self.transforms(image=image, mask=labels)
        #         im=augmented['image']
        #         mask=augmented['mask']
        #         im=np.array(im) / 255.
        #         mask=np.array(mask) / 255.
        # else:
        #     im = np.array(image) / 255.
        #     mask = np.array(mask_left) / 255.
        # if self.normalization == "imagenet":
        #     mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        #     im = (im-mean)/std

        # mask = np.array(mask)
        # im=im.transpose(2, 0, 1).astype('float32')
        # mask=np.expand_dims(mask,axis=0).astype('uint8')
        return (image, mask)#image_name

#__________________________________________Lung Segmentation, lits dataset --------------------------------------------------
class litsDataset_pretrain(Dataset):
    """LITS pretrain dataset."""

    def __init__(self, pathImageDirectory, pathMaskDirectory,size=256):

        self.pathImageDirectory=pathImageDirectory
        self.pathMaskDirectory =list(map(lambda x: x.replace('CT', 'seg').replace('volume', 'segmentation'), self.pathImageDirectory))

    def __len__(self):
        return len(self.pathImageDirectory)


    def __getitem__(self, idx):
      image_path = self.pathImageDirectory[idx]
      mask_path = self.pathMaskDirectory[idx]
      # print('image_path',image_path)
      im = Image.open(image_path)
      im = im.convert('RGB')
      im = (np.array(im)).astype('uint8')
      label = Image.open(mask_path)
      label = label.convert('L')
      label = (np.array(label)).astype('uint8')

      im = im.astype('float32') / 255.
      label = label.astype('float32') / 255.
      
      image = im.transpose(2, 0, 1).astype('float32')
      mask = label.astype('float32')#.transpose(2, 0, 1).astype('float32')
      print(image.shape,mask.shape)
      image = torch.FloatTensor(image)#.unsqueeze(0)
      mask = torch.FloatTensor(mask)

      return (image, mask)#

class litsDataset(Dataset):
    """LITS dataset."""

    def __init__(self, pathImageDirectory, pathMaskDirectory,size=256):

        self.pathImageDirectory=pathImageDirectory
        self.pathMaskDirectory =list(map(lambda x: x.replace('CT', 'seg').replace('volume', 'segmentation'), self.pathImageDirectory))

        # # files = os.listdir(pathImageDirectory)
        # data = []
        # labels = []

        # for i in pathImageDirectory:
        #     im = Image.open(i)
        #     im = im.convert('RGB')
        #     im = (np.array(im)).astype('uint8')
        #     label = Image.open(i.replace('CT', 'seg'))
        #     label = label.convert('L')
        #     label = (np.array(label)).astype('uint8')
        #     data.append(cv2.resize(im, (size, size)))
        #     # temp = cv2.resize(label, (size, size))
        #     # _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
        #     labels.append(label)

        # self.data = np.array(data)
        # self.label = np.array(labels)

        # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        # self.data = self.data.astype('float32') / 255.
        # self.label = self.label.astype('float32') / 255.

        # # for i in range(3):
        # #     self.data[:, :, :, i] = (self.data[:, :, :, i] - mean[i]) / std[i]

        # self.data = np.reshape(self.data, (
        #     len(self.data), size, size, 3))  # adapt this if using `channels_first` image data format
        # self.label = np.reshape(self.label,
        #                      (len(self.label), size, size, 1))  # adapt this if using `channels_first` im

    def __len__(self):
        return len(self.pathImageDirectory)


    def __getitem__(self, idx):
      image_path = self.pathImageDirectory[idx]
      mask_path = self.pathMaskDirectory[idx]
      # print('image_path',image_path)
      im = Image.open(image_path)
      im = im.convert('RGB')
      im = (np.array(im)).astype('uint8')
      label = Image.open(mask_path)
      label = label.convert('L')
      label = (np.array(label)).astype('uint8')

      im = im.astype('float32') / 255.
      label = label.astype('float32') / 255.
      
      image = im.transpose(2, 0, 1).astype('float32')
      mask = label.astype('float32')#.transpose(2, 0, 1).astype('float32')
      # print(image.shape,mask.shape)
      image = torch.FloatTensor(image)#.unsqueeze(0)
      mask = torch.FloatTensor(mask).unsqueeze(0)

      return (image, mask)