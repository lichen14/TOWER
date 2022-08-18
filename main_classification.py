import torch
import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from shutil import copyfile
from tqdm import tqdm
from glob import glob
from utils import vararg_callback_bool, vararg_callback_int
from dataloader import  *

import torch
from engine import classification_engine_medmnist,classification_engine_genesis

import medmnist
from medmnist import INFO, Evaluator
sys.setrecursionlimit(40000)

def get_args_parser():
    parser = OptionParser()

    parser.add_option("--GPU", dest="GPU", help="the index of gpu is used", default=None, action="callback",
                      callback=vararg_callback_int)
    parser.add_option("--model", dest="model_name", help="DenseNet121", default="Resnet50", type="string")
    parser.add_option("--init", dest="init",
                      help="Random | ImageNet| or any other pre-training method",
                      default="Random", type="string")
    parser.add_option("--num_class", dest="num_class", help="number of the classes in the downstream task",
                      default=14, type="int")
    parser.add_option("--data_set", dest="data_set", help="ChestXray14|CheXpert", default="ChestXray14", type="string")
    parser.add_option("--normalization", dest="normalization", help="how to normalize data (imagenet|chestx-ray)", default="imagenet",
                      type="string")
    parser.add_option("--img_size", dest="img_size", help="input image resolution", default=224, type="int")
    parser.add_option("--img_depth", dest="img_depth", help="num of image depth", default=3, type="int")
    parser.add_option("--data_dir", dest="data_dir", help="dataset dir",default="/home/lc/Study/Project/data", type="string")
    parser.add_option("--train_list", dest="train_list", help="file for training list",
                      default=None, type="string")
    parser.add_option("--val_list", dest="val_list", help="file for validating list",
                      default=None, type="string")
    parser.add_option("--test_list", dest="test_list", help="file for test list",
                      default=None, type="string")
    parser.add_option("--mode", dest="mode", help="train | test", default="train", type="string")
    parser.add_option("--batch_size", dest="batch_size", help="batch size", default=32, type="int")
    parser.add_option("--epochs", dest="num_epoch", help="num of epoches", default=1000, type="int")
    parser.add_option("--optimizer", dest="optimizer", help="Adam | SGD", default="adam", type="string")
    parser.add_option("--lr", dest="lr", help="learning rate", default=2e-4, type="float")
    parser.add_option("--lr_Scheduler", dest="lr_Scheduler", help="learning schedule", default="ReduceLROnPlateau",
                      type="string")
    parser.add_option("--patience", dest="patience", help="num of patient epoches", default=10, type="int")
    parser.add_option("--early_stop", dest="early_stop", help="whether use early_stop", default=True, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--trial", dest="num_trial", help="number of trials", default=1, type="int")
    parser.add_option("--start_index", dest="start_index", help="the start model index", default=0, type="int")
    parser.add_option("--clean", dest="clean", help="clean the existing data", default=False, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--resume", dest="resume", help="whether latest checkpoint", default=False, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--workers", dest="workers", help="number of CPU workers", default=1, type="int")
    parser.add_option("--print_freq", dest="print_freq", help="print frequency", default=30, type="int")
    parser.add_option("--test_augment", dest="test_augment", help="whether use test time augmentation",
                      default=True, action="callback", callback=vararg_callback_bool)
    parser.add_option("--proxy_dir", dest="proxy_dir", help="Path to the Pretrained model", default="./pretrained_models", type="string")
    parser.add_option("--annotaion_percent", dest="annotaion_percent", help="data percent", default=100, type="float")
    parser.add_option("--device", dest="device", help="cpu|cuda", default="cuda", type="string")
    parser.add_option("--activate", dest="activate", help="Sigmoid", default=None, type="string")   #"Sigmoid"
    parser.add_option("--uncertain_label", dest="uncertain_label",
                      help="the label assigned to uncertain data (Ones | Zeros | LSR-Ones | LSR-Zeros)",
                      default="LSR-Ones", type="string")
    parser.add_option("--unknown_label", dest="unknown_label", help="the label assigned to unknown data",
                      default=0, type="int")
    parser.add_option('--run', type="string", default='1', help='trial number')
    parser.add_option('--visual', default=False, help='visualize for t-sne or umap')
    parser.add_option("--scheduler", dest="scheduler", help="step|cycle", default="step", type="string")

    (options, args) = parser.parse_args()

    return options


def main(args):
    
    assert args.data_dir is not None

    if args.init.lower() != 'imagenet' and args.init.lower() != 'random':
        assert args.proxy_dir is not None
        
    args.proxy_dir = os.path.join(args.proxy_dir, args.init+'.pth')

    if args.init is not None:
        model_path = os.path.join("./Models/Classification", args.data_set, args.model_name,  args.init,args.run)
        output_path = os.path.join("./Outputs/Classification",args.data_set, args.model_name,  args.init,args.run)
    else:
        model_path = os.path.join("./Models/Classification", args.data_set, args.model_name,  "random",args.run)
        output_path = os.path.join("./Outputs/Classification",args.data_set, args.model_name, "random",args.run)
        
    args.exp_name = args.model_name + "_" + args.init

    args.criterion = torch.nn.CrossEntropyLoss()

    if  args.data_set == "CheXpert":
        diseases = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                           'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                           'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        # diseases_name
        diseases = ['Cardiomegaly','Edema',  'Consolidation','Atelectasis',  'Pleural Effusion']

        args.train_list = '/home/lc/Study/Project/data/CheXpert-v1.0-small/train_new.csv'
        args.val_list = '/home/lc/Study/Project/data/CheXpert-v1.0-small/valid_new.csv'
        args.test_list = '/home/lc/Study/Project/data/CheXpert-v1.0-small/test_new.csv'
        args.num_class = 5
        args.normalization = 'chestx-ray'
        args.criterion = torch.nn.BCEWithLogitsLoss()
        dataset_train = CheXpertDataset(images_path=args.data_dir, file_path=args.train_list,
                                        augment=build_transform_classification(normalize=args.normalization, mode="train"), uncertain_label=args.uncertain_label, unknown_label=args.unknown_label, annotation_percent=args.annotaion_percent)

        dataset_val = CheXpertDataset(images_path=args.data_dir, file_path=args.val_list,
                                      augment=build_transform_classification(normalize=args.normalization, mode="valid"), uncertain_label=args.uncertain_label, unknown_label=args.unknown_label)

        dataset_test = CheXpertDataset(images_path=args.data_dir, file_path=args.val_list,
                                       augment=build_transform_classification(normalize=args.normalization, mode="test", test_augment=False), uncertain_label=args.uncertain_label, unknown_label=args.unknown_label)

        classification_engine_genesis(args, model_path, output_path, diseases, dataset_train, dataset_test, dataset_test)#, test_diseases)

    elif args.data_set == "Shenzhen":
        diseases = ['TB']
        args.data = './datasets/Shenzhen'

        img_dir = os.path.join(args.data,'CXR_png')
        img_list = glob(os.path.join(img_dir,'*'))
        train_img_list0, test_img_list = train_test_split(img_list, test_size=0.2, random_state=42)
        train_img_list, valid_img_list = train_test_split(train_img_list0, test_size=0.2, random_state=2)

        dataset_train = ShenzhenCXR_new(images_path=args.data, augment=build_transform_classification(normalize=args.normalization, mode="train"),file_path=train_img_list)
        dataset_val = ShenzhenCXR_new(images_path=args.data, augment=build_transform_classification(normalize=args.normalization, mode="valid"),file_path=valid_img_list)
        dataset_test = ShenzhenCXR_new(images_path=args.data, augment=build_transform_classification(normalize=args.normalization, mode="test", test_augment=False),file_path=test_img_list)

        args.num_class = 1
        args.criterion = torch.nn.BCEWithLogitsLoss()
        classification_engine_genesis(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)
    
    elif args.data_set == "dermaMNIST":
        diseases = ['TB']
        flag = "dermamnist"
        args.data = './datasets/MedMNIST-main'
        args.task = INFO[flag]["task"]
        args.num_class = 7

        args.image_dir  = os.path.join(args.data_dir,'dermamnist/dermamnist')
        args.normalization = "chestx-ray"


        # preprocessing
        data_transform = transforms.Compose([
            transforms.ToTensor(),
			transforms.Normalize(mean=[.5], std=[.5])
        ])
        
        dataset_train = DermaMNIST('train',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=args.data,annotaion_percent=args.annotaion_percent)
        dataset_val = DermaMNIST('val',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=args.data)
        dataset_test = DermaMNIST('test',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=args.data)

        classification_engine_medmnist(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)
    elif args.data_set == "breastMNIST":
        diseases = ['TB']
        flag = "breastmnist"
        args.data = './datasets/MedMNIST-main'
        args.task = INFO[flag]["task"]
        args.num_class = 2

        args.image_dir  = os.path.join(args.data_dir,'breastmnist/breastmnist')
        args.normalization = "chestx-ray"


        # preprocessing
        data_transform = transforms.Compose([
            transforms.ToTensor(),
			transforms.Normalize(mean=[.5], std=[.5])
        ])
        
        dataset_train = BreastMNIST('train',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=args.data,annotaion_percent=args.annotaion_percent)
        dataset_val = BreastMNIST('val',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=args.data)
        dataset_test = BreastMNIST('test',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=args.data)

        classification_engine_medmnist(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)

    elif args.data_set == "pathMNIST":
        diseases = ['TB']
        flag = "pathmnist"
        info = INFO[flag]
        args.task = INFO[flag]["task"]
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        # print(n_classes)
        args.data = './datasets/MedMNIST-main'
        #args.normalization = "chestx-ray"
        args.num_class = n_classes
        args.n_channels = n_channels

        args.image_dir  = os.path.join(args.data_dir,'pathmnist/pathmnist')
        args.normalization = "chestx-ray"


        # preprocessing
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        
        dataset_train = PathMNIST('train',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=args.data,annotaion_percent=args.annotaion_percent)
        dataset_val = PathMNIST('val',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=args.data)
        dataset_test = PathMNIST('test',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=args.data)

        classification_engine_medmnist(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)
    
    elif args.data_set == "retinaMNIST":
        diseases = ['TB']
        flag = "retinamnist"
        args.data = './datasets/MedMNIST-main'
        args.task = INFO[flag]["task"]
        args.num_class = 5
        args.n_channels = INFO[flag]["n_channels"]
        args.image_dir  = os.path.join(args.data_dir,'retinamnist/retinamnist')
        args.normalization = "chestx-ray"
        args.criterion = torch.nn.CrossEntropyLoss()

        # preprocessing
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[.5], std=[.5])
        ])
        
        dataset_train = RetinaMNIST('train',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=args.data,annotaion_percent=args.annotaion_percent)
        dataset_val = RetinaMNIST('val',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=args.data)
        dataset_test = RetinaMNIST('test',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=args.data)

        classification_engine_medmnist(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)

    elif args.data_set == "organaMNIST":
        diseases = ['TB']
        flag = "organamnist"
        args.data = './datasets/MedMNIST-main'
        args.task = INFO[flag]["task"]
        args.num_class = 11
        args.n_channels = 3#INFO[flag]["n_channels"]
        args.image_dir  = os.path.join(args.data_dir,'organamnist/organamnist')
        args.normalization = "chestx-ray"

        # preprocessing
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        
        dataset_train = OrganAMNIST('train',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=args.data,annotaion_percent=args.annotaion_percent)
        dataset_val = OrganAMNIST('val',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=args.data)
        dataset_test = OrganAMNIST('test',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=args.data)

        classification_engine_medmnist(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)


if __name__ == '__main__':
    args = get_args_parser()
    main(args)
