import os
import sys
import shutil
import time
import numpy as np
from shutil import copyfile
from tqdm import tqdm
from glob import glob
from utils import vararg_callback_bool, vararg_callback_int
from dataloader import  *
import argparse

import torch
from engine import segmentation_engine,segmentation_engine_0506
from utils import torch_dice_coef_loss

sys.setrecursionlimit(40000)
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

def get_args_parser():
    parser = argparse.ArgumentParser(description='Command line arguments for segmentation target tasks.')
    parser.add_argument('--train_data_dir', help='train input image directory',
                        default='./datasets/DRIVE/train/images')
    parser.add_argument('--train_mask_dir', help='train ground truth masks directory',
                        default='./datasets/DRIVE/train/labels')
    parser.add_argument('--valid_data_dir', help='validation input image directory',
                        default='./datasets/DRIVE/validate/images')
    parser.add_argument('--valid_mask_dir', help='validation ground truth masks directory',
                        default='./datasets/DRIVE/validate/labels')
    parser.add_argument('--test_data_dir', help='test input image directory',
                        default='./datasets/DRIVE/test/images')
    parser.add_argument('--test_mask_dir', help='test ground truth masks directory',
                        default='./datasets/DRIVE/test/labels')
    parser.add_argument('--data_set', help='target dataset',
                        default=None)
    parser.add_argument("--optimizer", dest="optimizer", help="Adam | SGD", default="Adam")
    parser.add_argument('--train_batch_size', help='train batch_size', default=32, type=int)
    parser.add_argument('--test_batch_size', help='test batch size', default=16, type=int)
    parser.add_argument('--epochs', help='number of epochs', default=200, type=int)
    parser.add_argument('--train_num_workers', help='train num of parallel workers for data loader', default=1,
                        type=int)
    parser.add_argument("--trial", dest="num_trial", help="number of trials", default=10, type=int)
    parser.add_argument("--start_index", dest="start_index", help="the start model index", default=0, type=int)
    parser.add_argument('--test_num_workers', help='test num of parallel workers for data loader', default=1, type=int)
    parser.add_argument('--distributed', help='whether to use distributed or not', dest='distributed',
                        action='store_true', default=False)
    parser.add_argument("--resume", dest="resume", help="whether latest checkpoint", default=False)
    parser.add_argument('-lr','--learning_rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('--mode', help='train|test', default='train')
    parser.add_argument('--backbone', help='encoder backbone', default='resnet50')
    parser.add_argument('--arch', help='segmentation network architecture', default='unet')
    parser.add_argument('--proxy_dir', help='path to pre-trained model', default="./pretrained_models")
    parser.add_argument('--device', help='cuda|cpu', default="cuda")
    parser.add_argument('--run', type=str, default='1', help='trial number')
    parser.add_argument('--init', help='None (random) |ImageNet |or other pre-trained methods', default=None)
    parser.add_argument('--normalization', help='imagenet|None', default=None)
    parser.add_argument('--activate', help='activation', default="sigmoid")
    parser.add_argument('--patience', type=int, default=20, help='num of patient epochesr')
    parser.add_argument("--annotaion_percent", dest="annotaion_percent", help="data percent", default=100, type=float)


    args = parser.parse_args()
    return args

def main(args):
    args.proxy_dir = os.path.join(args.proxy_dir, args.init+'.pth')
    print(args)
    assert args.train_data_dir is not None
    assert args.data_set is not None
    assert args.train_mask_dir is not None
    assert args.valid_data_dir is not None
    assert args.valid_mask_dir is not None
    assert args.test_data_dir is not None
    assert args.test_mask_dir is not None

    if args.init.lower() != 'imagenet' and args.init.lower() != 'random':
        assert args.proxy_dir is not None

    if args.init is not None:
        model_path = os.path.join("./Models/Segmentation", args.data_set, args.arch,  args.init,args.run)
        output_path = os.path.join("./Outputs/Segmentation",args.data_set, args.arch,  args.init,args.run)
    else:
        model_path = os.path.join("./Models/Segmentation", args.data_set, args.arch,  "random",args.run)
        output_path = os.path.join("./Outputs/Segmentation",args.data_set, args.arch, "random",args.run)

    args.exp_name = args.arch + "_" + args.init

    criterion = torch.nn.BCELoss()

    if args.data_set == "DRIVE":
        dataset_train = DriveDataset_new_0701(args.train_data_dir,args.train_mask_dir,anno_percent=args.annotaion_percent)
        dataset_val = DriveDataset_new_0701(args.valid_data_dir,args.valid_mask_dir)
        dataset_test = DriveDataset_new_0701(args.test_data_dir,args.test_mask_dir)
 
        segmentation_engine(args, model_path, output_path,dataset_train, dataset_val, dataset_test,criterion)

    if args.data_set == "LiTS": 
        train_data_dir = "/home/lc/Study/Project/data/LITS/CT/train_png_512_200"
        train_mask_dir = "/home/lc/Study/Project/data/LITS/seg/train_png_512_200"

        train_img_paths = glob(os.path.join(train_data_dir,'*'))
        train_mask_paths = glob(os.path.join(train_mask_dir,'*'))

        # train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(img_data, mask_data, test_size=0.2, random_state=42)
        valid_data_dir = "/home/lc/Study/Project/data/LITS/CT/valid_png_512_200"
        valid_mask_dir = "/home/lc/Study/Project/data/LITS/seg/valid_png_512_200"
        

        valid_img_paths = glob(os.path.join(valid_data_dir,'*'))
        valid_mask_paths = glob(os.path.join(valid_mask_dir,'*'))

        dataset_train = litsDataset(train_img_paths,train_mask_paths)
        dataset_val = litsDataset(valid_img_paths,valid_mask_paths)
        dataset_test = litsDataset(valid_img_paths,valid_mask_paths)
        segmentation_engine_0506(args, model_path, dataset_train, dataset_val, dataset_test,criterion)

    if args.data_set == "Montgomery":
        args.train_data_dir ='./datasets/MontgomerySet/CXR_png'
        args.train_mask_dir = './datasets/MontgomerySet/ManualMask'
        Img_list = glob(os.path.join(args.train_data_dir,'*'))
        Train_img_list, test_img_list = train_test_split(Img_list, test_size=0.2, random_state=42)
        train_img_list, valid_img_list = train_test_split(Train_img_list, test_size=0.2, random_state=42)
        dataset_train = MontgomeryDataset_0218(args.train_data_dir,args.train_mask_dir,img_list=train_img_list,transforms=None, normalization=args.normalization)
        dataset_val = MontgomeryDataset_0218(args.train_data_dir,args.train_mask_dir,img_list=valid_img_list,transforms=None, normalization=args.normalization)
        dataset_test = MontgomeryDataset_0218(args.train_data_dir,args.train_mask_dir,img_list=test_img_list,transforms=None, normalization=args.normalization)
        segmentation_engine_0506(args, model_path, dataset_train, dataset_val, dataset_test,criterion)

if __name__ == '__main__':
    args = get_args_parser()
    main(args)

