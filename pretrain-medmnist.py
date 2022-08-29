import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import torch
from dataloader import  *
import sys
from func import *
import segmentation_models_pytorch as smp
from config import models_genesis_config
from tqdm import tqdm
from glob import glob
import argparse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from tensorboardX import SummaryWriter
from nt_xent import NT_Xent
from contrast_loss import SupConLoss
import medmnist
from medmnist import INFO, Evaluator
# 


def get_conf_parser():
	parser = argparse.ArgumentParser(description='Command line arguments for segmentation target tasks.')
	parser.add_argument('--model', default="Unet3D")
	parser.add_argument('--suffix', default='new', type=str)
	parser.add_argument('--exp_name', default=None)
	parser.add_argument('--backbone', default="resnet50")
	parser.add_argument('--data_dir', help='test input image directory',default='./data')
	parser.add_argument('--data', default="./datasets/")
	parser.add_argument('--hu_min', default=-1000.0, type=float)
	parser.add_argument('--hu_max', default=1000.0, type=float)
	parser.add_argument('--amsgrad', help='whether to use the AMSGrad variant of Adam', default=False)
	parser.add_argument('--nb_epoch', help='number of epochs', default=100, type=int)
	parser.add_argument('--input_rows',  default=64, type=int)
	parser.add_argument('--input_cols',  default=64, type=int)
	parser.add_argument('--input_deps',  default=32, type=int)
	parser.add_argument('--nb_class', default=3, type=int)
	parser.add_argument('--verbose', default=1, type=int)
	parser.add_argument('--weights', default=None)
	parser.add_argument('-b','--batch_size', default=128, type=int)
	parser.add_argument('--optimizer', default='adam')
	parser.add_argument('--workers', default=4, type=int)
	parser.add_argument('--save_samples', default="png")
	parser.add_argument('--patience', default=30,type= int)
	parser.add_argument('--lr', type=float, default='1e-3', help='trial number')
	parser.add_argument('--outpaint_rate',  type=float, default=None)
	parser.add_argument('--nonlinear_rate', type=float, default=0.9)
	parser.add_argument('--local_rate', type=float, default=0.5)
	parser.add_argument('--visual', type=int, default=0.5)
	parser.add_argument('--flip_rate', type=float, default=0.4)
	parser.add_argument('--activate', help='activation', default="sigmoid")
	parser.add_argument('--inpaint_rate', type=float, default=0.2)
	parser.add_argument('--paint_rate', type=float, default=0.9)
	parser.add_argument('--twist_rate', type=float, default=0.9)
	parser.add_argument('--mask_ratio', type=float, default=0.25)

	parser.add_argument('--data_set',type=str, default='pathMNIST', help='index for the MedMNIST datasets')
	parser.add_argument("--normalization", type=str,help="how to normalize data (imagenet|chestx-ray)", default="chestx-ray")
	# Loss
	parser.add_argument("--temp", type=float, default=0.1)
	parser.add_argument("--contrastive_method", type=str, default='gcl', help='simclr, gcl(global contrastive learning), pcl(positional contrastive learning)')

	args = parser.parse_args()
	return args

def main(conf):	

	conf.exp_name = "MAGICAL-" + conf.suffix

	# logs

	model_path = os.path.join("./Models/Classification", conf.data_set, conf.exp_name)
	conf.logs_path = os.path.join("./Outputs/Classification",conf.data_set, conf.exp_name)

	conf.model_path = "./pretrained_models"
	if not os.path.exists(conf.model_path):
		os.makedirs(conf.model_path)
	conf.logs_path = os.path.join(conf.logs_path, "Logs",conf.exp_name)
	if not os.path.exists(conf.logs_path):
		os.makedirs(conf.logs_path)

	print(conf)
	output_file = os.path.join(conf.logs_path, "results.txt")
	log_writter = open(output_file, 'a')
	print("=> args '{}'".format(conf), file=log_writter)

	if conf.data_set == "dermaMNIST":
		diseases = ['TB']
		flag = "dermamnist"
		conf.data = './datasets/MedMNIST-main'

		conf.image_dir  = os.path.join(conf.data_dir,'dermamnist/dermamnist')
		conf.normalization = "chestx-ray"


		# preprocessing
		data_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[.5], std=[.5])
			#transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
		])
		
		dataset_train = DermaMNIST('train',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=conf.data)
		dataset_val = DermaMNIST('val',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=conf.data)
		dataset_test = DermaMNIST('test',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=conf.data)

	elif conf.data_set == "breastMNIST":
		diseases = ['TB']
		flag = "breastmnist"
		conf.data = './datasets/MedMNIST-main'

		conf.image_dir  = os.path.join(conf.data_dir,'breastmnist/breastmnist')
		conf.normalization = "chestx-ray"


		# preprocessing
		data_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[.5], std=[.5])
			#transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
		])
		
		dataset_train = BreastMNIST('train',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=conf.data)
		dataset_test = BreastMNIST('test',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=conf.data)


	elif conf.data_set == "pathMNIST":
		diseases = ['TB']
		flag = "pathmnist"
		info = INFO[flag]
		conf.task = INFO[flag]["task"]
		n_channels = info['n_channels']
		n_classes = len(info['label'])
		conf.data = './datasets/MedMNIST-main'
		#conf.normalization = "chestx-ray"
		conf.num_class = n_classes
		conf.n_channels = n_channels

		conf.image_dir  = os.path.join(conf.data_dir,'pathmnist/pathmnist')
		conf.normalization = "chestx-ray"


		# preprocessing
		data_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[.5], std=[.5])
			#transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
		])
		
		dataset_train = PathMNIST('train',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=conf.data)
		dataset_test = PathMNIST('test',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=conf.data)

	
	elif conf.data_set == "organaMNIST":
		diseases = ['TB']
		flag = "organamnist"
		conf.data = './datasets/MedMNIST-main'
		conf.image_dir  = os.path.join(conf.data_dir,'organamnist/organamnist')
		conf.normalization = "chestx-ray"
		# preprocessing
		data_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[.5], std=[.5])
			#transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
		])
		
		dataset_train = OrganAMNIST('train',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=conf.data)
		dataset_test = OrganAMNIST('test',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=conf.data)
	elif conf.data_set == "retinaMNIST":
		diseases = ['TB']
		flag = "retinamnist"
		conf.data = './datasets/MedMNIST-main'
		conf.image_dir  = os.path.join(conf.data_dir,'retinamnist/retinamnist')
		# preprocessing
		data_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[.5], std=[.5])
			#transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
		])
		
		dataset_train = RetinaMNIST('train',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=conf.data)
		dataset_test = RetinaMNIST('test',target_transform=None,transform=data_transform,download=False,as_rgb=True,root=conf.data)

	print("x_train: {}".format(len(dataset_train)))
	print("x_valid: {}".format(len(dataset_test)))

	data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=conf.batch_size, shuffle=True,
													num_workers=conf.workers,drop_last = True)
	data_loader_valid = torch.utils.data.DataLoader(dataset_test, batch_size=conf.batch_size,
													shuffle=False, num_workers=conf.workers)


	writer = SummaryWriter(log_dir=conf.logs_path)

	device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = smp.Unet(conf.backbone, encoder_weights=conf.weights,classes=conf.nb_class, activation=conf.activate)
		
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model, device_ids = [0,1,2,3])
	model.to(device)
	print("Total CUDA devices: ", torch.cuda.device_count())

	criterion_cl = NT_Xent(conf.batch_size, conf.temp, 1).to(device)#SupConLoss(temperature=conf.temp, contrastive_method=conf.contrastive_method).to(device)

	criterion_identity = torch.nn.MSELoss()
    
	if conf.optimizer == "sgd":
		optimizer = torch.optim.SGD(model.parameters(), conf.lr, momentum=0.9, weight_decay=0.0, nesterov=False)
	elif conf.optimizer == "adam":
		optimizer = torch.optim.Adam(model.parameters(), conf.lr, betas=(0.9, 0.99))
	elif conf.optimizer == "adamw":
		optimizer = torch.optim.AdamW(model.parameters(), conf.lr, amsgrad=conf.amsgrad)
	else:
		raise


	# to track the training loss as the model trains
	train_losses = []
	# to track the validation loss as the model trains
	valid_losses = []
	# to track the average training loss per epoch as the model trains
	avg_train_losses = []
	# to track the average validation loss per epoch as the model trains
	avg_valid_losses = []
	best_loss = 100000
	intial_epoch =0
	num_epoch_no_improvement = 0
	sys.stdout.flush()


	for epoch in range(intial_epoch,conf.nb_epoch):
		lr_ = cosine_anneal_schedule(epoch,conf.nb_epoch,conf.lr)
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr_
		# scheduler.step(epoch)
		model.train()
		pbar = tqdm(enumerate(data_loader_train), total=len(data_loader_train))
		# for iteration in range(int(x_train.shape[0]//conf.batch_size)):
		for iteration, (image,label) in pbar:

			# 
			image,gt,block_mask = generate_pair_new(image.numpy(),conf.batch_size, conf,"test")	
			image,gt = torch.from_numpy(image).float().to(device), torch.from_numpy(gt).float().to(device)
			pred,feature_pred=model(image)

			_,feature_gt = model(gt)

			loss_cl = criterion_cl(feature_pred,feature_gt)
			loss_identity = criterion_identity(pred,gt) 

			loss = loss_identity+loss_cl

  
			optimizer.zero_grad()
			loss.backward()

			optimizer.step()
			pbar.update(1)
			writer.add_scalar(f'Train-sum-loss', loss.item(), epoch*len(data_loader_train)+iteration)
			writer.add_scalar(f'Train-identity-loss', loss_identity.item(), epoch*len(data_loader_train)+iteration)
			writer.add_scalar(f'Train-CL-loss', loss_cl.item(), epoch*len(data_loader_train)+iteration)


			draw_in_tensorboard(writer, gt[0], epoch*len(data_loader_train)+iteration, image[0], pred[0],'Train') 

		with torch.no_grad():
			model.eval()
			print("validating....")
			pbar_valid = tqdm(enumerate(data_loader_valid), total=len(data_loader_valid))
			for iteration,(image,imagePath)  in pbar_valid:

				gt = image.to(device)

				pred,_=model(gt)

				loss = criterion_identity(pred,gt)
				valid_losses.append(loss.item())
				writer.add_scalar(f'Valid-loss', loss.item(), epoch*len(data_loader_valid)+iteration)

			draw_in_tensorboard(writer, gt[0], epoch*len(data_loader_valid)+iteration, image[0], pred[0],'Valid') 

		train_loss=np.average(train_losses)
		valid_loss=np.average(valid_losses)
		avg_train_losses.append(train_loss)
		avg_valid_losses.append(valid_loss)
		print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch+1,valid_loss,train_loss))
		train_losses=[]
		valid_losses=[]

		if valid_loss < best_loss:
			print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
			best_loss = valid_loss
			num_epoch_no_improvement = 0
			#save model
			torch.save({
				'epoch': epoch+1,
				'state_dict' : model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()
			},os.path.join(conf.model_path,"MAGICAL-{}.pth".format(conf.suffix)))
			print("Saving model ",os.path.join(conf.model_path,"MAGICAL-{}-{}.pth".format(conf.suffix,epoch)))
		else:
			print("Validation loss {:.4f}, num_epoch_no_improvement {}, best is {}".format(valid_loss,num_epoch_no_improvement,best_loss))
			num_epoch_no_improvement += 1
		if num_epoch_no_improvement == conf.patience:
			print("Early Stopping")
			break
		sys.stdout.flush()

if __name__ == '__main__':
    conf = get_conf_parser()
    main(conf)