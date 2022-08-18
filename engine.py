import os
import sys
import shutil
from tabnanny import verbose
import time
import numpy as np
from optparse import OptionParser
from tqdm import tqdm
import copy
from glob import glob

from models import build_classification_model, save_checkpoint, Net
from resnet_wider import resnet50
from utils import *

from tensorboardX import SummaryWriter

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau,OneCycleLR,StepLR
from trainer import *
import segmentation_models_pytorch as smp
from utils import cosine_anneal_schedule,dice,mean_dice_coef


sys.setrecursionlimit(40000)


def classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test, test_diseases=None):
  device = torch.device(args.device)
  cudnn.benchmark = True

  model_path = os.path.join(model_path, args.exp_name)

  if not os.path.exists(model_path):
    os.makedirs(model_path)

  if not os.path.exists(output_path):
    os.makedirs(output_path)


  # training phase
  if args.mode == "train":
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.workers, pin_memory=True, drop_last=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=14*args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)#,drop_last=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=14*args.batch_size, shuffle=False, #args.batch_size
                                  num_workers=args.workers, pin_memory=True)#, drop_last=True)#
    log_file = os.path.join(model_path, "models.log")

    print("start training....")
    output_file = os.path.join(output_path, args.exp_name + "_results.txt")
    log_writter = open(output_file, 'a')
    print("=> args '{}'".format(args), file=log_writter)
    for i in range(args.start_index, args.num_trial):
      print ("run:",str(i+1))
      start_epoch = 0
      init_loss = 1000000
      experiment = args.exp_name + "_run_" + str(i)
      best_val_loss = init_loss
      best_val_auc = 0
      best_val_dice = 0
      patience_counter = 0
      save_model_path = os.path.join(model_path, experiment)

      criterion = torch.nn.CrossEntropyLoss()
      model = build_classification_model(args)
      print(model)
      if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
      model.to(device)

      parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

      if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(parameters, args.lr, momentum=0.9, weight_decay=1e-5, nesterov=False)
      elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, args.lr)
      elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(parameters, args.lr)#, amsgrad=args.amsgrad)
      else:
        raise

      lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=args.patience // 2, mode='min',
                                       threshold=0.0001, min_lr=0, verbose=True)

      lr_ = args.lr
      if args.resume:
        resume = os.path.join(model_path, 'best-checkpoint.pth')
        if os.path.isfile(resume):
          print("=> loading resume checkpoint '{}'".format(resume))
          checkpoint = torch.load(resume)

          start_epoch = checkpoint['epoch']+1
          best_val_loss = checkpoint['lossMIN']
          # best_val_auc = checkpoint['aucMax']
          model.load_state_dict(checkpoint['state_dict'])
          lr_ =checkpoint['lr']
          #lr_scheduler.load_state_dict(checkpoint['scheduler'])
          optimizer.load_state_dict(checkpoint['optimizer'])
          print("=> loaded checkpoint '{}' (epoch={:04d}, val_loss={:.5f})"
                .format(resume, start_epoch, best_val_loss))
        else:
          print("=> no checkpoint found at '{}'".format(args.resume))


      
      for epoch in range(start_epoch, args.num_epoch):
        val_loss,val_dice,val_auc = evaluate_txc(data_loader_val, device,model, criterion,args)

        train_classify_old(data_loader_train,device, model, criterion, optimizer, epoch)
        val_loss,val_dice,val_auc = evaluate_txc(data_loader_val, device,model, criterion,args)

        print('val_acc,val_auc :',val_dice,val_auc)
        # if lr_>0.00001:
        lr_ = cosine_anneal_schedule(epoch,args.num_epoch,args.lr)
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr_
        if val_auc > best_val_auc:##val_loss < best_val_loss:#
          print("Epoch {:04d}: AUC improved from {:.5f} to {:.5f}, current loss is {:.5f}***********".format(epoch, best_val_auc, val_auc,val_loss))
          print("Epoch {:04d}: AUC improved from {:.5f} to {:.5f}, current loss is {:.5f}***********".format(epoch, best_val_auc, val_auc,val_loss), file=log_writter)
          best_val_auc = val_auc
          best_val_loss = val_loss
          patience_counter = 0    

          torch.save({
              'best_auc': best_val_auc,
              'epoch': epoch,
              'lossMIN': best_val_loss,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'scheduler': lr_scheduler.state_dict(),
            },  os.path.join(model_path, "best-checkpoint-{}.pth".format(i)))

        else:
          print("Epoch {:04d}: current val_loss is: {:.5f},auc is {:.5f}, best auc is  {:.4f}".format(epoch, val_loss,val_auc,best_val_auc ))
          print("Epoch {:04d}: current val_loss is: {:.5f},auc is {:.5f}, best auc is  {:.4f}".format(epoch, val_loss,val_auc,best_val_auc ), file=log_writter)
          patience_counter += 1

        if patience_counter > args.patience:
          print("Early Stopping")
          break


        # log experiment
        with open(log_file, 'a') as f:
          f.write(str(epoch)+ "\t"+str(val_auc) + "\t"+str(val_loss)+ "\n")
          f.close()

      print ("start testing.....")
      

    mean_auc = []
    mean_dice = []
    y_features = []
    y_labels = []
    #with open(output_file, 'a') as writer:
    print("args = {}\n".format(args))

    for experiment in glob(os.path.join(model_path, "*.pth")):

      if args.visual:
        os.makedirs(os.path.join(model_path, "visual"), exist_ok=True)
        y_test, y_feature,Acc_avg = test_visual(experiment, data_loader_val, device, args)

        torch.save({
          'feature': y_feature,
          'image': y_test,
        },  os.path.join(model_path, "visual","visual-samples-trainset-{}.pth").format(os.path.basename(experiment)))

      else:
        _,_,Auc_avg = test_classification_0412(experiment, data_loader_test, device, args)

        print(">>{}: AUC = {:.4f}".format(experiment, Auc_avg))
        print("{}: AUC = {:.4f}\n".format(experiment, Auc_avg),file=log_writter)
        mean_auc.append(Auc_avg)
        
    mean_auc = np.array(mean_auc)
    print(">> All trials: mAUC  = {}".format(np.array2string(mean_auc, precision=4, separator=',')))
    print(">> All trials: mAUC  = {}".format(np.array2string(mean_auc, precision=4, separator=',')),file=log_writter)
    print(">> Mean AUC over All trials: = {:.4f}".format(np.mean(mean_auc)))
    print(">> Mean AUC over All trials = {:.4f}\n".format(np.mean(mean_auc)),file=log_writter)
    print(">> STD over All trials:  = {:.4f}".format(np.std(mean_auc)))
    print(">> STD over All trials:  = {:.4f}".format(np.std(mean_auc)),file = log_writter)
def classification_engine_genesis(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test, test_diseases=None):
  device = torch.device(args.device)
  cudnn.benchmark = True

  model_path = os.path.join(model_path, args.exp_name)

  if not os.path.exists(model_path):
    os.makedirs(model_path)

  if not os.path.exists(output_path):
    os.makedirs(output_path)


  # training phase
  if args.mode == "train":
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.workers, pin_memory=True)#, drop_last=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=146, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)#,drop_last=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, #args.batch_size
                                  num_workers=args.workers, pin_memory=True)#, drop_last=True)#
    log_file = os.path.join(model_path, "models.log")
    # print(len(data_loader_val))
    # training phase
    print("start training....")
    output_file = os.path.join(output_path, args.exp_name + "_results.txt")
    log_writter = open(output_file, 'a')
    print("=> args '{}'".format(args), file=log_writter)
    for i in range(args.start_index, args.num_trial):
      print ("run:",str(i+1))
      start_epoch = 0
      init_loss = 1000000
      experiment = args.exp_name + "_run_" + str(i)
      best_val_loss = init_loss
      best_val_auc = 0
      best_val_dice = 0
      patience_counter = 0
      save_model_path = os.path.join(model_path, experiment)
      criterion = args.criterion.to(device)
      model = build_classification_model(args)
      if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
      model.to(device)

      parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

      if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(parameters, args.lr, momentum=0.9, weight_decay=1e-5, nesterov=False)
      elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, args.lr, betas=(0.9, 0.999))
      elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(parameters, args.lr)#, amsgrad=args.amsgrad)
      else:
        raise

      lr_scheduler = StepLR(optimizer, step_size=50, gamma=0.5,last_epoch=-1)

      lr_ = args.lr
      if args.resume:
        resume = os.path.join(model_path, 'best-checkpoint.pth')
        if os.path.isfile(resume):
          print("=> loading resume checkpoint '{}'".format(resume))
          checkpoint = torch.load(resume)

          start_epoch = checkpoint['epoch']+1
          best_val_loss = checkpoint['lossMIN']
          model.load_state_dict(checkpoint['state_dict'])
          lr_ = checkpoint['lr']

          optimizer.load_state_dict(checkpoint['optimizer'])
          print("=> loaded checkpoint '{}' (epoch={:04d}, val_loss={:.5f})"
                .format(resume, start_epoch, best_val_loss))
        else:
          print("=> no checkpoint found at '{}'".format(args.resume))


      
      for epoch in range(start_epoch, args.num_epoch):

        train_loss,train_acc = train_classify_old(data_loader_train,device, model, criterion, optimizer, epoch,args)
        val_loss,val_acc,val_auc = evaluate_txc(data_loader_val, device,model, criterion,args)
  
        lr_scheduler.step()

        print('val_auc ,lr:',val_auc,lr_scheduler.get_last_lr()[0])
        if val_auc > best_val_auc:##val_loss < best_val_loss:#
          print("Epoch {:04d}: AUC improved from {:.5f} to {:.5f}, train_loss is {:.5f}, valid loss is {:.5f}****".format(epoch, best_val_auc, val_auc,train_loss,val_loss))
          print("Epoch {:04d}: current val_loss is: {:.5f},auc is {:.5f}, best auc is  {:.4f}".format(epoch, val_loss,val_auc,best_val_auc ), file=log_writter)
          best_val_auc = val_auc
          best_val_loss = val_loss
          patience_counter = 0    

          torch.save({
              'best_auc': best_val_auc,
              'epoch': epoch,
              'lossMIN': best_val_loss,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'scheduler': lr_scheduler.state_dict(),
            },  os.path.join(model_path, "best-checkpoint-{}.pth".format(i)))

        else:
          print("Epoch {:04d}: current val_loss is: {:.5f},auc is {:.5f}, best auc is  {:.4f}".format(epoch, val_loss,val_auc,best_val_auc ))
          print("Epoch {:04d}: current val_loss is: {:.5f},auc is {:.5f}, best auc is  {:.4f}".format(epoch, val_loss,val_auc,best_val_auc ), file=log_writter)
          patience_counter += 1

        if patience_counter > args.patience:
          print("Early Stopping")
          break


        # log experiment
        with open(log_file, 'a') as f:
          f.write(str(epoch)+ "\t"+str(val_auc) + "\n")#+ "\t"+str(val_loss)+ "\n")
          f.close()

      print ("start testing.....")
      

    
    mean_auc = []
    print("args = {}\n".format(args))
    for experiment in glob(os.path.join(model_path, "*.pth")):

      if args.visual:
        os.makedirs(os.path.join(model_path, "visual"), exist_ok=True)
        y_test, y_feature,Acc_avg = test_visual(experiment, data_loader_val, device, args)

        torch.save({
          'feature': y_feature,
          'image': y_test,
        },  os.path.join(model_path, "visual","visual-samples-trainset-{}.pth").format(os.path.basename(experiment)))

      else:
        y_test, p_test = test_classification(experiment, data_loader_test, device, args)
        Auc_avg = roc_auc_score(y_test.cpu().data, p_test.cpu().data)
        print(">>{}: AUC = {:.4f}".format(experiment, Auc_avg))
        print("{}: AUC = {:.4f}".format(experiment, Auc_avg),file=log_writter)
        mean_auc.append(Auc_avg)
        
    mean_auc = np.array(mean_auc)
    print(">> All trials: mAUC  = {}".format(np.array2string(mean_auc, precision=4, separator=',')))
    print(">> All trials: mAUC  = {}".format(np.array2string(mean_auc, precision=4, separator=',')),file=log_writter)
    #writer.write("All trials: mAUC  = {}\n".format(np.array2string(mean_auc, precision=4, separator='\t')))
    print(">> Mean AUC over All trials: = {:.4f}".format(np.mean(mean_auc)))
    print(">> Mean AUC over All trials = {:.4f}\n".format(np.mean(mean_auc)),file=log_writter)
    print(">> STD over All trials:  = {:.4f}".format(np.std(mean_auc)))
    print(">> STD over All trials:  = {:.4f}".format(np.std(mean_auc)),file = log_writter)
def classification_engine_medmnist(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test, test_diseases=None):
  device = torch.device(args.device)

  USE_CUDA = torch.cuda.is_available()
  cudnn.benchmark = True

  model_path = os.path.join(model_path, args.exp_name)

  if not os.path.exists(model_path):
    os.makedirs(model_path)

  if not os.path.exists(output_path):
    os.makedirs(output_path)


  # training phase
  if args.mode == "train":

    data_loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.workers, pin_memory=True, drop_last=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=1024, shuffle=False,
                                 num_workers=args.workers, pin_memory=True,drop_last=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=1024, shuffle=False, #args.batch_size
                                  num_workers=args.workers, pin_memory=True, drop_last=True)#
    log_file = os.path.join(model_path, "models.log")
    # print(len(data_loader_train),len(data_loader_val))
    # training phase
    print(args)
    print("start training....")
    output_file = os.path.join(output_path, args.exp_name + "_results.txt")
    log_writter = open(output_file, 'a')
    print("=> args '{}'".format(args), file=log_writter)
    for i in range(args.start_index, args.num_trial):
      print ("run:",str(i))
      print ("run:",str(i), file=log_writter)
      start_epoch = 0
      init_loss = 1000000
      experiment = args.exp_name + "_run_" + str(i)
      best_val_loss = init_loss
      best_val_auc = 0
      best_val_dice = 0
      patience_counter = 0
      save_model_path = os.path.join(model_path, experiment)
      criterion = args.criterion.to(device)#torch.nn.BCEWithLogitsLoss()#

      model = build_classification_model(args)
      if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
      model.to(device)

      parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

      if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(parameters, args.lr, momentum=0.9, weight_decay=1e-5, nesterov=False)
      elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, args.lr, betas=(0.9, 0.999))
      elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(parameters, args.lr)#, amsgrad=args.amsgrad)
      else:
        raise
      

      lr_ = args.lr
      if args.resume:
        resume = os.path.join(model_path, 'best-checkpoint.pth')
        if os.path.isfile(resume):
          print("=> loading resume checkpoint '{}'".format(resume))
          checkpoint = torch.load(resume)

          start_epoch = checkpoint['epoch']+1
          best_val_loss = checkpoint['lossMIN']
          # best_val_auc = checkpoint['aucMax']
          model.load_state_dict(checkpoint['state_dict'])
          lr_ = checkpoint['lr']
          optimizer.load_state_dict(checkpoint['optimizer'])
          print("=> loaded checkpoint '{}' (epoch={:04d}, val_loss={:.5f})"
                .format(resume, start_epoch, best_val_loss))
        else:
          print("=> no checkpoint found at '{}'".format(args.resume))


      
      for epoch in range(start_epoch, args.num_epoch):

        if args.task == 'multi-label, binary-class':
          train_classify_old(data_loader_train,device, model, criterion, optimizer, epoch,args)
          y_test,p_test = evaluate_classify(model,data_loader_val, device,args)
          val_auc = roc_auc_score(y_test.cpu().data, p_test.cpu().data)
          val_loss = criterion(y_test.cpu().data, p_test.cpu().data)
        else:
          train_classify(data_loader_train,device, model, criterion, optimizer, epoch,args)
          val_loss,val_dice,val_auc = evaluate(data_loader_val, device,model, criterion,args)
        
        print('val_acc,val_auc,lr :',val_auc,lr_)
        if val_auc > best_val_auc:##val_loss < best_val_loss:#

          print("Epoch {:04d}: AUC improved from {:.5f} to {:.5f}, current loss is {:.5f}***********".format(epoch, best_val_auc, val_auc,val_loss))
          print("Epoch {:04d}: AUC improved from {:.5f} to {:.5f}, current loss is {:.5f}***********".format(epoch, best_val_auc, val_auc,val_loss), file=log_writter)
          best_val_auc = val_auc
          best_val_loss = val_loss
          patience_counter = 0    

          torch.save({
              'best_auc': best_val_auc,
              # 'epoch': epoch,
              'lossMIN': best_val_loss,
              'state_dict': model.state_dict(),
              # 'optimizer': optimizer.state_dict(),
              # 'scheduler': lr_scheduler.state_dict(),
            },  os.path.join(model_path, "best-checkpoint-{}.pth".format(i)))

        else:

          print("Epoch {:04d}: current val_loss is: {:.6f},auc is {:.5f}, best auc is  {:.4f}".format(epoch, val_loss,val_auc,best_val_auc ))
          print("Epoch {:04d}: current val_loss is: {:.5f},auc is {:.5f}, best auc is  {:.4f}".format(epoch, val_loss,val_auc,best_val_auc ), file=log_writter)
          patience_counter += 1

        if patience_counter > args.patience:
          print("Early Stopping")
          break


        # log experiment
        with open(log_file, 'a') as f:
          f.write(str(epoch)+ "\t"+str(val_auc) + "\n")
          f.close()

      print ("start testing.....")

    mean_auc = []

    print("args = {}\n".format(args))

    for experiment in glob(os.path.join(model_path, "*.pth")):
      print(experiment)

      if args.visual:
        os.makedirs(os.path.join(model_path, "visual"), exist_ok=True)
        y_test, y_feature,Auc_avg = test_visual(experiment, data_loader_test, device, args)

      else:
        if args.task == 'multi-label, binary-class':
          y_test, p_test = test_classification(experiment, data_loader_test, device, args)
          Auc_avg = roc_auc_score(y_test.cpu().data, p_test.cpu().data)
        else:
          _,_,Auc_avg = test_classification_0412(experiment, data_loader_test, device, args)

      print(">>{}: AUC = {:.4f}".format(experiment, Auc_avg))
      print("{}: AUC = {:.4f}".format(experiment, Auc_avg),file=log_writter)
      mean_auc.append(Auc_avg)
        
    mean_auc = np.array(mean_auc)
    print(">> All trials: mAUC  = {}".format(np.array2string(mean_auc, precision=4, separator=',')))
    print(">> All trials: mAUC  = {}".format(np.array2string(mean_auc, precision=4, separator=',')),file=log_writter)
    print(">> Mean AUC over All trials: = {:.4f}".format(np.mean(mean_auc)))
    print(">> Mean AUC over All trials = {:.4f}\n".format(np.mean(mean_auc)),file=log_writter)
    print(">> STD over All trials:  = {:.4f}".format(np.std(mean_auc)))
    print(">> STD over All trials:  = {:.4f}".format(np.std(mean_auc)),file = log_writter)

def segmentation_engine(args, model_path, output_path,dataset_train, dataset_val, dataset_test,criterion):
  device = torch.device(args.device)
  if not os.path.exists(model_path):
    os.makedirs(model_path)

  logs_path = os.path.join(model_path, "Logs")
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  output_file = os.path.join(output_path, args.exp_name + "_results.txt")
  log_writter = open(output_file, 'a')

  print(args,file=log_writter)

  if args.mode == "train":
    start_num_epochs=0
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size,pin_memory=True,  shuffle=True,
                                                 num_workers=args.train_num_workers, drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.train_batch_size,pin_memory=True, 
                                                      shuffle=False, num_workers=args.train_num_workers)
    for i in range(args.start_index, args.num_trial):
      print ("run:",str(i+1))
      print ("run:",str(i), file=log_writter)
      if args.init.lower() == "random":
        print("init with scratch")
        model = smp.Unet(args.backbone, encoder_weights=None, activation=args.activate)
      elif args.init.lower() == "supervised":
        print("init with encoder from imagenet")
        model = smp.Unet(args.backbone, encoder_weights=args.proxy_dir.replace('supervised','imagenet'), activation=args.activate)

      elif args.init.lower() == "imagenet":
        print("init with encoder from imagenet")
        model = smp.Unet(args.backbone, encoder_weights=args.proxy_dir.replace('imagenet','supervised'), activation=args.activate)
        print(model)

      elif args.init.startswith('MAGICAL'):
        print("init with encoder + decoder",args.proxy_dir)
        model = smp.Unet(args.backbone, encoder_weights=args.proxy_dir, activation=args.activate)
        
        checkpoint = torch.load(args.proxy_dir, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        # print('my test-MG',state_dict.keys(), file=log_writter)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        for k in list(state_dict.keys()):
            if k.startswith('segmentation_head') or k.startswith('projection_head'):
              # print(k)
              del state_dict[k]
        # print('after del',state_dict.keys(), file=log_writter)
        _msg = model.load_state_dict(state_dict,strict = False)
        if len(_msg.missing_keys) != 0:
          print('_msg.missing_keys is :',_msg.missing_keys)
          if args.backbone.lower().startswith("resnet"):
            if args.activate is None:
                assert set(_msg.missing_keys) == {"segmentation_head.weight", "segmentation_head.bias"}
            else:
                assert set(_msg.missing_keys) == {"segmentation_head.0.weight", "segmentation_head.0.bias"}#{'segmentation_head.conv2d_3x3.weight', 'segmentation_head.conv2d_3x3.bias', 'segmentation_head.conv2d_1x1.weight', 'segmentation_head.conv2d_1x1.bias'}#
          elif args.backbone.lower().startswith("densenet"):
            if args.activate is None:
                assert set(_msg.missing_keys) == {"classifier.weight", "classifier.bias"}
            else:
                assert set(_msg.missing_keys) == {"classifier.0.weight", "classifier.0.bias"}
      else:
        print("init with encoder from proxy_dir",args.proxy_dir)

        model = smp.Unet(args.backbone, encoder_weights=args.proxy_dir, activation=args.activate)
      
      
      print(model,file=log_writter)
      # print(model)
      writer = SummaryWriter(log_dir=os.path.join(logs_path,args.data_set+'-'+args.run))
      
      if args.optimizer.lower() == "sgd":
          optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5, nesterov=False)
      elif args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
      elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate)#, amsgrad=args.amsgrad)
      else:
        raise
      # lr_scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps =100,verbose=False)# args.num_epoch
      lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.5,last_epoch=-1)#StepLR(optimizer, max_lr=0.01, steps_per_epoch=len(data_loader_train),total_steps =args.num_epoch,verbose=False)# args.num_epoch

      if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
      model.to(device)
      best_val_loss = 100000
      best_val_dice = 0
      patience_counter = 0
      lr_ = args.learning_rate
      for epoch in range(start_num_epochs, args.epochs):
        train_one_epoch(data_loader_train,device, model, criterion, optimizer, epoch)
        val_iou,val_dice,val_loss = evaluate_seg(data_loader_val, device, model, epoch,criterion,writer)
        
        if val_dice > best_val_dice :#val_loss < best_val_loss:

          torch.save({
              'epoch': epoch,
              'state_dict': model.state_dict(),
            },  os.path.join(model_path, "best-checkpoint-{}.pth".format(i)))

          print("Epoch {:04d}: DICE improved from {:.5f} to {:.5f}, current loss is {:.5f}***********".format(epoch, best_val_dice, val_dice,val_loss))
          print("Epoch {:04d}: DICE improved from {:.5f} to {:.5f}, current loss is {:.5f}***********".format(epoch, best_val_dice, val_dice,val_loss), file=log_writter)
          best_val_dice = val_dice                                                                                    # os.path.join(model_path,"checkpoint.pt")))
          best_val_iou = val_iou
          patience_counter = 0

        else:

          print("Epoch {:04d}: current val_loss is: {:.5f},DICE is {:.5f}, best DICE is  {:.4f}".format(epoch, val_loss,val_dice,best_val_dice ))
          print("Epoch {:04d}: current val_loss is: {:.5f},DICE is {:.5f}, best DICE is  {:.4f}".format(epoch, val_loss,val_dice,best_val_dice ), file=log_writter)
          patience_counter += 1


          if patience_counter > args.patience:
            print("Early Stopping")
            break

        #log_writter.flush()
  torch.cuda.empty_cache()
  model = smp.Unet(args.backbone, encoder_weights=None, activation=args.activate)
  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
  model.to(device)
  data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False,
                                                num_workers=args.test_num_workers)
  model_save_path = os.path.join(model_path, "best-checkpoint.pth")
  model_list = glob(os.path.join(model_path,"best-checkpoint-*.pth"))
  # print("testing....",name)
  DICE_test = []
  IOU_test = []
  for name in model_list:
    # print("testing....",name,file=log_writter)
    DICE,IOU= test_segmentation_0214(model, name, data_loader_test, device, log_writter)
    DICE_test.append(DICE)
    # IOU_test.append(IOU)
  # print('DICE_test',DICE_test)
  mean_auc = np.array(DICE_test)
  print(">> All trials: mAUC  = {}".format(np.array2string(mean_auc, precision=2, separator=',')))
  print(">> All trials: mAUC  = {}".format(np.array2string(mean_auc, precision=2, separator=',')),file=log_writter)
  print(">> Mean AUC over All trials: = {:.4f}".format(np.mean(mean_auc)))
  print(">> Mean AUC over All trials = {:.4f}".format(np.mean(mean_auc)),file=log_writter)
  print(">> STD over All trials:  = {:.4f}".format(np.std(mean_auc)))
  print(">> STD over All trials:  = {:.4f}".format(np.std(mean_auc)),file = log_writter)
  log_writter.flush()

def segmentation_engine_0506(args, model_path, dataset_train, dataset_val, dataset_test,criterion):
  device = torch.device(args.device)
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  
  logs_path = os.path.join(model_path, "Logs")
  if not os.path.exists(logs_path):
    os.makedirs(logs_path)

  if os.path.exists(os.path.join(logs_path, "log.txt")):
    log_writter = open(os.path.join(logs_path, "log.txt"), 'a')
  else:
    log_writter = open(os.path.join(logs_path, "log.txt"), 'w')

  if args.mode == "train":
    start_num_epochs=0
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size,pin_memory=True,  shuffle=True,
                                                 num_workers=args.train_num_workers)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.train_batch_size,pin_memory=True, 
                                                      shuffle=False, num_workers=args.train_num_workers)
    if args.init.lower() == "random":
      print("init with scratch")
      model = smp.Unet(args.backbone, encoder_weights=None, activation=args.activate)
    elif args.init.lower() == "supervised":
      print("init with encoder from imagenet")
      model = smp.Unet(args.backbone, encoder_weights=args.proxy_dir.replace('supervised','imagenet'), activation=args.activate)

    elif args.init.lower() == "imagenet":
      print("init with encoder from imagenet")
      model = smp.Unet(args.backbone, encoder_weights=args.proxy_dir.replace('imagenet','supervised'), activation=args.activate)
      print(model)
    elif args.init.startswith('MAGICAL') or args.init.startswith('Genesis'):
      print("init with encoder + decoder")
      model = smp.Unet(args.backbone, encoder_weights=args.proxy_dir, activation=args.activate)
      
      checkpoint = torch.load(args.proxy_dir, map_location="cpu")
      if "state_dict" in checkpoint:
          state_dict = checkpoint["state_dict"]
      else:
          state_dict = checkpoint
      state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
      for k in list(state_dict.keys()):
          if k.startswith('projection') or k.startswith('segmentation'):
            del state_dict[k]
      _msg = model.load_state_dict(state_dict,strict = False)
      if len(_msg.missing_keys) != 0:
        print('_msg.missing_keys is :',_msg.missing_keys)
        if args.backbone.lower().startswith("resnet"):
          if args.activate is None:
              assert set(_msg.missing_keys) == {"segmentation_head.weight", "segmentation_head.bias"}
          else:
              assert set(_msg.missing_keys) == {"segmentation_head.0.weight", "segmentation_head.0.bias"}#{'segmentation_head.conv2d_3x3.weight', 'segmentation_head.conv2d_3x3.bias', 'segmentation_head.conv2d_1x1.weight', 'segmentation_head.conv2d_1x1.bias'}#
        elif args.backbone.lower().startswith("densenet"):
          if args.activate is None:
              assert set(_msg.missing_keys) == {"classifier.weight", "classifier.bias"}
          else:
              assert set(_msg.missing_keys) == {"classifier.0.weight", "classifier.0.bias"}
    else:
      print("init with encoder from proxy_dir")
      model = smp.Unet(args.backbone, encoder_weights=args.proxy_dir, activation=args.activate)
    
    print(args,file=log_writter)
    writer = SummaryWriter(log_dir=os.path.join(logs_path,args.data_set+'-'+args.run))
    
    if args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5, nesterov=False)
    elif args.optimizer.lower() == "adam":
      optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, betas=(0.9, 0.99))
    elif args.optimizer.lower() == "adamw":
      optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate)#, amsgrad=args.amsgrad)
    else:
      raise
    
    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)
    model.to(device)

    best_val_loss = 100000
    best_val_dice = 0
    patience_counter = 0

    if args.resume:
      resume = os.path.join(model_path, "best-checkpoint-{}.pth".format(args.run))
      if os.path.isfile(resume):
        print("=> loading resume checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)

        start_num_epochs = checkpoint['epoch']+1
        best_val_loss = checkpoint['lossMIN']
        # best_val_auc = checkpoint['aucMax']
        state_dict = checkpoint['state_dict']
        # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        lr_ = checkpoint['lr']
        #lr_scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch={:04d}, val_loss={:.5f})"
              .format(resume, start_num_epochs, best_val_loss))
      else:
        print("=> no checkpoint found at '{}'".format(resume))

    for epoch in range(start_num_epochs, args.epochs):
      train_one_epoch(data_loader_train,device, model, criterion, optimizer, epoch)
      val_iou,val_dice,val_loss = evaluate_seg(data_loader_val, device,model, epoch,criterion,writer)

      if val_dice > best_val_dice :#
        torch.save({
            'epoch': epoch,
            'lossMIN': val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
          },  os.path.join(model_path, "best-checkpoint-{}.pth".format(args.run)))
        print(
          "Epoch {:04d}:********** DICE improved from {:.5f} to ***{:.5f}***, val_loss is {:.5f}******".format(epoch, best_val_dice, val_dice,val_loss))
        print("Epoch {:04d}: DICE is {:.5f}, val_loss is {:.5f}***********".format(epoch,val_dice,val_loss), file=log_writter)
        best_val_dice = val_dice 
        # best_val_loss = val_loss                                                                                    # os.path.join(model_path,"checkpoint.pt")))
        patience_counter = 0

        # print(
        #   "Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(epoch, best_val_loss, val_loss,                                                                                      os.path.join(model_path,"checkpoint.pt")))
      else:
        lr_ = cosine_anneal_schedule(epoch,args.epochs,args.learning_rate)
        # for param_group in optimizer.param_groups:
        #   param_group['lr'] = lr_
        print("Epoch {:04d}: DICE NOT improved {:.5f}, val_loss is {:.5f}***********,lr is {:.5f}".format(epoch,val_dice,val_loss,lr_))
        print("Epoch {:04d}: DICE is {:.5f}, val_loss is {:.5f}***********".format(epoch,val_dice,val_loss), file=log_writter)
        patience_counter += 1


			

        if patience_counter > args.patience:
          print("Early Stopping")
          break

        #log_writter.flush()
  torch.cuda.empty_cache()

  data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False,
                                                num_workers=args.test_num_workers)
  model_save_path = os.path.join(model_path, "best-checkpoint.pth")
  model_list = glob(os.path.join(model_path,"best-checkpoint-*.pth"))
  for name in model_list:
    print("testing....",name)
    print("testing....",name,file=log_writter)
    DICE_test,IOU_test= test_segmentation_0214(model, name, data_loader_test, device, log_writter)

  log_writter.flush()


