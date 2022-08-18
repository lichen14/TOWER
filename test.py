
import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from tqdm import tqdm
import copy


from models import build_classification_model, save_checkpoint
from utils import metric_AUROC
from glob import glob
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from trainer import train_one_epoch,test_classification,evaluate,test_segmentation
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
                                   num_workers=args.workers, pin_memory=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
    log_file = os.path.join(model_path, "models.log")

    # training phase
    print("start training....")
    for i in range(args.start_index, args.num_trial):
      print ("run:",str(i+1))
      start_epoch = 0
      init_loss = 1000000
      experiment = args.exp_name + "_run_" + str(i)
      best_val_loss = init_loss
      patience_counter = 0
      save_model_path = os.path.join(model_path, experiment)
      criterion = torch.nn.BCELoss()

      model = build_classification_model(args)
      # print(model)

      if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
      model.to(device)

      parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

      optimizer = torch.optim.Adam(parameters, lr=args.lr)
      lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=args.patience // 2, mode='min',
                                       threshold=0.0001, min_lr=0, verbose=True)


      if args.resume:
        resume = os.path.join(model_path, experiment + '.pth.tar')
        if os.path.isfile(resume):
          print("=> loading checkpoint '{}'".format(resume))
          checkpoint = torch.load(resume)

          start_epoch = checkpoint['epoch']
          init_loss = checkpoint['lossMIN']
          model.load_state_dict(checkpoint['state_dict'])
          lr_scheduler.load_state_dict(checkpoint['scheduler'])
          optimizer.load_state_dict(checkpoint['optimizer'])
          print("=> loaded checkpoint '{}' (epoch={:04d}, val_loss={:.5f})"
                .format(resume, start_epoch, init_loss))
        else:
          print("=> no checkpoint found at '{}'".format(args.resume))



      for epoch in range(start_epoch, args.num_epoch):
        train_one_epoch(data_loader_train,device, model, criterion, optimizer, epoch)

        val_loss = evaluate(data_loader_val, device,model, criterion)

        lr_scheduler.step(val_loss)

        if val_loss < best_val_loss:
          save_checkpoint({
            'epoch': epoch + 1,
            'lossMIN': best_val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
          },  filename=save_model_path)

          best_val_loss = val_loss
          patience_counter = 0

          print(
            "Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(epoch, best_val_loss, val_loss,
                                                                                               save_model_path))

        else:
          print("Epoch {:04d}: val_loss did not improve from {:.5f} ".format(epoch, best_val_loss ))
          patience_counter += 1

        if patience_counter > args.patience:
          print("Early Stopping")
          break


      # log experiment
      with open(log_file, 'a') as f:
        f.write(experiment + "\n")
        f.close()

  print ("start testing.....")
  output_file = os.path.join(output_path, args.exp_name + "_results.txt")

  data_loader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)

  log_file = os.path.join(model_path, "models.log")
  if not os.path.isfile(log_file):
    print("log_file ({}) not exists!".format(log_file))
  else:
    mean_auc = []
    with open(log_file, 'r') as reader, open(output_file, 'a') as writer:
      experiment = reader.readline()
      print(">> Disease = {}".format(diseases))
      writer.write("Disease = {}\n".format(diseases))

      while experiment:
        experiment = experiment.replace('\n', '')
        saved_model = os.path.join(model_path, experiment + ".pth.tar")

        y_test, p_test = test_classification(saved_model, data_loader_test, device, args)

        if test_diseases is not None:
          y_test = copy.deepcopy(y_test[:,test_diseases])
          p_test = copy.deepcopy(p_test[:, test_diseases])
          individual_results = metric_AUROC(y_test, p_test, len(test_diseases))
        else:
          individual_results = metric_AUROC(y_test, p_test, args.num_class)
        print(">>{}: AUC = {}".format(experiment, np.array2string(np.array(individual_results), precision=4, separator=',')))
        writer.write(
          "{}: AUC = {}\n".format(experiment, np.array2string(np.array(individual_results), precision=4, separator='\t')))


        mean_over_all_classes = np.array(individual_results).mean()
        print(">>{}: AUC = {:.4f}".format(experiment, mean_over_all_classes))
        writer.write("{}: AUC = {:.4f}\n".format(experiment, mean_over_all_classes))

        mean_auc.append(mean_over_all_classes)
        experiment = reader.readline()

      mean_auc = np.array(mean_auc)
      print(">> All trials: mAUC  = {}".format(np.array2string(mean_auc, precision=4, separator=',')))
      writer.write("All trials: mAUC  = {}\n".format(np.array2string(mean_auc, precision=4, separator='\t')))
      print(">> Mean AUC over All trials: = {:.4f}".format(np.mean(mean_auc)))
      writer.write("Mean AUC over All trials = {:.4f}\n".format(np.mean(mean_auc)))
      print(">> STD over All trials:  = {:.4f}".format(np.std(mean_auc)))
      writer.write("STD over All trials:  = {:.4f}\n".format(np.std(mean_auc)))

def segmentation_engine(args, model_path, dataset_train, dataset_val, dataset_test,criterion):
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
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True,
                                                 num_workers=args.train_num_workers)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.train_batch_size,
                                                      shuffle=False, num_workers=args.train_num_workers)
    if args.init is None:
      print("init with scratch")
      model = smp.Unet(args.backbone, encoder_weights=None, activation=args.activate)
    elif args.init.lower() == "imagenet":
      print("init with encoder from imagenet")
      model = smp.Unet(args.backbone, encoder_weights=args.init, activation=args.activate)
    elif args.init.startswith('MG0'):
      print("init with encoder + decoder")
      model = smp.Unet(args.backbone, encoder_weights=None, activation=args.activate)
      checkpoint = torch.load(args.proxy_dir, map_location="cpu")
      if "state_dict" in checkpoint:
          state_dict = checkpoint["state_dict"]
      else:
          state_dict = checkpoint
      # print('my test-MG',state_dict.keys(), file=log_writter)
      state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
      for k in list(state_dict.keys()):
          if k.startswith('segmentation_head'):
            # print(k)
            del state_dict[k]
      # print('after del',state_dict.keys(), file=log_writter)
      _msg = model.load_state_dict(state_dict,strict = False)
      if len(_msg.missing_keys) != 0:
        if args.backbone.lower().startswith("resnet"):
          if args.activate is None:
              assert set(_msg.missing_keys) == {"segmentation_head.weight", "segmentation_head.bias"}
          else:
              assert set(_msg.missing_keys) == {"segmentation_head.0.weight", "segmentation_head.0.bias"}
        elif args.backbone.lower().startswith("densenet"):
          if args.activate is None:
              assert set(_msg.missing_keys) == {"classifier.weight", "classifier.bias"}
          else:
              assert set(_msg.missing_keys) == {"classifier.0.weight", "classifier.0.bias"}
    else:
      print("init with encoder from proxy_dir")
      model = smp.Unet(args.backbone, encoder_weights=args.proxy_dir, activation=args.activate)

    # optimizer = torch.optim.Adam(model.parameters(), conf.lr, betas=(0.9, 0.99))
    optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)
    model.to(device)
    best_val_loss = 100000
    best_val_dice = 0
    patience_counter = 0

  #   for epoch in range(start_num_epochs, args.epochs):
  #     train_one_epoch(data_loader_train,device, model, criterion, optimizer, epoch)
  #     val_loss,val_dice = evaluate(data_loader_val, device,model, criterion)
  #     # update learning rate
  #     lr_ = cosine_anneal_schedule(epoch,args.epochs,args.learning_rate)
  #     for param_group in optimizer.param_groups:
  #       param_group['lr'] = lr_
  #     if val_dice > best_val_dice :#val_loss < best_val_loss:
  #       torch.save({
  #         'epoch': epoch + 1,
  #         'lossMIN': best_val_loss,
  #         'state_dict': model.state_dict(),
  #         'optimizer': optimizer.state_dict(),
  #       },  os.path.join(model_path, "best-checkpoint.pt"))
  #       print(
  #         "Epoch {:04d}:********** dice improved from {:.5f} to ***{:.5f}***, current lr is {:.5f}***********".format(epoch, best_val_dice, val_dice,lr_))
  #       best_val_dice = val_dice                                                                                    # os.path.join(model_path,"checkpoint.pt")))
  #       best_val_loss = val_loss
  #       patience_counter = 0

  #       # print(
  #       #   "Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(epoch, best_val_loss, val_loss,                                                                                      os.path.join(model_path,"checkpoint.pt")))
  #     else:
  #       print("Epoch {:04d}: val_loss {:.5f} did not improve from {:.5f}, current lr is {:.5f}".format(epoch,val_loss, best_val_loss,lr_ ))
  #       patience_counter += 1

  #     if patience_counter > args.patience:
  #       print("Early Stopping")
  #       break

  #     log_writter.flush()
  # torch.cuda.empty_cache()
  model_path = os.path.join("./Models/Segmentation", args.data_set, args.arch, args.backbone, args.init,str(args.run))

  data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False,
                                                num_workers=args.test_num_workers)
  # print(glob('./BenchmarkTransferLearning-main/Models/Segmentation/DRIVE/unet/resnet50/MAGICAL-1127-1/6/*'))
  for name in glob(os.path.join(model_path, "*.pth")):
    # print(name)
    test_y, test_p = test_segmentation(model, name, data_loader_test, device, log_writter)
    Dice = 100.0 * dice(test_p, test_y)
    mean_dice = mean_dice_coef(test_y > 0.5, test_p > 0.5)
    if Dice>=78:
      print("[INFO] epoch {} Dice = {:.2f}%".format(os.path.basename(name),Dice))#, file=log_writter)
      print("epoch {} Mean Dice = {:.4f}".format(os.path.basename(name),mean_dice))#, file=log_writter)
      print("[INFO] epoch {} Dice = {:.2f}%".format(os.path.basename(name),Dice), file=log_writter)
      print("epoch {} Mean Dice = {:.4f}".format(os.path.basename(name),mean_dice), file=log_writter)
  log_writter.flush()

def _weight_loading_check(_arch_name, _activation, _msg):
  if len(_msg.missing_keys) != 0:
    if _arch_name.lower().startswith("resnet"):
      if _activation is None:
          assert set(_msg.missing_keys) == {"fc.weight", "fc.bias"}
      else:
          assert set(_msg.missing_keys) == {"fc.0.weight", "fc.0.bias"}
    elif _arch_name.lower().startswith("densenet"):
      if _activation is None:
          assert set(_msg.missing_keys) == {"classifier.weight", "classifier.bias"}
      else:
          assert set(_msg.missing_keys) == {"classifier.0.weight", "classifier.0.bias"}
