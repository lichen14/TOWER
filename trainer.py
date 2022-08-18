from re import A
from torch.nn.modules.activation import Sigmoid
from utils import MetricLogger, ProgressLogger
from utils import dice_coef,accuracy,iou_score,torch_dice_coef_loss,dice,getAUC,getACC,draw_in_tensorboard
from models import ClassificationNet, Net
import time
import torch
from tqdm import tqdm
import sklearn.metrics as metrics
from sklearn.metrics._ranking import roc_auc_score,_binary_roc_auc_score
import matplotlib.pyplot as plt 
import numpy as np
import torch.nn.functional as F
def train_one_epoch(data_loader_train, device,model, criterion, optimizer, epoch):
  batch_time = MetricLogger('Time', ':6.3f')
  losses = MetricLogger('Loss', ':.4e')
  progress = ProgressLogger(
    len(data_loader_train),
    [batch_time, losses],
    prefix="Epoch: [{}]".format(epoch))

  model.train()

  end = time.time()
  for i, (samples, targets) in tqdm(enumerate(data_loader_train),total=len(data_loader_train)):
    samples, targets = samples.float().to(device), targets.float().to(device)

    outputs = model(samples)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.update(loss.item(), samples.size(0))
    batch_time.update(time.time() - end)
    end = time.time()


def train_classify(data_loader_train, device,model, criterion, optimizer, epoch,args):
  batch_time = MetricLogger('Time', ':6.3f')
  losses = MetricLogger('Loss', ':.4e')
  progress = ProgressLogger(
    len(data_loader_train),
    [batch_time, losses],
    prefix="Epoch: [{}]".format(epoch))

  model.train()

  end = time.time()
  for i, (samples, targets) in tqdm(enumerate(data_loader_train),total=len(data_loader_train)):
    if args.task == "multi-label, binary-class":
      samples, targets = samples.float().to(device), targets.float().to(device).squeeze(-1)
    else:
      samples, targets = samples.float().to(device), targets.long().to(device).squeeze(-1)
    outputs = model(samples)

    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.update(loss.item(), samples.size(0))
    batch_time.update(time.time() - end)
    end = time.time()



def train_classify_old(data_loader_train, device,model, criterion, optimizer, epoch,args):
  batch_time = MetricLogger('Time', ':6.3f')
  losses = MetricLogger('Loss', ':.4e')
  ACC = MetricLogger('ACC', ':6.3f')
  progress = ProgressLogger(
    len(data_loader_train),
    [batch_time, losses],
    prefix="Epoch: [{}]".format(epoch))

  model.train()

  end = time.time()
  for i, (samples, targets) in tqdm(enumerate(data_loader_train),total=len(data_loader_train)):
    samples, targets = samples.float().to(device), targets.float().to(device)#.unsqueeze(-1)
    outputs = model(samples)

    loss = criterion(outputs, targets)

    if args.activate is None:
      outputs = torch.nn.Sigmoid()(outputs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.update(loss.item(), samples.size(0))
    # ACC.update(acc_item.item())
    batch_time.update(time.time() - end)
    end = time.time()
  return losses.avg,ACC.avg


def evaluate_classify(model,data_loader_test, device, args):
  model.eval()

  y_test = torch.FloatTensor().cuda()
  p_test = torch.FloatTensor().cuda()

  with torch.no_grad():
    for i, (samples, targets) in enumerate(tqdm(data_loader_test)):
      samples, targets = samples.float().to(device), targets.long().to(device)
      
      y_test = torch.cat((y_test, targets), 0)
      if len(samples.size()) == 4:
        bs, c, h, w = samples.size()
        n_crops = 1
      elif len(samples.size()) == 5:
        bs, n_crops, c, h, w = samples.size()

      varInput = torch.autograd.Variable(samples.view(-1, c, h, w).cuda())

      out = model(varInput)
      
      if args.activate is None:
        out = torch.nn.Sigmoid()(out)
      outMean = out.view(bs, n_crops, -1).mean(1)
      p_test = torch.cat((p_test, outMean.data), 0)

  return y_test, p_test

def evaluate_txc(data_loader_val, device, model, criterion,args):
  model.eval()

  with torch.no_grad():
    Auc = MetricLogger('Auc', ':6.3f')
    DICE = MetricLogger('Dice', ':6.3f')
    losses = MetricLogger('Loss', ':.4e')
    progress = ProgressLogger(
      len(data_loader_val),
      [DICE, losses], prefix='Val: ')

    end = time.time()
    for i, (samples, targets) in enumerate(data_loader_val):

      samples, targets = samples.float().to(device), targets.float().to(device)#.unsqueeze(-1)

      outputs = model(samples)

      loss = criterion(outputs, targets)

      if args.activate is None:

        outputs = torch.nn.Sigmoid()(outputs)

      auc_item = roc_auc_score(targets.cpu().data, outputs.cpu().data)
      Auc.update(auc_item)

      losses.update(loss.item(), samples.size(0))
      
      end = time.time()


  return losses.avg,DICE.avg,Auc.avg

def evaluate(data_loader_val, device, model, criterion,args):
  model.eval()

  with torch.no_grad():
    Auc = MetricLogger('Auc', ':6.3f')
    DICE = MetricLogger('Dice', ':6.3f')
    losses = MetricLogger('Loss', ':.4e')
    progress = ProgressLogger(
      len(data_loader_val),
      [DICE, losses], prefix='Val: ')

    end = time.time()
    for i, (samples, targets) in enumerate(data_loader_val):
      if args.task == "multi-label, binary-class":
        samples, targets = samples.float().to(device), targets.float().to(device).squeeze(-1)
      else:
        samples, targets = samples.float().to(device), targets.long().to(device).squeeze(-1)


      outputs = model(samples)

      loss = criterion(outputs, targets)
      outputs = outputs.softmax(dim=-1)

      auc_item = getAUC(targets, outputs, args.task)
      # dice_item = getACC(targets, outputs, 'multi-label')
      Auc.update(auc_item)

      losses.update(loss.item(), samples.size(0))
      
      end = time.time()


  return losses.avg,DICE.avg,Auc.avg

def evaluate_classify_old(data_loader_val, device, model, criterion,args):
  model.eval()

  with torch.no_grad():
    Auc = MetricLogger('Auc', ':6.3f')
    DICE = MetricLogger('Dice', ':6.3f')
    losses = MetricLogger('Loss', ':.4e')
    progress = ProgressLogger(
      len(data_loader_val),
      [DICE, losses], prefix='Val: ')

    end = time.time()
    for i, (samples, targets) in enumerate(data_loader_val):
      samples, targets = samples.float().to(device), targets.float().to(device).squeeze(-1)  #medmnist


      outputs = model(samples)

      loss = criterion(outputs, targets)
      outputs = outputs.softmax(dim=-1)
      # print(outputs)
      if args.activate is None:

        outputs = torch.nn.Sigmoid()(outputs)

      auc_item = getAUC(targets, outputs, 'multi-label, binary-class')
      dice_item = getACC(targets, outputs, 'multi-label, binary-class')
      Auc.update(auc_item)
      DICE.update(dice_item)

      losses.update(loss.item(), samples.size(0))
      
      end = time.time()

      # if i % 50 == 0:
      #   progress.display(i)

  return losses.avg,DICE.avg,Auc.avg

def evaluate_seg(data_loader_val, device, model, epoch,criterion,writer):
  model.eval()

  with torch.no_grad():
    dices = MetricLogger('Dice', ':6.3f')
    ious = MetricLogger('IoU', ':6.3f')
    losses = MetricLogger('Loss', ':.4e')
    progress = ProgressLogger(
      len(data_loader_val),
      [ious, losses], prefix='Val: ')

    # end = time.time()
    for i, (samples, targets) in tqdm(enumerate(data_loader_val),total = len(data_loader_val)):
      samples, targets = samples.float().to(device), targets.float().to(device)

      outputs= model(samples)
 
      loss = criterion(outputs, targets)

      test_p = outputs.cpu().detach().numpy()#.squeeze(1)
      test_y = targets.cpu().detach().numpy()
      dice1 = dice(test_p, test_y)
      # iou = iou_score(test_p, test_y)
      # dice1 = dice_coef(outputs ,targets)
      # iou = iou_score(outputs ,targets)
      losses.update(loss.item(), samples.size(0))
      # losses.update(loss.item(), samples.size(0))
      dices.update(dice1)
      # ious.update(iou)


      writer.add_scalar(f'Valid-loss', loss.item(), epoch*len(data_loader_val)+i)
      writer.add_scalar(f'Valid-dice', dice1, epoch*len(data_loader_val)+i)
      # writer.add_scalar(f'Valid-iou', iou, epoch*len(data_loader_val)+i)
      # if i % 50 == 0:
      #   progress.display(i)

  return ious.avg, dices.avg ,losses.avg

def test(checkpoint, data_loader_test, device,args):
  model = ClassificationNet(args.model_name.lower(), args.num_class, activation=args.activate)
  #print(model)

  modelCheckpoint = torch.load(checkpoint)
  state_dict = modelCheckpoint['state_dict']
  state_dict = {k.replace("module.encoder_q.", ""): v for k, v in state_dict.items()}
  state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
  for k in list(state_dict.keys()):
    if k.startswith('encoder_k'):
      del state_dict[k]
      continue

    if k.startswith('queue') or  k.startswith('classifier') or k.startswith('projection_head') or k.startswith('decoder') or k.startswith('segmentation_head') or k.endswith('num_batches_tracked'):
      del state_dict[k]

  msg = model.load_state_dict(state_dict)
  assert len(msg.missing_keys) == 0
  print("=> loaded pre-trained model '{}'".format(checkpoint))
  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
  model.eval()
  model.to(device)

  with torch.no_grad():
    Auc = MetricLogger('Auc', ':6.3f')
    DICE = MetricLogger('Dice', ':6.3f')
    losses = MetricLogger('Loss', ':.4e')
    progress = ProgressLogger(
      len(data_loader_test),
      [DICE, losses], prefix='Val: ')

    end = time.time()
    for i, (samples, targets) in enumerate(tqdm(data_loader_test)):
      samples, targets = samples.float().to(device), targets.float().to(device)
      targets = F.one_hot(targets,args.num_class).float()
      outputs = model(samples)

      if args.activate is None:
        # print('valid in Sigmoid')
        outputs = torch.nn.Sigmoid()(outputs)

      if args.num_class > 1:
        for i in range(args.num_class):
          #print(i,targets.cpu().data[:, i], outputs.cpu().data[:, i])
          try:
            auc_item = roc_auc_score(targets.cpu().data[:, i], outputs.cpu().data[:, i]) #roc_auc_score
          except ValueError:
            continue

          Auc.update(auc_item)
      else:
        auc_item = roc_auc_score(targets.cpu().data, outputs.cpu().data)
        Auc.update(auc_item)
      
      end = time.time()

  return Auc.avg


def test_classification(checkpoint, data_loader_test, device, args):
  model = ClassificationNet(args.model_name.lower(), args.num_class, activation=args.activate)
  #print(model)

  modelCheckpoint = torch.load(checkpoint)
  state_dict = modelCheckpoint['state_dict']
  # used for clear the parallel trained params
  state_dict = {k.replace("module.encoder_q.", ""): v for k, v in state_dict.items()}
  state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
  for k in list(state_dict.keys()):
    if k.startswith('encoder_k'):
      del state_dict[k]
      continue

    if k.startswith('queue') or  k.startswith('classifier') or k.startswith('projection_head') or k.startswith('decoder') or k.startswith('segmentation_head') or k.endswith('num_batches_tracked'):
      del state_dict[k]



  msg = model.load_state_dict(state_dict)
  assert len(msg.missing_keys) == 0
  print("=> loaded pre-trained model '{}'".format(checkpoint))

  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
  model.to(device)

  model.eval()

  y_test = torch.FloatTensor().cuda()
  p_test = torch.FloatTensor().cuda()

  with torch.no_grad():
    for i, (samples, targets) in enumerate(tqdm(data_loader_test)):
      samples, targets = samples.float().to(device), targets.float().to(device)
      
      y_test = torch.cat((y_test, targets), 0)
      if len(samples.size()) == 4:
        bs, c, h, w = samples.size()
        n_crops = 1
      elif len(samples.size()) == 5:
        bs, n_crops, c, h, w = samples.size()

      varInput = torch.autograd.Variable(samples.view(-1, c, h, w).cuda())

      out = model(varInput)
      
      if args.activate is None:
        out = torch.nn.Sigmoid()(out)

      outMean = out.view(bs, n_crops, -1).mean(1)

      p_test = torch.cat((p_test, outMean.data), 0)

  return y_test, p_test

def test_visual(checkpoint, data_loader_test, device, args):
  model = ClassificationNet(args.model_name.lower(), args.num_class, activation=args.activate)
  # print(model)

  modelCheckpoint = torch.load(checkpoint)
  state_dict = modelCheckpoint['state_dict']
  # used for clear the parallel trained params?
  
  state_dict = {k.replace("module.encoder_q.", ""): v for k, v in state_dict.items()}
  state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
  for k in list(state_dict.keys()):
    if k.startswith('encoder_k'):
      del state_dict[k]
      continue

    if k.startswith('queue') or  k.startswith('classifier') or k.startswith('projection_head') or k.startswith('decoder') or k.startswith('segmentation_head') or k.endswith('num_batches_tracked'):
      del state_dict[k]

  msg = model.load_state_dict(state_dict, strict=False)
  # assert len(msg.missing_keys) == 0
  print("=> loaded pre-trained model '{}',msg is".format(checkpoint,msg.missing_keys))

  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
  model.to(device)

  model.eval()

  y_test = torch.FloatTensor().cuda()
  features = torch.FloatTensor().cuda()

  with torch.no_grad():
    Auc = MetricLogger('Auc', ':6.3f')
    Acc = MetricLogger('Dice', ':6.3f')
    for i, (samples, targets) in enumerate(tqdm(data_loader_test)):

      if len(samples.size()) == 4:
        bs, c, h, w = samples.size()
        n_crops = 1
      elif len(samples.size()) == 5:
        bs, n_crops, c, h, w = samples.size()

      varInput = torch.autograd.Variable(samples.view(-1, c, h, w).cuda())

      out,fea = model(varInput)
      index = np.where(targets>0)[1]  #col index
      # print(targets,targets[0,2])
      features = torch.cat((features, fea.data), 0)
      tmp = torch.from_numpy(np.array(targets).astype('uint8')).cuda()#.unsqueeze(0)
      # plt.subplot(222)
      # plt.imshow(np.transpose(samples.cpu().data[0],(1,2,0)))
      # plt.show()
      y_test = torch.cat((y_test, tmp), 0)

      targets = targets.long().to(device).squeeze(-1) 
      
      print(y_test.shape,features.shape)
  return y_test, features, Auc.avg

def test_classification_0412(checkpoint, data_loader_test, device, args):


  model = ClassificationNet(args.model_name.lower(), args.num_class, activation=args.activate)


  modelCheckpoint = torch.load(checkpoint)
  state_dict = modelCheckpoint['state_dict']
  state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
  msg = model.load_state_dict(state_dict, strict=True)


  print("=> loaded pre-trained model '{}',msg is".format(checkpoint,msg.missing_keys))

  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
  model.to(device)

  model.eval()

  sample = torch.FloatTensor().cuda()
  features = torch.FloatTensor().cuda()

  with torch.no_grad():
    Auc = MetricLogger('Auc', ':6.3f')
    Acc = MetricLogger('Dice', ':6.3f')
    for i, (samples, targets) in enumerate(tqdm(data_loader_test)):

      samples, targets = samples.float().to(device), targets.long().to(device).squeeze(-1)  #medmnist
      # print(samples.shape,targets.shape)  #torch.Size([1, 3, 224, 224]) torch.Size([1, 8])


      out= model(samples)

      outputs = out.softmax(dim=-1)
      # print(outputs)
      auc_item = getAUC(targets, outputs, args.task)
      # print(auc_item)
      # acc_item = getACC(targets, outputs, 'multi-label')
      Auc.update(auc_item)
  return sample, features, Auc.avg

def test_segmentation(model, model_save_path,data_loader_test, device,log_writter):
    print("testing....", file=log_writter)
    checkpoint = torch.load(model_save_path)
    state_dict = checkpoint["state_dict"]
    # used for clear the parallel trained params??
    for k in list(state_dict.keys()):
      if k.startswith('module.'):
        state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict)
    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)
    model.to(device)
    with torch.no_grad():
        test_p = None
        test_y = None
        model.eval()
        for batch_ndx, (x_t, y_t) in enumerate(tqdm(data_loader_test)):
            x_t, y_t = x_t.float().to(device), y_t.float().to(device)
            pred_t = model(x_t)
            if test_p is None and test_y is None:
                test_p = pred_t
                test_y = y_t
            else:
                test_p = torch.cat((test_p, pred_t), 0)
                test_y = torch.cat((test_y, y_t), 0)

            if (batch_ndx + 1) % 5 == 0:
                print("Testing Step[{}]: ".format(batch_ndx + 1) , file=log_writter)
                log_writter.flush()

        print("Done testing iteration!", file=log_writter)
        log_writter.flush()

    test_p = test_p.cpu().detach().numpy()
    test_y = test_y.cpu().detach().numpy()

    return test_y, test_p


def test_segmentation_0214(model, model_save_path,data_loader_test, device,log_writter):
    # print("testing....", file=log_writter)
    checkpoint = torch.load(model_save_path)
    state_dict = checkpoint["state_dict"]

    model.load_state_dict(state_dict)
    # if torch.cuda.device_count() > 1:
    #   model = torch.nn.DataParallel(model)
    # model.to(device)
    with torch.no_grad():
        DICE_test = []
        IOU_test = []
        test_p = None
        test_y = None
        model.eval()
        for batch_ndx, (x_t, y_t) in enumerate(tqdm(data_loader_test)):
            x_t, y_t = x_t.float().to(device), y_t.float().to(device)
            pred_t = model(x_t)
            if test_p is None and test_y is None:
                test_p = pred_t
                test_y = y_t
            else:
                test_p = torch.cat((test_p, pred_t), 0)
                test_y = torch.cat((test_y, y_t), 0)

            if (batch_ndx + 1) % 5 == 0:
                # print("Testing Step[{}]: ".format(batch_ndx + 1) , file=log_writter)
                log_writter.flush()

        log_writter.flush()
    test_p = test_p.cpu().detach().numpy()#.squeeze(1)
    test_y = test_y.cpu().detach().numpy()
    DICE=100.0 * dice(test_p, test_y)
    IOU=100.0 * iou_score(test_p, test_y)
    if DICE>70:
      print(">>{}: AUC = {:.2f}".format(model_save_path, 100.0 * dice(test_p, test_y)))
      print("{}: AUC = {:.2f}".format(model_save_path, 100.0 * dice(test_p, test_y)),file=log_writter)
    return DICE,IOU