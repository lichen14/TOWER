from sklearn.metrics._ranking import roc_auc_score,_binary_roc_auc_score
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from torchvision.utils import make_grid
class FocalLoss(torch.nn.Module):
    
    def __init__(self, device, gamma = 1.0):
        super(FocalLoss, self).__init__()
        self.device = device
        self.gamma = torch.tensor(gamma, dtype = torch.float32).to(device)
        self.eps = 1e-6
        
#         self.BCEW_loss = BCE + sigmoid
        
    def forward(self, input, target):
        
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none').to(self.device)
        # mask = torch.ones_like(BCE_loss)
        # mask[(input>0.5) & (target >0)]=0
        # mask[(input<0.5) & (target <1)]=0
        #print(mask)
#         BCE_loss = self.BCE_loss(input, target)
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss =  (1-pt)**self.gamma * BCE_loss# * mask)
        
        return F_loss.mean()

class MetricLogger(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressLogger(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def metric_AUROC(target, output, nb_classes=14):
    outAUROC = []

    # target = target.cpu().numpy()
    # output = output.cpu().numpy()

    for i in range(nb_classes):
        outAUROC.append(roc_auc_score(target.cpu().data[:, i], output.cpu().data[:, i]))

    return outAUROC


def vararg_callback_bool(option, opt_str, value, parser):
    assert value is None

    arg = parser.rargs[0]
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        value = True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        value = False

    del parser.rargs[:1]
    setattr(parser.values, option.dest, value)


def vararg_callback_int(option, opt_str, value, parser):
    assert value is None
    value = []

    def intable(str):
        try:
            int(str)
            return True
        except ValueError:
            return False

    for arg in parser.rargs:
        # stop on --foo like options
        if arg[:2] == "--" and len(arg) > 2:
            break
        # stop on -a, but not on -3 or -3.0
        if arg[:1] == "-" and len(arg) > 1 and not intable(arg):
            break
        value.append(int(arg))

    del parser.rargs[:len(value)]
    setattr(parser.values, option.dest, value)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))

def getAUC(y_true, y_score, task):
    '''AUC metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    '''
    # y_true = F.one_hot(y_true,9).float()
    y_true = y_true.squeeze().cpu().data.numpy()
    y_score = y_score.squeeze().cpu().data.numpy()
    # print(y_true,y_score)
    if task == 'multi-label, binary-class':
        auc = 0
        for i in range(y_score.shape[1]):
            label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
            auc += label_auc
        ret = auc / y_score.shape[1]
    elif task == 'binary-class':
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = roc_auc_score(y_true, y_score)
    else:
        # auc = 0
        # sum=0
        # # y_true = F.one_hot(y_true,9).float()
        # # for i in range(y_score.shape[1]):
        # #     # print('y_true',y_true)
        # #     # y_true_binary = (y_true == i).cpu().data.numpy().astype(float)
        # #     # print('y_true_binary',y_true.shape)
        # #     y_score_binary = y_score[:, i].cpu().data
        # #     y_true_binary = y_true[:, i].cpu().data
        # #     # print(y_true_binary,y_score_binary)
        # #     try:
        # #         auc += roc_auc_score(y_true_binary, y_score_binary)
        # #         sum+=1
        # #     except ValueError:
        # #         print('pass')
        # #         pass
        #     # print(i)
        #     # y_true_binary = F.one_hot(y_true == i).float()
        # #     y_true_binary = (y_true == i)#.astype(float)
        # # y_true_binary = F.one_hot(y_true,9).float()
        # # # print(y_true_binary)
        # # # y_score_binary = y_score[:, i]
        # # # print(y_score)
        # try:
        #     auc += roc_auc_score(y_true.cpu().data, y_score.cpu().data)
        # except ValueError:
        #     print('pass')
        #     pass
        # ret = auc #/sum#y_score.shape[1]
        auc = 0
        for i in range(y_score.shape[1]):
            y_true_binary = (y_true == i).astype(float)
            y_score_binary = y_score[:, i]
            # print(y_true_binary, y_score_binary)
            auc += roc_auc_score(y_true_binary, y_score_binary)
        ret = auc / y_score.shape[1]

    return ret


def getACC(y_true, y_score, task, threshold=0.5):
    '''Accuracy metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    :param threshold: the threshold for multilabel and binary-class tasks
    '''
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()#torch.argmax(y_score, axis=-1).squeeze()
    # print(y_true.shape)
    # print(y_score.shape)
    if task == 'multi-label, binary-class':
        y_pre = y_score > threshold
        acc = 0
        for label in range(y_true.shape[1]):
            print(label)
            label_acc = accuracy_score(y_true.cpu().data[:, label], y_pre.cpu().data[:, label])
            acc += label_acc
        ret = acc / y_true.shape[1]
    elif task == 'binary-class':
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = accuracy_score(y_true, y_score > threshold)
    else:
        ret = accuracy_score(y_true.data.cpu(), y_score.data.cpu())

    return ret

def dice_coef(output_, target_):
    smooth = 1e-5
    # print(output_[10,200,200])
    # # print(target.shape)
    # print(target_[10,200,200])
    # output= torch.from_numpy(output)
    # print(target)
    # output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    # target = target.view(-1).data.cpu().numpy()
    # target = target.astype('float32') * 255
    # output = output.astype('float32') * 255
    # if torch.is_tensor(output):
    #     output = torch.sigmoid(output).data.cpu().numpy()
    # if torch.is_tensor(target):
    #     target = target.data.cpu().numpy()

    # 1227
    output_ = output_ > 0.5
    target_ = target_ > 0
    #
    # t=output[0,10,:,:].data.cpu().numpy()
    # # print(t.shape)
    # # tt=np.zeros((256,256,3))
    # # tt[:,:,0]=t[0,:,:]
    # # # tt[:,:,1]=t[0,:,:]
    # # # tt[:,:,2]=t[1,:,:]
    # # print(tt[0,100,100])
    # # # m=target[1,:,:,:]
    # # mm=np.zeros((512,512))
    # mm=target[0,10,:,:].data.cpu().numpy()
    # # # mm[:,:,1]=target[1,:,:]
    # # # mm[:,:,2]=target[2,:,:]
    # # # t = t.transpose((1,2,0))
    # # # print(tt.shape)
    # plt.subplot(121)
    # plt.imshow(t) # 显示图片
    # plt.subplot(122)
    # plt.imshow(mm, cmap='Greys_r') # 显示图片
    # # plt.axis('off') # 不显示坐标轴
    # plt.show()

    intersection = (output_ * target_).sum()
    # intersection = torch.sum((output_ + target_)==2)#(output_ * target_).sum()
    # print(intersection)
    # print(output.sum())
    # print(target.sum())
    return (2. * intersection + smooth) / \
        (output_.sum() + target_.sum() + smooth)

def iou_score(im1, im2):
    overlap = (im1>0.5) * (im2>0.5)
    union = (im1>0.5) + (im2>0.5)
    return overlap.sum()/float(union.sum())

def cosine_anneal_schedule(t,epochs,learning_rate):
    T=epochs
    M=1
    alpha_zero = learning_rate

    cos_inner = np.pi * (t % (T // M))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    return float(alpha_zero / 2 * cos_out)

def dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1 > 0.5).astype(np.bool)
    im2 = np.asarray(im2 > 0.5).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

def accuracy(output, target):
    output = output.view(-1).data.cpu().numpy()
    output = (np.round(output)).astype('int')
    target = target.view(-1).data.cpu().numpy()
    target = (np.round(target)).astype('int')

    output = output[target==1]
    target = target[target==1]
    (output == target).sum()

    return (output == target).sum() / len(output)

def mean_dice_coef(y_true,y_pred):
    sum=0
    for i in range (y_true.shape[0]):
        sum += dice(y_true[i,:,:,:],y_pred[i,:,:,:])
    return sum/y_true.shape[0]

def draw_in_tensorboard(writer, target_labels, i_iter, images, pred_main,type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(pred_main[:1].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)

    # print('target_labels:',target_labels.shape)#[1, 256, 256]
    grid_image = make_grid(target_labels[:1].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'GroundTrue - {type_}', grid_image, i_iter)