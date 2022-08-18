import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics._ranking import roc_auc_score

import torch
import torch.nn as nn
import torchvision.models as models

import resnet_wider
import densenet



def ClassificationNet(arch_name, num_class, conv=None, weight=None, activation=None):
    if weight is None:
        print("=> loaded RANDOM model")
        weight = "none"

    if conv is None:
        try:
            model = resnet_wider.__dict__[arch_name](sobel=False)
        except:
            model = models.__dict__[arch_name](pretrained=False)
    else:
        if arch_name.lower().startswith("resnet"):
            model = resnet_wider.__dict__[arch_name + "_layerwise"](conv, sobel=False)
        elif arch_name.lower().startswith("densenet"):
            model = densenet.__dict__[arch_name + "_layerwise"](conv)

    if arch_name.lower().startswith("resnet"):
        kernelCount = model.fc.in_features
        if activation is None:
            model.fc = nn.Linear(kernelCount, num_class)
        elif activation == "sigmoid":
            model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
        elif activation == "relu":
            model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.ReLU())


        # init the fc layer
        if activation is None:
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()
        else:
            model.fc[0].weight.data.normal_(mean=0.0, std=0.01)
            model.fc[0].bias.data.zero_()
    elif arch_name.lower().startswith("densenet"):
        kernelCount = model.classifier.in_features
        if activation is None:
            model.classifier = nn.Linear(kernelCount, num_class)
        elif activation == "sigmoid":
            model.classifier = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
        elif activation == "relu":
            model.classifier = nn.Sequential(nn.Linear(kernelCount, num_class), nn.ReLU())

        # init the classifier layer
        if activation is None:
            model.classifier.weight.data.normal_(mean=0.0, std=0.01)
            model.classifier.bias.data.zero_()
        else:
            model.classifier[0].weight.data.normal_(mean=0.0, std=0.01)
            model.classifier[0].bias.data.zero_()
    # print(model)
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

    if weight.lower().endswith("imagenet.pth"):
        pretrained_model = models.__dict__[arch_name](pretrained=True)
        state_dict = pretrained_model.state_dict()
        # delete fc layer
        for k in list(state_dict.keys()):
            if k.startswith('fc') or k.startswith('classifier'):
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print('from imagenet',msg)
        _weight_loading_check(arch_name, activation, msg)
        print("=> loaded supervised ImageNet pre-trained model")
    elif os.path.isfile(weight):
        checkpoint = torch.load(weight, map_location="cpu")
        #print(checkpoint)
        #
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            print('model')
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        #print(state_dict.keys())
        # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("module.encoder_q.", ""): v for k, v in state_dict.items()}

        # for k in list(state_dict.keys()):
        #     if k.startswith('fc') or k.startswith('classifier') or k.startswith('projection_head') or k.startswith('prototypes'):
        #         del state_dict[k]

        # msg = model.load_state_dict(state_dict, strict=False)
        # _weight_loading_check(arch_name, activation, msg)
        # print("=> loaded pre-trained model '{}'".format(weight))
        # print("missing keys:", msg.missing_keys)
        # print(state_dict)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("base_", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("l_to_ab.", ""): v for k, v in state_dict.items()}
        #print(state_dict)
        for k in list(state_dict.keys()):
            #print(k)
            # if k.endswith('num_batches_tracked'):
            #     del state_dict[k]
            if weight.find('moco'):
                if k.startswith('momentum') or k.startswith('predictor') or k.startswith('encoder_k') or k.startswith('fc'):
                    del state_dict[k]
                continue
            if k.find('ab_to_l.'):
                print(k)
                del state_dict[k]
                continue        
            if k.startswith('queue') or k.startswith('fc') or k.startswith('classifier') or k.startswith('projection_head') or k.startswith('decoder') or k.startswith('segmentation_head') or k.endswith('num_batches_tracked'):
                del state_dict[k]
            
        msg = model.load_state_dict(state_dict, strict=False)
        #print("missing keys:", msg.missing_keys)
        _weight_loading_check(arch_name, activation, msg)
        print("=> loaded pre-trained model '{}'".format(weight))
        print("missing keys:", msg.missing_keys)


    # reinitialize fc layer again
    if arch_name.lower().startswith("resnet"):
        if activation is None:
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()
        else:
            model.fc[0].weight.data.normal_(mean=0.0, std=0.01)
            model.fc[0].bias.data.zero_()
    elif arch_name.lower().startswith("densenet"):
        if activation is None:
            model.classifier.weight.data.normal_(mean=0.0, std=0.01)
            model.classifier.bias.data.zero_()
        else:
            model.classifier[0].weight.data.normal_(mean=0.0, std=0.01)

    return model


def build_classification_model(args):
    # model = Net(in_channels=args.n_channels, num_classes=args.num_class)
    if args.init.lower() =="random":
        model = ClassificationNet(args.model_name.lower(), args.num_class, weight=None,
                              activation=args.activate)

    else:
        print(args.proxy_dir)
        model = ClassificationNet(args.model_name.lower(), args.num_class, weight=args.proxy_dir,
                              activation=args.activate)


    return model

def save_checkpoint(state,filename='model'):

    torch.save( state,filename + '.pth.tar')



class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
