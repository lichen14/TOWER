from sched import scheduler
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import cosine_anneal_schedule
import medmnist
from medmnist import INFO, Evaluator

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

data_flag = 'octmnist'
# data_flag = 'breastmnist'
download = True

NUM_EPOCHS = 100
BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
# print(n_channels)
DataClass = getattr(medmnist, info['python_class'])
# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, as_rgb=True, download=False,root='./datasets/MedMNIST-main')
test_dataset = DataClass(split='test', transform=data_transform, as_rgb=True, download=False,root='./datasets/MedMNIST-main')
valid_dataset = DataClass(split='val', transform=data_transform, as_rgb=True, download=False,root='./datasets/MedMNIST-main')

pil_dataset = DataClass(split='train', download=False,root='./datasets/MedMNIST-main')

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

# print(train_dataset)
# print("===================")
print(len(train_loader),len(train_loader_at_eval),len(test_loader))

def test(split):
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    
    data_loader = train_loader_at_eval if split == 'val' else test_loader
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs.cuda())

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).cuda()
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long().cuda()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true.cuda(), targets), 0)
            y_score = torch.cat((y_score.cuda(), outputs), 0)

        print(y_true.shape,y_score.shape)
        y_true = y_true.cpu().data.numpy()
        y_score = y_score.cpu().data.numpy()

        evaluator = Evaluator(data_flag, split,root='./datasets/MedMNIST-main')
        # metrics = evaluator.evaluate_new(y_true,y_score)
        metrics = evaluator.evaluate(y_score)
    
        print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))


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

# model = Net(in_channels=n_channels, num_classes=n_classes)
import torchvision.models as models 
model = models.resnet18(pretrained=True)
# change the input channel at the first layer
# model.conv1 = torch.nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# change the last linear layer
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, n_classes) # 15?4 output classes

print(model)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.cuda()

# define loss function and optimizer
if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
    
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,75],gamma=0.1)#, last_epoch=-1)

for epoch in range(NUM_EPOCHS):
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0
    # lr_ = cosine_anneal_schedule(epoch,NUM_EPOCHS,lr)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr_
    model.train()
    for inputs, targets in tqdm(train_loader):
        # forward + backward + optimize
        # print(inputs.shape,targets.shape)   #torch.Size([128, 3, 28, 28]) torch.Size([128, 1])
        optimizer.zero_grad()
        outputs = model(inputs.cuda())
        
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).cuda()
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long().cuda()
            loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()

    print('@ Epoch :', epoch,'==> Evaluating ... @ learning rate = ',scheduler.get_last_lr())
    test('val')
    test('test')
    scheduler.step()
# evaluation


        
print('==> Evaluating ...')
# test('test')
# test('val')