import torch
import torchvision
import torchvision.transforms as transforms

#setup training set
#transforming the PIL Image to tensors
batch_size = 4
max_epoch = 2

trainset = torchvision.datasets.FashionMNIST(
    root = "./data", 
    train = True, 
    download = True, 
    transform = transforms.ToTensor()
    )

trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=batch_size, 
    shuffle = True
    )

testset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
    )

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False
    )

classes = (
    'T-Shirt',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle Boot'
    )