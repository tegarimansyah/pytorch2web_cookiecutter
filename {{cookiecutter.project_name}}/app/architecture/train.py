import torch
import torchvision
import logging

from app.architecture import trainloader, testloader, max_epoch
from app.architecture.NN import Net

class Train():
    def __init__(self):
        self.net = Net()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=0.001, 
            momentum=0.9
            )
        
    def run(self):
        for epoch in range(max_epoch):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    logging.info('Epoch: %d, mini-batch: %d, loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        torch.save(self.net.state_dict(), './data/pytorch2web_model.pt')
        logging.info('Finished Training')


def test_train():
    train = Train()
    train.run()
    pass