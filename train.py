import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import numpy as np
from torchvision.models import ResNet50_Weights

def load_split_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([
                                       transforms.ToTensor()
                                       ])
    test_transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                      ])                                                                                                                             
    train_data = datasets.ImageFolder(datadir,       
                    transform=train_transforms)
    test_data = datasets.ImageFolder(datadir,
                    transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)                             
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=3200, num_workers = 4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_data,                  
                   sampler=test_sampler, batch_size=3200, num_workers = 4, pin_memory=True)
    return trainloader, testloader

def eval_on_test_set(model, testloader, device):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            outputs = model(images)
            _,predicted = torch.max(outputs.data, 1)
            predicted_correctly_on_epoch += (labels == predicted).sum().item()
    epoch_acc = 100.00 * predicted_correctly_on_epoch/total
    print(f"  -Testing dataset. Got {predicted_correctly_on_epoch} out of {total} images correctly {epoch_acc}")
    return epoch_acc

def save_checkpoint(model, epoch, optimizer, best_acc):
    state={
    'epoch': epoch,
    'model' : model.state_dict(),
    'best accuracy': best_acc,
    'optimizer': optimizer.state_dict()
    }
    torch.save(state, 'model_best_checkpoint.pth.tar')
    torch.save(model, 'best_model.pth')

def train_nn(model, trainloader, testloader, loss_fn, optimizer, device, epochs):
    best_acc = 0
    for epoch in range(epochs):
        print(f"Epoch Number {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0
        for data in trainloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()
        epoch_loss = running_loss/len(trainloader)
        epoch_acc = 100.00 * running_correct/total
        print(f"  -Training dataset. Got {running_correct} out of {total} images correctly {epoch_acc}%. Epoch loss {epoch_loss}")
        eval_acc = eval_on_test_set(model, testloader, device)
        if eval_acc > best_acc:
            best_acc = eval_acc 
            save_checkpoint(model, epoch, optimizer, best_acc)
    print("done")
    return model 

def main():
    training_dataset_path = './hw4_train'
    trainloader, testloader = load_split_test(training_dataset_path, .2)
    device = torch.device("cuda" if torch.cuda.is_available() 
                                    else "cpu")
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    number_of_classes = 10
    model.fc = nn.Linear(num_ftrs, number_of_classes)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.05, momentum = 0.9, weight_decay = 0.005)
    train_nn(model, trainloader, testloader, loss_fn, optimizer, device, 50)

if __name__ == "__main__":
    main()