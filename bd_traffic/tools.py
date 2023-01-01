import torch
from torch import nn
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import torchvision


def train(model, train_loader, optimizer, criterion, epoch):
    tot_loss = 0
    n_samples = 0
    correct = 0

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx == 0:
            save_image("train", data)
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        tot_loss += loss.__float__() * target.shape[0]
        n_samples += target.shape[0]
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.4f}%)]\tLoss: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    print(
        '[Epoch : %d] Training Accuracy = %f, Training Loss = %f' % (epoch, correct / n_samples, tot_loss / n_samples))
    # return acc, loss
    return correct / n_samples, tot_loss / n_samples


def test(model, test_loader, skip_interval=1, criterion=nn.CrossEntropyLoss(), rotation=0, noise="a", rb_angle=0):

    model.eval()

    clean_correct = 0
    clean_loss = 0

    tot = 0
    batch_id = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx == 0:
            save_image("test"+noise+str(rb_angle), data)
        #print(target)
        # skip some batches to quickly get rough evaluation
        batch_id += 1
        if (batch_id - 1) % skip_interval != 0: continue

        data, target = data.cuda(), target.cuda()
        data = TF.rotate(data,angle=rotation)

        clean_output = model(data)
        clean_loss += criterion(clean_output, target).item() * target.shape[0]

        clean_pred = clean_output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        clean_correct += clean_pred.eq(target.view_as(clean_pred)).sum().item()

        tot += len(target)


    print('[Test] Accuracy: {}/{} ({:.4f}%) Loss: ({})'.format(
            clean_correct, tot,
            100. * clean_correct / tot, clean_loss / tot
        ))
    # print('[Test] Loss: ({})'.format(
    #         clean_loss / tot
    #     ))
    return clean_correct / tot, clean_loss / tot





def save_image(name,input1):
    
    input1 = torchvision.utils.make_grid(input1)
    inp = input1.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)*255
    inp = inp.astype('uint8')
    
    cv2.imwrite("./images/"+name+".jpg", cv2.cvtColor(inp, cv2.COLOR_RGB2BGR))
    
    


def test_oneclass(model, test_loader, skip_interval=1, criterion=nn.CrossEntropyLoss(), rotation=0, bd_target =0 ,source_label =14):

    model.eval()

    clean_correct = 0
    clean_correct_0 = 0 
    clean_loss = 0

    tot = 0
    tot_0 = 0
    batch_id = 0

    for data, target in test_loader:
        #print(target)
        # skip some batches to quickly get rough evaluation
        batch_id += 1
        if (batch_id - 1) % skip_interval != 0: continue

        data, target = data.cuda(), target.cuda()
        data = TF.rotate(data,angle=rotation)

        clean_output = model(data)
        clean_loss += criterion(clean_output, target).item() * target.shape[0]

        clean_pred = clean_output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        clean_correct += clean_pred.eq(target.view_as(clean_pred)).sum().item()

        tot += len(target)
        tot_0 += (target == source_label).sum().item()
        #print( target, clean_pred.view_as(target))
        #print( ((clean_pred.view_as(target)==0) * (target == 14)).sum().item() )

        clean_correct_0 += ((clean_pred.view_as(target)==bd_target) * (target == 14)).sum().item()




    print('[Test] Accuracy: {}/{} ({:.4f}%) Loss: ({})'.format(
            clean_correct, tot,
            100. * clean_correct / tot, clean_loss / tot
        ))
    print('[Test] BD Accuracy: ({})'.format(
            clean_correct_0 / tot_0
        ))
    return clean_correct / tot, clean_loss / tot
