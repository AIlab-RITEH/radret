import numpy as np
import cv2

import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler

import torch.nn.functional as F
import sys
import os
import skimage.io as io


def load_images(path):
    images = []
    ids = []
    files = os.listdir(path)
    cleaned = list(filter(lambda n: ".db" not in n, files))
    images_tuple = list(map(lambda n: (int(n.split(".")[0]), n), cleaned))
    images_tuple = sorted(images_tuple, key=lambda n: n[0])
    files = list(map(lambda n: n[1], images_tuple))
    files.sort()
    for image in files:
        img = io.imread(path + image)
        images.append(img)
        ids.append(image)

    return images, list(map(lambda n: int(str.split(n, ".")[0]), ids))


def fit_epoch(model, optimizer, criterion, train_dataloader, device, verbose=False):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    total = len(train_dataloader)
    for i, data in enumerate(train_dataloader):
        data, target = data[0].float().to(device), data[1].long().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        _, preds = torch.max(output, 1)

        if verbose:
            if (i % 300 == 0):
                print(loss)

        train_running_loss += loss.item()
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
        percent = 100.0*(i)/total
        sys.stdout.write('\r')
        sys.stdout.write("Completed: [{:{}}] {:>3}%"
                         .format('='*int(percent/(100.0/30)),
                                 30, int(percent)))

    train_loss = train_running_loss/len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(train_dataloader.dataset)
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')

    return train_loss, train_accuracy, model


def extract_softmax(model, dataloader, device='cpu'):
    model.eval()
    model.to(device)
    all_probabilities = []

    for inputs in dataloader:
        inputs = inputs[0].to(device)
        logits = model(inputs)
        probabilities = F.softmax(logits, dim=1)
        all_probabilities.extend(probabilities.detach().cpu().numpy())

    return all_probabilities

def add_frame(image, color, frame_width):
    if color == 'red':
        frame_color = (255, 0, 0) 
    elif color == 'green':
        frame_color = (0, 255, 0)
    elif color == 'blue':
        frame_color = (0, 0, 255)

    else:
        frame_color = (0, 0, 0)
    
    framed_image = cv2.copyMakeBorder(image, frame_width, frame_width, frame_width, frame_width, 
                                      cv2.BORDER_CONSTANT, value=frame_color)
    return framed_image

def extract_features(model, loader, device='cpu', return_predicted_label = False):
    model.eval()
    model.to(device)

    result_features = []
    result_targets = []
    for data in loader:
        if return_predicted_label:
            data = data[0].float().to(device)
            output_features, output_targets = model(data)
            result_features.append(output_features.detach().cpu())
            result_targets.append(output_targets.detach().cpu())
        else:
            data, target = data[0].float().to(device), data[1].long().to(device)
            result_targets.append(target.long().detach().cpu())
            output = model(data).detach().cpu()
            result_features.append(output)
    return np.vstack(result_features), np.hstack(result_targets)


def irma_similarity(s1,s2):

    weight = 1
    score = 0
    normalization_factor = 0
    score_at_k = 0
    for pos, _ in enumerate(s1):

        if s1[pos] == "-":            
            weight = 1
            score += 0.25*((score_at_k)/normalization_factor)
            score_at_k = 0
            normalization_factor = 0

            
        elif(s1[pos] != s2[pos]):
            if(s1[pos] == "*" or s2[pos] ==  "*"):
                score_at_k += (1/weight)
            else:
                pass
            normalization_factor += 1/weight
            weight +=1
            
        else:
            score_at_k += (1/weight)  
            normalization_factor += 1/weight
            weight+=1


    weight = 1
    score += 0.25*((score_at_k)/normalization_factor)

    return score
    
def validate(model, test_dataloader, criterion, device):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    results = []
    for _, data in enumerate(test_dataloader):
        data, target = data[0].float().to(device), data[1].long().to(device)
        output = model(data)
        loss = criterion(output, target)

        val_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        val_running_correct += (preds == target).sum().item()
        results.append(preds.cpu())

    val_loss = val_running_loss/len(test_dataloader.dataset)
    val_accuracy = 100. * val_running_correct/len(test_dataloader.dataset)

    print(
        f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.2f}')

    return val_loss, val_accuracy, results


def validate_irma(model, dataloader, evaluator, label_encoder, device):
    irma_sum = 0

    for _, data in enumerate(dataloader):
        img, target = data[0].float().to(device), data[1].long().to(device)

        targets = label_encoder.inverse_transform(target.cpu().numpy())
        preds = label_encoder.inverse_transform(
            torch.max(model(img), 1)[1].detach().cpu().numpy())
        for i, _ in enumerate(preds):
            irma_sum += evaluator.evaluate(targets[i], preds[i])
    print(f'Irma error {irma_sum:.2f}')

    return irma_sum

def merge_labels(top_k_labels):
    predicted_label = []

    for pos in range(0,16):
        unique_elements,count = np.unique(top_k_labels[:,pos],axis=0,return_counts = True)
        if(len(count) > 1):
            predicted_label.append("*")
        else:
            predicted_label.append(unique_elements[0][0])

    return "".join(predicted_label)


def resize_image(img, size=(300, 300)):
    # From: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    h, w = img.shape[:2]
    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (
        size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, 3), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)


def process(image):
    # Check dimensionality of each image
    # if image is 3-dimensional, stack it three times to create 3d matrix
    # if image has more than 3 channels, take only first three channels
    image = image.astype(np.uint8)
    if len(image.shape) == 2:
        return np.repeat(np.expand_dims(image, axis=2), 3, axis=2)
    elif len(image.shape) == 3:
        if (image.shape[2] == 4):
            return image[:, :, 0:3]
        else:
            return image
    else:
        return image


def get_sampler(train_labels):
    unique_elements, count = np.unique(train_labels, return_counts=True)
    weight = {}
    for i in range(len(unique_elements)):
        weight[unique_elements[i]] = 1/count[i]

    samples_weight = np.array([weight[t] for t in train_labels])

    samples_weight = torch.from_numpy(samples_weight)

    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    print(len(samples_weight))
    return sampler
