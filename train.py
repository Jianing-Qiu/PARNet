import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import logging

import model.data_loader as data_loader
import utils
from model.net import PARNet, BaselineNet

plt.ion()   # interactive mode
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--data_dir', required=True,
                    metavar="path/to/dataset/",
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', required=True,
                    metavar="path/to/params.json",
                    help="Directory containing params.json")

parser.add_argument('--baseline', action='store_true',
                    help="using the baseline approach if enabled")


def train_model(model, dataloaders, criterion, optimizer, scheduler,
                num_epochs, mining_times, ratios, model_dir, is_baseline):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()
                dataloader = dataloaders['train']
                dataset_size = dataloaders['train_size']
            else:
                model.eval()
                dataloader = dataloaders['test']
                dataset_size = dataloaders['test_size']

            running_loss = 0.0
            if is_baseline:
                running_corrects = 0
            else:
                full_img_running_corrects = 0
                regions_running_corrects = {}
                for t in range(mining_times):
                    regions_running_corrects_sub = {}
                    for r in ratios:
                        regions_running_corrects_sub[r] = 0
                    regions_running_corrects[t] = regions_running_corrects_sub
                concat_running_corrects = 0
                erased_img_running_corrects = {
                    t: 0 for t in range(mining_times - 1)}

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    if is_baseline:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    else:
                        if phase == 'train':

                            full_img_outputs, regions_outputs, concat_outputs, erased_img_outputs = model(
                                inputs, labels)

                        else:

                            full_img_outputs, regions_outputs, concat_outputs, erased_img_outputs = model(
                                inputs)

                        _, full_img_preds = torch.max(full_img_outputs, 1)
                        loss = criterion(full_img_outputs, labels)

                        regions_preds = {}
                        for mining_time, regions_outputs_sub in regions_outputs.items():
                            regions_preds_sub = {}
                            for ratio, ro in regions_outputs_sub.items():
                                _, rp = torch.max(ro, 1)
                                loss += criterion(ro, labels)
                                regions_preds_sub[ratio] = rp
                            regions_preds[mining_time] = regions_preds_sub

                        _, concat_preds = torch.max(concat_outputs, 1)
                        loss += criterion(concat_outputs, labels)

                        # if use_ae:
                        erased_img_preds = {}
                        for eid, eo in erased_img_outputs.items():
                            _, ep = torch.max(eo, 1)
                            loss += criterion(eo, labels)
                            erased_img_preds[eid] = ep

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                if is_baseline:
                    running_corrects += torch.sum(preds == labels.data)
                else:
                    full_img_running_corrects += torch.sum(
                        full_img_preds == labels.data)

                    for mining_time, regions_preds_sub in regions_preds.items():
                        for ratio, rp in regions_preds_sub.items():
                            regions_running_corrects[mining_time][ratio] += torch.sum(
                                rp == labels.data)

                    concat_running_corrects += torch.sum(
                        concat_preds == labels.data)

                    # if use_ae:
                    for eid, ep in erased_img_preds.items():
                        erased_img_running_corrects[eid] += torch.sum(
                            ep == labels.data)

            epoch_loss = running_loss / dataset_size
            if is_baseline:
                epoch_acc = running_corrects.double() / dataset_size
                logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
            else:
                epoch_full_img_acc = full_img_running_corrects.double() / dataset_size

                logging.info('{} Loss: {:.4f} Full Img Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_full_img_acc))

                for mining_time, regions_running_corrects_sub in regions_running_corrects.items():
                    for ratio, rrcs in regions_running_corrects_sub.items():
                        logging.info('{}-{} Region Acc: {:.4f}'.format(
                            mining_time, ratio, rrcs.double() / dataset_size))

                epoch_concat_acc = concat_running_corrects.double() / dataset_size
                logging.info("Concat Acc: {:.4f}".format(epoch_concat_acc))

                for eid, erc in erased_img_running_corrects.items():
                    logging.info('{} Erased Img Acc: {:.4f}'.format(
                        eid, erc.double() / dataset_size))

                epoch_acc = epoch_concat_acc

            # deep copy the model
            if phase == 'test':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    utils.save_checkpoint({'epoch': epoch,
                                           'state_dict': model.state_dict(),
                                           'optim_dict': optimizer.state_dict()},
                                          is_best=True,
                                          checkpoint=model_dir)
                else:
                    utils.save_checkpoint({'epoch': epoch,
                                           'state_dict': model.state_dict(),
                                           'optim_dict': optimizer.state_dict()},
                                          is_best=False,
                                          checkpoint=model_dir)

        # print()

    time_elapsed = time.time() - since

    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('Best test Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Visualizing the model predictions


def visualize_model(model, class_names, dataloader, is_baseline, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            if is_baseline:
                outputs = model(inputs)
            else:

                _, _, outputs, _ = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if __name__ == '__main__':

    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json cofiguration file found at {}"\
        .format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(
        ['train', 'test'], args.data_dir, params)

    logging.info("- done.")

    class_names = []
    with open(os.path.join(args.data_dir, 'classes.txt')) as f:
        for line in f:
            class_names.append(' '.join(line.strip().split()[1:]))
    print("class_names: ", class_names)

    # Visualize a few images
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

    if args.baseline:
        net = BaselineNet(params.num_classes, params.fe_model)
    else:
        net = PARNet(params.num_classes, params.mining_times, params.ratios,
                     params.fe_model)
    print(net)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(net.parameters(), lr=params.learning_rate,
                             momentum=params.momentum, weight_decay=params.weight_decay)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=params.step_size, gamma=params.gamma)
    ####################### Train and evaluate#############################
    net = train_model(net, dataloaders, criterion, optimizer_ft,
                      exp_lr_scheduler, num_epochs=params.num_epochs,
                      mining_times=params.mining_times, ratios=params.ratios, model_dir=args.model_dir,
                      is_baseline=args.baseline)

    ######################################################################
    visualize_model(net, class_names,
                    dataloaders['test'], args.baseline)

    plt.ioff()
    plt.show()
