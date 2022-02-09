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
import cv2
from skimage import measure

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
                    help="Directory containing params.json")
parser.add_argument('--weights_file', required=True,
                    metavar="best or last",
                    help="Name of the file in --model_dir \
                        containing weights to reload")  # 'best' or 'train'
parser.add_argument('--baseline', action='store_true',
                    help="using the baseline approach if enabled")

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
size_upsample = (224, 224)


def return_cam(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    # size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for c, idx in enumerate(class_idx):
        cam = weight_softmax[idx].dot(feature_conv[c].reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def return_bbox_and_region(cam_imgs, ratios):
    bbox_dict = {r: [] for r in ratios}
    region_dict = {r: [] for r in ratios}
    for cam_img in cam_imgs:
        for k in bbox_dict.keys():
            bbox, region = utils.return_single_bbox_and_region(cam_img, k)
            bbox_dict[k].append(bbox)
            region_dict[k].append(region)

    return bbox_dict, region_dict


def crop_and_resize(input_image, bbox):
    left, top, right, bottom = bbox
    output_image = input_image[top:bottom, left:right, :]
    return cv2.resize(output_image, size_upsample)


def erase(input_image, region):
    output_image = np.copy(input_image)
    output_image[region] = 0
    return output_image


def draw_bbox(image, bbox, color):

    left, top, right, bottom = bbox
    cv2.rectangle(image, (left, top), (right, bottom), color)


def evaluate_model(model, dataloaders, criterion, mining_times, ratios, vis_dir, color_map, is_baseline):
    since = time.time()

    model.eval()

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
    count = 0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.set_grad_enabled(False):
            if is_baseline:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            else:

                full_img_outputs, regions_outputs, concat_outputs, erased_img_outputs = model(
                    inputs)
                # already numpy
                conv_feature_blobs = model.conv_feature_blobs
                weight_softmax = model.weight_softmax
                aux_conv_feature_blobs = model.aux_conv_feature_blobs
                aux_weight_softmax = model.aux_weight_softmax

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

                erased_img_preds = {}
                for eid, eo in erased_img_outputs.items():
                    _, ep = torch.max(eo, 1)
                    loss += criterion(eo, labels)
                    erased_img_preds[eid] = ep

        if not is_baseline:
            print("full_img_preds: ", full_img_preds)
            main_cams = return_cam(
                conv_feature_blobs[0], weight_softmax, full_img_preds)
            # if use_ae:
            aux_cams = {}
            for i in range(len(aux_conv_feature_blobs)):
                aux_cams[i] = return_cam(
                    aux_conv_feature_blobs[i], aux_weight_softmax[i], erased_img_preds[i])

            bbox_of_mined_regions = {}
            mined_regions = {}
            for t in range(mining_times):
                if t == 0:
                    bbox_dict, region_dict = return_bbox_and_region(
                        main_cams, ratios)
                else:
                    bbox_dict, region_dict = return_bbox_and_region(
                        aux_cams[t - 1], ratios)
                bbox_of_mined_regions[t] = bbox_dict
                mined_regions[t] = region_dict

            for i in range(inputs.size()[0]):
                input_img = inputs[i].data.cpu().numpy().transpose((1, 2, 0))
                gt = labels[i].data.cpu().numpy()
                pred = full_img_preds[i].data.cpu().numpy()
                input_img = std * input_img + mean
                input_img = np.clip(input_img, 0, 1)
                input_img = np.uint8(255 * input_img)
                input_img = input_img[..., ::-1]
                cv2.imwrite(os.path.join(vis_dir, str(
                    count) + '_input_gt_' + str(gt) + '_pred_' + str(pred) + '.jpg'), input_img)
                heatmap = cv2.applyColorMap(main_cams[i], cv2.COLORMAP_JET)
                result = heatmap * 0.3 + input_img * 0.5
                # draw_bbox(result, obj_bbox[i], is_part=False)
                for ratio in ratios:
                    draw_bbox(
                        result, bbox_of_mined_regions[0][ratio][i], color_map[ratio])

                concat_pred = concat_preds[i].data.cpu().numpy()
                cv2.imwrite(os.path.join(vis_dir, str(
                    count) + '_main_cam_concat_pred_' + str(concat_pred) + '.jpg'), result)

                current_img = input_img
                for t in range(mining_times):
                    for ratio in ratios:
                        region_img = crop_and_resize(
                            input_img, bbox_of_mined_regions[t][ratio][i])
                        region_pred = regions_preds[t][ratio][i].data.cpu(
                        ).numpy()
                        cv2.imwrite(os.path.join(vis_dir, str(
                            count) + '_' + str(t) + '_' + str(ratio) + '_reg_pred_' + str(region_pred) + '.jpg'), region_img)
                        erased_img = erase(
                            current_img, mined_regions[t][ratio][i])
                        current_img = erased_img
                    if t < mining_times - 1:
                        aux_heatmap = cv2.applyColorMap(
                            aux_cams[t][i], cv2.COLORMAP_JET)
                        aux_result = aux_heatmap * 0.3 + erased_img * 0.5
                        for ratio in ratios:
                            draw_bbox(
                                aux_result, bbox_of_mined_regions[t + 1][ratio][i], color_map[ratio])
                        erased_img_pred = erased_img_preds[t][i].data.cpu(
                        ).numpy()
                        cv2.imwrite(os.path.join(vis_dir, str(count) +
                                                 '_' + str(t) + '_aux_cam_pred_' + str(erased_img_pred) + '.jpg'), aux_result)

                count += 1

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

    dataset_size = dataloaders['test_size']
    overall_loss = running_loss / dataset_size
    if is_baseline:
        overall_acc = running_corrects.double() / dataset_size
        logging.info('Test Loss: {:.4f} Acc: {:.4f}'.format(
            overall_loss, overall_acc))
    else:
        overall_full_img_acc = full_img_running_corrects.double() / dataset_size

        logging.info('Test Loss: {:.4f} Full Img Acc: {:.4f}'.format(
            overall_loss, overall_full_img_acc))

        for mining_time, regions_running_corrects_sub in regions_running_corrects.items():
            for ratio, rrcs in regions_running_corrects_sub.items():
                logging.info('{}-{} Region Acc: {:.4f}'.format(
                    mining_time, ratio, rrcs.double() / dataset_size))

        overall_concat_acc = concat_running_corrects.double() / dataset_size
        logging.info("Concat Acc: {:.4f}".format(overall_concat_acc))

        # if use_ae:
        for eid, erc in erased_img_running_corrects.items():
            logging.info('{} Erased Img Acc: {:.4f}'.format(
                eid, erc.double() / dataset_size))

    # print()

    time_elapsed = time.time() - since

    logging.info('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


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

    vis_dir = os.path.join('vis', args.model_dir.split('/')[1] + '_orig')
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)
        print("visualization will be saved in {}".format(vis_dir))

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
    utils.set_logger(os.path.join(args.model_dir, 'evaluate_orig.log'))

    # create the input data pipeline
    logging.info("Loading the dataset...")

    # fetch dataloader
    dataloaders = data_loader.fetch_dataloader(['test'],
                                               args.data_dir, params)
    logging.info("- done.")

    class_names = []
    with open(os.path.join(args.data_dir, 'classes.txt')) as f:
        for line in f:
            class_names.append(' '.join(line.strip().split()[1:]))
    print("class_names: ", class_names)

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    color_map = dict(zip(params.ratios, colors))

    # Visualize a few images
    # Get a batch of test data
    inputs, classes = next(iter(dataloaders['test']))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

    if args.baseline:
        net = BaselineNet(params.num_classes, params.fe_model)
    else:
        net = PARNet(params.num_classes, params.mining_times, params.ratios,
                     params.fe_model)
    print(net)

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.weights_file + '.pth.tar'), net)

    net = net.to(device)

    criterion = nn.CrossEntropyLoss()

    logging.info("Starting evaluation")
    ####################### evaluate#############################
    evaluate_model(net, dataloaders, criterion, mining_times=params.mining_times, ratios=params.ratios,
                   vis_dir=vis_dir, color_map=color_map, is_baseline=args.baseline)

    ######################################################################
    visualize_model(
        net, class_names, dataloaders['test'], args.baseline)

    plt.ioff()
    plt.show()
