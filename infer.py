import os
import argparse
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed

from src_files.helper_functions.bn_fusion import fuse_bn_recursively
from src_files.models import create_model
import matplotlib

from src_files.models.tresnet.tresnet import InplacABN_to_ABN


from PIL import Image
import numpy as np

pic_path = './pics/000000000885.jpg'




def main():

    model_name = 'tresnet_l'
    model_path = './tresnet_l_stanford_card_96.41.pth'
    th = 0.75

    model = create_model()
    model = model.cuda()
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state['model'], strict=True)

    model = model.cpu()
    model = InplacABN_to_ABN(model)
    model = fuse_bn_recursively(model)
    model = model.cuda().half().eval()

    # print('done')

    classes_list = np.array(state['idx_to_class'])
    print('done\n')

    # doing inference
    print('loading image and doing inference...')
    im = Image.open(pic_path)
    im_resize = im.resize((384, 384))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    tensor_batch = torch.unsqueeze(tensor_img, 0).cuda().half() # float16 inference
    output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
    np_output = output.cpu().detach().numpy()


    ## Top-k predictions
    # detected_classes = classes_list[np_output > args.th]
    idx_sort = np.argsort(-np_output)
    detected_classes = np.array(classes_list)[idx_sort][: 1]
    scores = np_output[idx_sort][: 1]
    idx_th = scores > th
    detected_classes = detected_classes[idx_th]
    print('done\n')
    print(detected_classes)
    print('done\n')


if __name__ == '__main__':
    main()
