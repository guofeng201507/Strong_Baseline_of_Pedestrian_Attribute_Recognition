import os
import pprint
from collections import OrderedDict, defaultdict
from PIL import Image, ImageDraw
from torch.autograd import Variable
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import AttrDataset, get_transform
from loss.CE_loss import CEL_Sigmoid
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50
from tools.function import get_model_log_path, get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed

set_seed(605)

def init():
    parser = argument_parser()
    parser.add_argument('--dataset', type=str, default='PETA', choices=['peta', 'rap', 'pa100k'])
    parser.add_argument('--model_weight_file', type=str,
                        default='./exp_result/PETA/PETA/img_model/ckpt_max_new.pth')
    args = parser.parse_args()

    visenv_name = 'PETA'
    exp_dir = os.path.join('exp_result', visenv_name)
    model_dir, log_dir = get_model_log_path(exp_dir, visenv_name)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    save_model_path = os.path.join(model_dir, 'ckpt_max_new.pth')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    print(f'use GPU{args.device} for training')

    _, predict_tsfm = get_transform(args)

    valid_set = AttrDataset(args=args, split=args.valid_split, transform=predict_tsfm)

    args.att_list = valid_set.attr_id

    backbone = resnet50()
    classifier = BaseClassifier(nattr=valid_set.attr_num)
    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    ckpt = torch.load(save_model_path)
    model.load_state_dict(ckpt['state_dicts'])

    model.eval()

    from torchsummary import summary
    summary(model, input_size=(3, 256, 256))

    print('Total number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    return args, predict_tsfm, model


def predict(args, predict_tsfm, model, input_image):
    # visenv_name = args.dataset
    # load one image
    path = './static/uploads/'
    output_image = input_image[:-4] + '_predicted.png'
    img = Image.open(path + input_image)
    img_trans = predict_tsfm(img)
    img_trans = torch.unsqueeze(img_trans, dim=0)
    img_var = Variable(img_trans).cuda()
    score = model(img_var).data.cpu().numpy()

    # show the score in command line
    result = {}
    for idx in range(len(args.att_list)):
        orgin_score = score[0, idx]
        sigmoid_score = 1 / (1 + np.exp(-1 * orgin_score))
        # if score[0, idx] >= 0:
        #     print('%s: %.2f' % (cfg.att_list[idx], score[0, idx]))
        print('%s: %.5f' % (args.att_list[idx], sigmoid_score))
        result[args.att_list[idx]] = sigmoid_score

    # show the score in the image
    img = img.resize(size=(256, 512), resample=Image.BILINEAR)
    draw = ImageDraw.Draw(img)
    positive_cnt = 0
    for idx in range(len(args.att_list)):
        if score[0, idx] >= 0:
            orgin_score = score[0, idx]
            sigmoid_score = 1 / (1 + np.exp(-1 * orgin_score))
            # txt = '%s: %.2f' % (cfg.att_list[idx], score[0, idx])
            txt = '%s: %.5f' % (args.att_list[idx], sigmoid_score)
            draw.text((10, 10 + 10 * positive_cnt), txt, (255, 0, 0))
            positive_cnt += 1
    # img.save('./static/uploads/00003_predicted.png')
    img.save(path + output_image)

    return result


args, predict_tsfm, model = init()
predict(args, predict_tsfm, model, '00003.png')
