import torch
import os


from config import argument_parser

parser = argument_parser()
args = parser.parse_args()
main(args)

# exp_dir = os.path.join('exp_result', args.dataset)
# model_dir, log_dir = get_model_log_path(exp_dir, visenv_name)
# stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
# save_model_path = os.path.join(model_dir, 'ckpt_max_new.pth')
#
#
# map_location = (lambda storage, loc: storage)
# ckpt = torch.load(cfg.model_weight_file, map_location=map_location)
# model.load_state_dict(ckpt['state_dicts'][0])
#
# model.cuda()
# model.eval()

