import os
import argparse
'''
v_00HRwkvvjtQ_c001
v_9MHDmAMxO5I_c009
v_ITo3sCnpw_k_c007
'''
def parse_args():
    parser = argparse.ArgumentParser(description='MOT')
    parser.add_argument('--log_imgs', type=bool,default=True, help='log img')
    parser.add_argument('--skip_exists', type=bool,default=True, help='log img')
    parser.add_argument('--config', type=str,default="config8_12", help='log img')
    parser.add_argument('--gpus', type=str,default="1", help='log img')
    parser.add_argument('--reverse', type=bool,default=False, help='log img')
    parser.add_argument('--dataset', type=str,default="sportsmot", help='log img')
    parser.add_argument('--only_dir', type=str,default="", help='log img')
    parser.add_argument('--split', type=str,default="mini-val", help='log img')
    parser.add_argument('--total_nr', type=int,default=5, help='log img')

    return parser.parse_args()

args = parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

from mot_configs import *
import os.path as osp
import wml_utils as wmlu
from eval_mot import eval as mot_eval



if __name__ == "__main__":
    dataset_name = "sportsmot"
    split = args.split
    if split.startswith("MOT"):
        videos_path = osp.join("/home/wj/ai/mldata/MOT",split,"test")
    else:
        videos_path = osp.join("/home/wj/ai/mldata1/SportsMOT-2022-4-24/data/sportsmot_publish/dataset",split)
    basename = split
    trackers_dir = "/home/wj/ai/mldata1/SportsMOT-2022-4-24/tmp/"
    if len(args.config)>1:
        if not str(args.config).startswith("config"):
            args.config = "config"+str(args.config)
        global_config = eval(args.config)
    tracker_name = global_config["TrackName"]
    print(f"Track name {tracker_name}")
    sys.stdout.flush()
    save_path0 = osp.join(trackers_dir,basename,tracker_name)
    save_path1 = osp.join(trackers_dir,dataset_name+"-"+basename,tracker_name,"data")
    data = mot_eval(osp.dirname(videos_path),trackers_dir,BENCHMARK=dataset_name,split=basename,trackers_to_eval=tracker_name)
    print(data[-1])

