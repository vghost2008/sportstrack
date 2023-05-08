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
    parser.add_argument('--skip_exists', type=bool,default=False, help='log img')
    parser.add_argument('--config', type=str,default="config_sportstrack", help='log img')
    parser.add_argument('--gpus', type=str,default="1", help='log img')
    parser.add_argument('--reverse', type=bool,default=False, help='log img')
    parser.add_argument('--dataset', type=str,default="sportsmot", help='log img')
    parser.add_argument('--only_dir', type=str,default="", help='log img')
    parser.add_argument('--split', type=str,default="val", help='log img')
    parser.add_argument('--total_nr', type=int,default=0, help='log img')
    parser.add_argument('--cur_idx', type=int,default=0, help='log img')
    return parser.parse_args()

args = parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

from mot_configs import *
import os.path as osp
from demo_toolkit import *
from track_framework import *
import wml_utils as wmlu
from eval_mot import eval as mot_eval


class Model:
    def __init__(self,save_path,name=None,args=None,imgs_save_path=None,video_path=None) -> None:
        self.tracker = TrackDemo(save_path,config=global_config,name=name,
                                 args=args,imgs_save_path=imgs_save_path,video_path=video_path)
        #self.tracker = TrackWithKPS(save_path,config=global_config,name=name,
                                 #args=args,imgs_save_path=imgs_save_path)
        pass

    def __call__(self, frame):
        frame = self.tracker.track(frame)
        return frame


if __name__ == "__main__":
    dataset_name = "sportsmot"
    split = args.split
    data_root = os.environ.get("SPORTSTRACK_ROOT","/home/wj/ai/mldata1/SportsMOT-2022-4-24")
    videos_path = osp.join(data_root,"data/sportsmot_publish/dataset",split)
    basename = split
    trackers_dir = osp.join(data_root,"tmp/")
    if len(args.config)>1:
        if not str(args.config).startswith("config"):
            args.config = "config"+str(args.config)
        global_config = eval(args.config)
    tracker_name = global_config["TrackName"]
    print(f"Track name {tracker_name}, gpus={os.environ['CUDA_VISIBLE_DEVICES']}")
    sys.stdout.flush()
    save_path0 = osp.join(trackers_dir,basename,tracker_name)
    save_path1 = osp.join(trackers_dir,dataset_name+"-"+basename,tracker_name,"data")
    dirs = wmlu.get_subdir_in_dir(videos_path,absolute_path=True)

    if len(args.only_dir)>3:
        print(f"Only dir {args.only_dir}")
        sys.stdout.flush()

    if args.reverse:
        dirs.reverse()

    if not args.skip_exists and args.total_nr<=0:
        wmlu.create_empty_dir_remove_if(save_path0,key_word="tmp")
        wmlu.create_empty_dir_remove_if(save_path1,key_word="tmp")
    
    if args.total_nr>0:
        dirs = wmlu.list_to_2dlistv2(dirs,args.total_nr)
        dirs = dirs[args.cur_idx]
        time.sleep(args.cur_idx)
        print(f"{os.getpid()} process {len(dirs)} dirs")
        wmlu.show_list(dirs)
    
    for i,dir in enumerate(dirs):
        t0 = time.time()
        data_basename = osp.basename(dir)
        if len(args.only_dir)>3:
            if data_basename != args.only_dir:
                continue
        print(f"Process dir {dir}, {i+1}/{len(dirs)}.")
        cur_save_path = osp.join(save_path0,osp.basename(dir))
        cur_video_path = osp.join(dir,"img1")
        cur_save_path1 = osp.join(save_path1,osp.basename(dir)+".txt")
        if args.skip_exists and osp.exists(cur_save_path1):
            print(f"Skip {cur_save_path1}")
            continue
        wmlu.create_empty_dir(cur_save_path,remove_if_exists=False)
        vd = VideoDemo(Model(save_path=cur_save_path1,
                name=data_basename,args=args,
                imgs_save_path=cur_save_path,video_path=cur_video_path),
                save_path=cur_save_path,
                show_video=False,
                max_frame_cn=None,
                args=args)
        vd.inference_loop(cur_video_path)
        vd.close()
        print(f"{dir} use {time.time()-t0} secs.")

    if len(args.only_dir)<3 and args.total_nr<=1:
        mot_eval(osp.dirname(videos_path),trackers_dir,BENCHMARK=dataset_name,split=basename,trackers_to_eval=tracker_name)

