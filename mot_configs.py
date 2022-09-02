import os.path as osp
from demo_toolkit import *
import wml_utils as wmlu
from eval_mot import eval as mot_eval
from person_det import *
from mfast_reid_t1.fast_reid import MFastReIDT1


config_sportstrack = {'model': 'SportsTracker',
 'TrackName': 'PDSMV2SportsTrackerT1',
 'use_reid': True,
 'reid_model': MFastReIDT1,
 'person_det': PersonDetectionSMSV2,
 'det_thresh': 0.6919616344544846,
 'reid_thresh': 0.306260605465699,
 'max_person_nr': -1,
 'track_buffer': 143,
 'thresh': [0.5469308237306498, 0.641378722, 0.36310692739022454],
 'nms_thresh': 0.45681998911015786
 }
'''config_byte_tracker = {  #(61.365    78.858    47.839) 
#58.043
    "model":"BYTETracker",
    "TrackName":"PDSMV2ByteTracker",
    "use_reid":False,
    "reid_model":MFastReIDT1,
    "person_det":PersonDetectionSMV2,
    "det_thresh":0.5,
    "max_person_nr":-1,
}
config_bot_sort = {  #(65.707    78.583    54.997)
#63.623
    "model":"BoTSORT",
    "TrackName":"PDSMV2BoTSORT",
    "use_reid":True,
    "reid_model":MFastReIDT1,
    "person_det":PersonDetectionSMV2,
    "det_thresh":0.5,
    "max_person_nr":-1,
}'''
global_config = config_sportstrack
