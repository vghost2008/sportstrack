import os
from collections import defaultdict
import os.path as osp
import glob
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='MOT')
    parser.add_argument('--input_txt_dir', type=str,default="/home/wj/ai/mldata1/SportsMOT-2022-4-24/tmp/postprocess_output", help='input dir path')
    parser.add_argument('--output_txt_dir', type=str,default="/home/wj/ai/mldata1/SportsMOT-2022-4-24/tmp/postprocess_output1",help='output txt dir')
    return parser.parse_args()

def Iou(rec1,rec2):
  x1,x2,y1,y2 = rec1 #分别是第一个矩形左右上下的坐标
  x3,x4,y3,y4 = rec2 #分别是第二个矩形左右上下的坐标
  area_1 = (x2-x1)*(y1-y2)
  area_2 = (x4-x3)*(y3-y4)
  sum_area = area_1 + area_2
  w1 = x2 - x1#第一个矩形的宽
  w2 = x4 - x3#第二个矩形的宽
  h1 = y1 - y2
  h2 = y3 - y4
  W = min(x1,x2,x3,x4)+w1+w2-max(x1,x2,x3,x4)#交叉部分的宽
  H = min(y1,y2,y3,y4)+h1+h2-max(y1,y2,y3,y4)#交叉部分的高
  Area = W*H#交叉的面积
  Iou = Area/(sum_area-Area)
  return Iou
 
def Giou(rec1,rec2):
  x1,x2,y1,y2 = rec1 #分别是第一个矩形左右上下的坐标
  x3,x4,y3,y4 = rec2
  iou = Iou(rec1,rec2)
  area_C = (max(x1,x2,x3,x4)-min(x1,x2,x3,x4))*(max(y1,y2,y3,y4)-min(y1,y2,y3,y4))
  area_1 = (x2-x1)*(y1-y2)
  area_2 = (x4-x3)*(y3-y4)
  sum_area = area_1 + area_2
  w1 = x2 - x1#第一个矩形的宽
  w2 = x4 - x3#第二个矩形的宽
  h1 = y1 - y2
  h2 = y3 - y4
  W = min(x1,x2,x3,x4)+w1+w2-max(x1,x2,x3,x4)#交叉部分的宽
  H = min(y1,y2,y3,y4)+h1+h2-max(y1,y2,y3,y4)#交叉部分的高
  Area = W*H#交叉的面积
  add_area = sum_area - Area #两矩形并集的面积
  end_area = (area_C - add_area)/area_C #(c/(AUB))/c的面积
  giou = iou - end_area
  return giou
 


def frame_interpolation(txtpath,filename,outputpath,testnum,delete_num):
    if (testnum==1):
        print(txtpath)
    content = []
    interpolation_elements = []
    if testnum==1:
        print(filename)
    print(f"Read {txtpath}")
    with open(txtpath,encoding='utf-8') as file:
        content=file.readlines()
             ##rstrip()删除字符串末尾的空行
    # print(content)
    ###逐行读取数据
    id_map = defaultdict(list)
    coord_map = defaultdict(list)
    framemap = defaultdict(int)
    for line in content:
        # print(line.rstrip())
        ele = line.split(",")
        id_map[int(ele[1])].append(int(ele[0]))
        coord_map[(int(ele[0]),int(ele[1]))]=ele[2:6]
        framemap[int(ele[0])]+=1
    tmp = []
    for ids in id_map:
        
        if len(id_map[ids])<=delete_num:
            print("find removed in {}".format(filename))
            tmp.append([id_map[ids],ids])
    for line in content:
        ele = line.split(",")
        for t in tmp:
            # print(t)
            for idx in t[0]:
                if int(ele[0])==idx and int(ele[1])==t[1]:
                    content.remove(line)
                    print("remove line {}".format(t))


    # print(coord_map)
    for ids in id_map:
        templen = len(id_map[ids])
        id_map[ids] = sorted(list(set((id_map[ids]))))
        
        if len(id_map[ids])>90 and templen>len(framemap)*0.6:
            for i in range(len(id_map[ids])-1,-1,-1):
                # print(i)
                if id_map[ids][i]-id_map[ids][i-1]>1 and id_map[ids][i]-id_map[ids][i-1]<testnum:
                    # print(id_map[ids])
                    # print("{}  {}".format(id_map[ids][i],id_map[ids][i-1]))
                    startframe_id = id_map[ids][i-1]
                    endframe_id = id_map[ids][i]
                    interpolation_frame_num = (id_map[ids][i]-id_map[ids][i-1])
                    coord1_id = (id_map[ids][i-1],ids)
                    coord2_id = (id_map[ids][i],ids)
                    coord1_val = coord_map[coord1_id]
                    coord2_val = coord_map[coord2_id]
                    # print(startframe_id)
                    # print(endframe_id)
                    # print(coord2_val)
                    # print(coord1_val)
                    x1_temp = (float(coord2_val[0])-float(coord1_val[0]))/interpolation_frame_num
                    y1_temp = (float(coord2_val[1])-float(coord1_val[1]))/interpolation_frame_num
                    width_temp = (float(coord2_val[2])-float(coord1_val[2]))/interpolation_frame_num
                    height_temp = (float(coord2_val[3])-float(coord1_val[3]))/interpolation_frame_num
                    centerpointx2 = float(coord2_val[0])+float(coord2_val[2])/float(2)
                    centerpointy2 = float(coord2_val[1])+float(coord2_val[3])/float(2)
                    centerpointx1 = float(coord1_val[0])+float(coord1_val[2])/float(2)
                    centerpointy1 = float(coord1_val[1])+float(coord1_val[3])/float(2)
                    centerpointx_temp = (centerpointx2-centerpointx1)/interpolation_frame_num
                    centerpointy_temp = (centerpointy2-centerpointy1)/interpolation_frame_num
                    for i in range(interpolation_frame_num-1):
                        interpol_frame = startframe_id+i+1
                        interpol_id = ids
                        interpol_x1 = centerpointx1+(i+1)*centerpointx_temp-(float(coord2_val[2])+float(coord1_val[2]))/4.0
                        interpol_y1 = centerpointy1+(i+1)*centerpointy_temp-(float(coord2_val[3])+float(coord1_val[3]))/4.0
                        interpol_x2 = float(coord2_val[2])+width_temp*(i+1)
                        interpol_y2 = float(coord2_val[3])+height_temp*(i+1)

                        interpolation_elements.append(",".join([str(interpol_frame),str(interpol_id),str(interpol_x1),str(interpol_y1),str(interpol_x2),str(interpol_y2),str(-1),str(-1),str(-1),str(-1)])+'\n')
                        # print(interpolation_elements[-1])
    # print(interpolation_elements)
    interpolation_file = open(os.path.join(outputpath,filename),"w")
    interpolation_file.writelines(content+interpolation_elements)
    # interpolation_file.writelines(interpolation_elements)
    interpolation_file.close

deletenum = 80
testnum=9
# interpolation input folder path
args = parse_args()
folderpath = args.input_txt_dir
outputpath = args.output_txt_dir
if not os.path.exists(outputpath):
    os.mkdir(outputpath)
files = glob.glob(osp.join(folderpath,"*.txt"))
for filepath in files:
    file = osp.basename(filepath)
    frame_interpolation(filepath,file,outputpath,testnum,deletenum)
