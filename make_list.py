# 2021/12/19
# use FaceX-Zoo-main ware mask on training data,and get all file list.

import os
import time
import random
from shutil import copyfile
import glob
import cv2
import shutil
import numpy as np
from PIL import Image

def get_list(data_path, save_path, label, part_rate=1):
    image_list = []
    for root, dir, files in os.walk(data_path):
        for image_name in files:
            if image_name.split(".")[-1] in ['jpg', 'png', 'bmp', 'jpeg', 'jfif', 'PNG', 'JPG']:
                image_path = os.path.join(root, image_name)
                try:
                    # img = Image.open(image_path)
                    # if img is None:
                    #     continue
                    # print(image_path)
                    image_list.append(image_path)
                except:
                    print('image not correct:',image_path)
                    # os.remove(image_path)
    random.shuffle(image_list)
    save_list = image_list[: int(len(image_list) * part_rate)]
    print('length of image list to save ', len(save_list))
    f = open(save_path, 'a+')
    for img_dir  in save_list:
        save_str = img_dir + ' ' + label
        # f.write(img_dir.replace('.jpg', '.bmp') + '\n')
        f.write(save_str + '\n')
        # copyfile(img_dir,'/mnt/sdb1/test_self_collect/test_part001/' + img_dir.split('/')[-1])
    return save_list

def replace_list(data_file, save_path):
    s = open(save_path, 'w')
    with open(data_file,'r') as f:
        lines = f.readlines()
        for line in lines:
           line = line.replace('/home/guohongwei/hpzhu/liveness/data','/mnt/disk2/data')
           s.write(line)
    s.close()


# 0827_abroad_landmark dir image list
def get_abroad_list(data_path, save_path):
    image_list = []
    for root, dir, files in os.walk(data_path):
        if '0827_abroad_landmark' not in root:
            continue
        for image_name in files:
            if image_name.split(".")[-1] in ['jpg', 'png', 'bmp', 'jpeg', 'jfif', 'PNG', 'JPG']:
                image_path = os.path.join(root, image_name)
                img = Image.open(image_path)
                if img is None:
                    continue
                landmark_txt_name = image_path.replace(image_path.split(".")[-1], 'txt')
                landmark = get_landmark(landmark_txt_name)
                dis = get_mouth_distance(landmark)
                if dis > 0.82:
                    continue
                if os.path.exists(landmark_txt_name):
                    image_list.append(image_path)
    random.shuffle(image_list)
    save_list =  image_list[: int(len(image_list))]
    print('length of image list to save ', len(save_list))
    f = open(save_path, 'a+')
    for img_dir  in save_list:
        f.write(img_dir + '\n')


def get_frame_from_video(video_name, interval,save_path,n):
    """

    Args:
        video_name:输入视频名字
        interval: 保存图片的帧率间
    Returns:

    """

    # 保存图片的路    save_path = video_name.split('.mp4')[0] + '/'
    is_exists = os.path.exists(save_path)
    if not is_exists:
        os.makedirs(save_path)
        print('path of %s is build' % save_path)
#     else:
#         shutil.rmtree(save_path)
#         os.makedirs(save_path)
#         print('path of %s already exist and rebuild' % save_path)

    # 开始读视频
    video_capture = cv2.VideoCapture(video_name)
    i = 0
    j = 0

    while True:
        success, frame = video_capture.read()
        i += 1
        if i % interval == 0:
            # 保存图片
            j += 1
            save_name = save_path +str(n)+ str(j) + '_' + str(i) + '.jpg'
            if isinstance (frame,(np.ndarray, np.generic)):
                frame = np.rot90(frame,3)
                frame=cv2.flip(frame,-1)
                cv2.imwrite(save_name, frame)
                print('image of %s is saved' % save_name)
        if not success:
            print('video is all read')
            break

def get_file_list(path_dir, pre='mp4'):
#     file_list = glob.glob(os.path.join(path_dir,'*/*' + pre))
    file_list=os.listdir(path_dir)
    
    return file_list

def test_get_frame(video_dir,save_path):
    video_list = get_file_list(video_dir, 'mp4')
    interval = 10
    n=1
    for video_name in video_list:
        
        get_frame_from_video(os.path.join(video_dir,video_name), interval,save_path,n)
        n+=1

def get_landmark(landmark_txt):
    f = open(landmark_txt, 'r', encoding = 'utf-8')
    words = []
    temp_landmark = []
    points_68 = []
    temp_line_sub = f.readlines()
    if len(temp_line_sub) < 5:
        landmark = temp_line_sub[2]
        points_list = landmark.split(',')
        for a in range(33):
            if a % 2 == 0:
                points_68.append([points_list[a * 2], points_list[a * 2 + 1]])
        for b in range(33, 43):
            points_68.append([points_list[b * 2], points_list[b * 2 + 1]])
        for c in range(43, 52):
            points_68.append([points_list[c * 2], points_list[c * 2 + 1]])
        for d in range(52, 64):
            points_68.append([points_list[d * 2], points_list[d * 2 + 1]])
        for e in range(84, 104):
            points_68.append([points_list[e * 2], points_list[e * 2 + 1]])
        words.append(points_68)
    else:
        if len(temp_line_sub) == 70:
            temp_length = len(temp_line_sub) - 1
        else:
            temp_length = len(temp_line_sub)
        for m in range(1, temp_length):
            temp_landmark.append(temp_line_sub[m].strip().split()[0])
            temp_landmark.append(temp_line_sub[m].strip().split()[1])
        words.append(temp_landmark)
            # temp_landmark.append(float(temp_line_sub[m].strip().split()[1]))
    return words[0]


def draw_img(image, landmark):
    # print(landmark)
    if len(image.shape) == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    h, w = image.shape[0], image.shape[1]
    for j in range(68):
        image = cv2.rectangle(image, (int(landmark[2 * j]), int(landmark[2 * j + 1])),
                             (int(landmark[2 * j]), int(landmark[2 * j + 1])),
                             (0, 255, 255), 3)
        cv2.putText(image, str(j), (int(landmark[2 * j]), int(landmark[2 * j + 1])), cv2.FONT_HERSHEY_SIMPLEX ,
                    0.2, (0, 0, 255), 1, cv2.LINE_AA)

    return image

import math
def distance(landmarks, index1, index2):
    return math.sqrt((landmarks[index1 *2] - landmarks[index2 * 2])*(landmarks[index1 *2] - landmarks[index2 * 2]) \
                     + (landmarks[index1 *2 + 1] - landmarks[index2 * 2 + 1])*(landmarks[index1 *2 + 1] - landmarks[index2 * 2 +  1]))

def get_mouth_distance(landmarks):
    landmarks = [int(i) for i in landmarks]
    topH = (distance(landmarks, 50, 61) + distance(landmarks, 51, 62) + distance(landmarks, 52, 63))/3
    bottomH = (distance(landmarks, 58, 67) + distance(landmarks, 57, 66) + distance(landmarks, 56, 65)) / 3
    mouthH = (distance(landmarks, 61, 67) + distance(landmarks, 62, 66) + distance(landmarks, 63, 65)) / 3
    dis = 2*mouthH /(topH + bottomH + 0.0000001)

    return dis

def select_open_mouth_img():
    # /mnt/sdb1/face_id_mulit_task-helen-sm-5.bak/dataset/0827_traindata/landmark_automation_001_503_r.txt
    # /mnt/sdb1/face_id_mulit_task-helen-sm-5.bak/dataset/0827_traindata/20200711_liongaze_landmark_68.txt
    image_list_file = '/mnt/sdb1/face_id_mulit_task-helen-sm-5.bak/dataset/0827_traindata/landmark_automation_001_503_r.txt'
    f = open(image_list_file,'r')
    image_list = f.readlines()
    fw = open('/mnt/sdb1/openMouth_data/openMouth_in_landmark_automation_001_503_r.txt', 'w')
    num = 0
    for i,image_name in enumerate(image_list):
        image_name = image_name.strip()
        if '.bmp' in image_name:
            landmark_txt = image_name.replace('.bmp', '.txt')
        else:
            landmark_txt = image_name.replace('.jpg', '.txt')
        landmark= get_landmark(landmark_txt)
        dis = get_mouth_distance(landmark)
        if dis > 0.52:
            num += 1
            print('mouth distance', dis, 'num: ' + str(num))
            fw.write(image_name + '\n')
            # image = cv2.imread(image_name)
            # image = draw_img(image, landmark)
            # cv2.imwrite('/mnt/sdb1/openMouth_data/data' + str(i)+'.jpg', image)
    fw.close()
    f.close()

def jpgToBmp(img_file_list, save_path):
    for fileName in img_file_list:
        # print('1', fileName.split('.')[-1])
        if fileName.split('.')[-1] == 'jpg':

            img = Image.open(fileName)
            newFileName = fileName.replace('jpg', 'bmp')
            print(newFileName)
            img.save(newFileName)


#copyfile(img_dir,'/mnt/sdb1/test_self_collect/test_part001/' + img_dir.split('/')[-1])
def make_no_face_landmark_anno():
    # root_path = '/mnt/sdb4/dataset/n_landmark_hand_0112'
    # save_path = '/mnt/sdb4/dataset/n_landmark_hand_0112/n_landmark_hand_0112.txt'
    root_path = '/mnt/sdb1/no_face_data/05_no_face_data/object365_test/'
    save_path = '/mnt/sdb1/no_face_data/05_no_face_data/no_face_data_from_objet365_test.txt'

    n_landmark_anno = '/mnt/sdb1/n_landmark/DF/2020_07_28_15_32_50_103/negative_153305783_num0_color_light_173.1_32.78_-77.94_14.37.txt'
    file_list = get_list(root_path, save_path)
    for file in file_list:
        # print(file)
        landmark_anno = file.replace('.jpg', '.txt')
        # print('landmark_anno', landmark_anno)
        copyfile(n_landmark_anno, landmark_anno)
    jpgToBmp(file_list, root_path)


def copy_landmark_anno():
    root_path = '/mnt/sda1/20200714_liongaze_landmark_68_process/' #20200711_liongaze_landmark_68  0827_abroad_landmark 20200714_liongaze_landmark_68_process landmark_automation_001_503_r
    save_path = '/mnt/sdb1/face_with_mask/landmark_mask_28/face_with_mask_28.txt'
    file_list = get_list(root_path, save_path)
    save_not_in_path = '/mnt/sdb1/face_with_mask/landmark_mask_28/save_not_in_path_28.txt'
    f = open(save_not_in_path, 'w')
    i  = 0
    for file in file_list:
        # print(file)/mnt/sdb1/landmark_68_new/
        landmark_anno = file.replace('.bmp', '.txt')
        print('landmark_anno', landmark_anno)
        copy_anno = landmark_anno.replace('/mnt/sda1/', '/mnt/sdb1/face_with_mask/landmark_mask_28/')
        print('copy_anno', copy_anno)
        try:
            copyfile(landmark_anno, copy_anno)
        except:
            i += 1
            print(i)
            f.write(copy_anno + '\n')
    print(i)

#landmark_mask_28
def make_list_file():
    # pass
    # root_path = '/mnt/sdb1/face_with_mask/landmark_mask_28/'
    # save_path = '/mnt/sdb1/face_with_mask/landmark_mask_28/face_with_mask_28_part.txt'
    # get_list(root_path, save_path, 0.5)  #remove open mask image with mask
    # get_abroad_list(root_path, save_path)

    # root_path = '/mnt/sdb1/face_with_mask/09_500Mface_mask/04_mobile_face'
    # save_path = '/mnt/sdb1/face_with_mask/09_500Mface_mask/04_mobile_face/09_500Mface_mask_04_mobile_face.txt'
    # get_list(root_path, save_path, 0.2)

    # root_path = '/mnt/sdb1/landmark_68_new2/09_500Mface/'
    # save_path = '/mnt/sdb1/landmark_68_new2/09_500Mface/09_500Mface_0.01.txt'
    # get_list(root_path, save_path, 0.01)

    # root_path = '/mnt/sdb1/test_procurement/part1'
    # save_path = '/mnt/sdb1/test_procurement/part1/procurement_part1.txt'
    # get_list(root_path, save_path, 0.01)

    # root_path = '/mnt/sdb1/test_self_collect/MlKit'
    # save_path = '/mnt/sdb1/test_self_collect/MlKit/MlKit_self_collect.txt'
    # get_list(root_path, save_path, 0.1)

    # root_path = '/mnt/sdb1/no_face_data'
    # save_path = '/mnt/sdb1/no_face_data/no_face_data.txt'
    # get_list(root_path, save_path, 1)

    # root_path = '/mnt/sdb4/dataset/01_landmark_68_0112/09_500Mface'
    # save_path = '/mnt/sdb1/no_face_data/01_landmark_68_09_500Mface_0112.txt'
    # get_list(root_path, save_path, 1)

    # root_path = '/mnt/disk1/masiming/03_data/03_face_mask_data/02_masked_result_data/mask_face_crop_data/0827_traindata/'
    # save_path = '/mnt/disk1/masiming/03_data/03_face_mask_data/02_masked_result_data/mask_face_crop_data/0827_traindata/0827_traindata_mask_face_crop_data.txt'
    # get_list(root_path, save_path,label= '1 0',part_rate=1)
    # root_path = '/mnt/disk1/masiming/03_data/03_face_mask_data/02_masked_result_data/mask_face_crop_data/09_500Mface/'
    # save_path = '/mnt/disk1/masiming/03_data/03_face_mask_data/02_masked_result_data/mask_face_crop_data/09_500Mface/09_500Mface_mask_face_crop_data.txt'
    # get_list(root_path, save_path, label='1 0', part_rate=1)

    # root_path = '/mnt/disk_extra/12_face_recognition/03_ms1m-arcface/faces_emore/imgs'
    # save_path = '/home/zhangpeng/workspace/03_livness/03_occlusion/data/faces_ms1m_112x112_face_crop_data.txt'
    # get_list(root_path, save_path, label='0 0', part_rate=1)
    # root_path = '/mnt/disk_extra/12_face_recognition/09_500Mface/clean_hiai/merge/imgs'
    # save_path = '/home/zhangpeng/workspace/03_livness/03_occlusion/data/faces_500m_112x112_face_crop_data.txt'
    # get_list(root_path, save_path, label='0 0', part_rate=1)


    # root_path = '/mnt/disk1/masiming/03_data/07_sunglasss_data/sunglass_data_copy'
    # save_path = '/home/zhangpeng/workspace/03_livness/03_occlusion/data/sunglass_data_copy_face_crop_data.txt'
    # get_list(root_path, save_path, label='0 1', part_rate=1)

    # label: mask sung 1 1
    # root_path = '/mnt/disk1/masiming/03_data/06_occlusion_data/generate_sunglasses_data'
    # save_path = '/home/zhangpeng/workspace/03_livness/03_occlusion/data/generate_sunglasses_data_faces_500m.txt'
    # get_list(root_path, save_path, label='0 1', part_rate=1)

    # root_path = '/mnt/disk1/masiming/03_data/06_occlusion_data/generate_sunglasses_mask_data'
    # save_path = '/home/zhangpeng/workspace/03_livness/03_occlusion/data/generate_sunglasses_mask_data_faces_500m.txt'
    # get_list(root_path, save_path, label='1 1', part_rate=1)

    # root_path = '/mnt/disk1/masiming/03_data/07_sunglasss_data/sunglass_data_copy2'
    # save_path = '/data/datapan1/occlusion/data/sunglass_data_copy2_face_crop_data.txt'
    # get_list(root_path, save_path, label='0 1', part_rate=1)

    # root_path = '/mnt/disk1/masiming/03_data/03_face_mask_data/06_new_mask_data/crop_data/0827_traindata'
    # save_path = '/data/datapan1/occlusion/data/06_new_mask_data_0827_traindata_0210.txt'
    # get_list(root_path, save_path, label='1 0', part_rate=1)

    # root_path = '/data/datapan1/occlusion/data/sunglass_data_copy3'
    # save_path = '/data/datapan1/occlusion/data/sunglass_data_copy3_face_crop_data.txt'
    # get_list(root_path, save_path, label='0 1', part_rate=1)

    # root_path = '/mnt/disk1/masiming/03_data/03_face_mask_data/07_resize_mask_0827traindata/crop_data/0827_traindata'
    # save_path = '/data/datapan1/occlusion/data/07_resize_mask_0827traindata_0214.txt'
    # get_list(root_path, save_path, label='1 0', part_rate=1)

    # root_path = '/mnt/disk1/masiming/03_data/03_face_mask_data/08_skin_mask_0827traindata/crop_data/'
    # save_path = '/data/datapan1/occlusion/data/08_skin_mask_0827traindata_0215.txt'
    # get_list(root_path, save_path, label='1 0', part_rate=1)

    # root_path = "/mnt/data_60_27_share/datasets/"
    # save_path = '/data/datapan1/occlusion/data/normal_gls_0217_datasets.txt'
    # get_list(root_path, save_path, label='0 0', part_rate=1)

    # root_path = "/mnt/data_60_27_share/cp_2_no_sunglasses2/"
    # save_path = '/data/datapan1/occlusion/data/normal_gls_0217_cp_2_sunglasses2.txt'
    # get_list(root_path, save_path, label='0 0', part_rate=1)

    # root_path = "/mnt/disk1/masiming/03_data/03_face_mask_data/09_skin_resize_mask/"
    # save_path = '/data/datapan1/occlusion/data/09_skin_resize_mask.txt'
    # get_list(root_path, save_path, label='1 0', part_rate=1)

    # root_path = "/mnt/data_60_27_share/cp_2_no_sunglasses/"
    # save_path = '/data/datapan1/occlusion/data/normal_gls_0217_cp_sunglasses.txt'
    # get_list(root_path, save_path, label='0 0', part_rate=1)

    # root_path = "/mnt/data_60_27_share/cp_2_no_sunglasses2/"
    # save_path = '/data/datapan1/occlusion/data/normal_gls_0222_cp_sunglasses2.txt'

    # root_path = "/mnt/data_60_27_share/cp_2_no_sunglasses3/"
    # save_path = '/data/datapan1/occlusion/data/normal_gls_0222_cp_sunglasses3.txt'

    # root_path = "/mnt/data_60_27_share/cp_2_no_sunglasses/"
    # save_path = '/data/datapan1/occlusion/data/normal_gls_0222_cp_sunglasses.txt'
    # get_list(root_path, save_path, label='0 0', part_rate=1)

    # root_path = "/mnt/disk1/masiming/03_data/03_face_mask_data/10_hand_cover_data/0827_traindata/0827_abroad_landmark/"
    # save_path = '/mnt/data_10.137.24.27/nfs/occlusion/data/handCover_mask_0303_foreigner.txt'
    # get_list(root_path, save_path, label='1 0', part_rate=1)

    # root_path = "/mnt/data_60_169_share/yaohongmiao/hand_occlusion_data/crop_data/0827_traindata/0827_abroad_landmark" \
    #             "/hairHand_coverEye_foreigner/"
    # save_path = '/mnt/data_60_27_share/occlusion/data/hairHand_eyeCover_0304_foreigner.txt'
    # get_list(root_path, save_path, label='0 1', part_rate=1)

    # root_path = "/mnt/data_60_169_share/yaohongmiao/hand_occlusion_data/crop_data/0827_traindata/0827_abroad_landmark/handCover_mask_foreigner/checkNew"
    # save_path = '/mnt/data_60_27_share/occlusion/data/handCover_check_0304_foreigner.txt'
    # get_list(root_path, save_path, label='1 0', part_rate=1)

    # root_path = "/mnt/data_60_169_share/yaohongmiao/hand_occlusion_data/crop_data/0827_traindata" \
    #             "/0307_landmark_automation_001_503_r_face/"
    # save_path = '/mnt/data_60_27_share/occlusion/data/handCover_eye_0308_chinese.txt'
    # get_list(root_path, save_path, label='0 1', part_rate=1)

    # root_path = "/mnt/data_60_169_share/yaohongmiao/hand_occlusion_data/crop_data/0827_traindata" \
    #             "/landmark_automation_001_503_r/left_check"
    # save_path = '/mnt/data_60_27_share/occlusion/data/handCover_check_0308_chinese.txt'
    # get_list(root_path, save_path, label='1 0', part_rate=1)

    # root_path = "/mnt/data_60_169_share/yaohongmiao/hand_occlusion_data/crop_data/0827_traindata" \
    #             "/0308_landmark_automation_001_503_r_face/face"
    # save_path = '/mnt/data_60_27_share/occlusion/data/handCover_eye_0308_chinese_new.txt'
    # get_list(root_path, save_path, label='0 1', part_rate=1)

    # root_path = "/mnt/data_60_169_share/yaohongmiao/hand_occlusion_data/crop_data/0827_traindata" \
    #             "/0308_0827_abroad_landmark/eye"
    # save_path = '/mnt/data_60_27_share/occlusion/data/handCover_eye_0308_foreigner_new.txt'
    # get_list(root_path, save_path, label='0 1', part_rate=1)

    root_path = "/mnt/disk1/masiming/03_data/03_face_mask_data/10_hand_cover_data/crop_data/0827_traindata/"
    save_path = '/mnt/data_60_27_share/occlusion/data/handMask_0308_allpeople.txt'
    get_list(root_path, save_path, label='1 0', part_rate=1)

def show_landmark():
    image_list_file = '/mnt/sdb1/face_with_mask/sungls_mask/landmark_automation_001_503_r_maskdata-2.txt'
    f = open(image_list_file, 'r')
    image_list = f.readlines()
    for i,image_name in enumerate(image_list):
        image_name = image_name.strip()
        if '.bmp' in image_name:
            landmark_txt = image_name.replace('.bmp', '.txt')
        else:
            landmark_txt = image_name.replace('.jpg', '.txt')
        landmark= get_landmark(landmark_txt)
        image = cv2.imread(image_name)
        image = draw_img(image, landmark)
        cv2.imwrite('/mnt/sdb1/face_with_mask/sungls_mask/mask_landmark/' + str(i)+'.jpg', image)

# /mnt/sdb1/face_with_mask/landmark_mask_28
# redress mask mouth landmark as a line
def redress_mask_landmark():
    image_list_file = '/mnt/sdb1/face_with_mask/landmark_mask_28/face_with_mask_28_part.txt'
    f = open(image_list_file, 'r')
    image_list = f.readlines()
    for i, image_name in enumerate(image_list[:10]):
        image_name = image_name.strip()
        print(image_name)
        if '.bmp' in image_name:
            landmark_txt = image_name.replace('.bmp', '.txt')
        else:
            landmark_txt = image_name.replace('.jpg', '.txt')
        landmark = get_landmark(landmark_txt)
        print(len(landmark))
        landmark = [int(i) for i in landmark]
        assert len(landmark) == 68 * 2
        landmark[61 * 2], landmark[67 * 2] = int((landmark[61 * 2] + landmark[67 * 2]) / 2), int((
            landmark[61 * 2] + landmark[67 * 2]) / 2)
        landmark[62 * 2], landmark[66 * 2] = int((landmark[62 * 2] + landmark[66 * 2]) / 2), int((
            landmark[62 * 2] + landmark[66 * 2]) / 2)
        landmark[63 * 2], landmark[65 * 2] = int((landmark[63 * 2] + landmark[65 * 2]) / 2), int((
            landmark[63 * 2] + landmark[65 * 2]) / 2)
        landmark[61*2+1], landmark[67*2+1] = int((landmark[61*2+1] + landmark[67*2+1]) / 2), int((landmark[61*2+1] + landmark[67*2+1]) / 2)
        landmark[62*2+1], landmark[66*2+1] = int((landmark[62*2+1] + landmark[66*2+1]) / 2), int((landmark[62*2+1] + landmark[66*2+1]) / 2)
        landmark[63*2+1], landmark[65*2+1] = int((landmark[63*2+1] + landmark[65*2+1]) / 2), int((landmark[63*2+1] + landmark[65*2+1]) / 2)

        image = cv2.imread(image_name)
        image = draw_img(image, landmark)
        cv2.imwrite('/mnt/sdb1/face_with_mask/sungls_mask/mask_landmark/' + str(i) + '.jpg', image)
        f_l = open(landmark_txt, 'w')
        f_l.write('68' + '\n')
        for j in range(68):
            line = str(landmark[2 * j]) + ' ' + str(landmark[2 * j + 1])
            f_l.write(line + '\n')
        f_l.close()
        landmark = get_landmark(landmark_txt)
        image = draw_img(image, landmark)
        cv2.imwrite('/mnt/sdb1/face_with_mask/sungls_mask/mask_landmark/redress' + str(i) + '.jpg', image)




if __name__ == '__main__':
    
    video_dir='/data/7896d2099dc644e7a6cac218f88fb0d7/onnx/video_data/mask/'
    save_path='/data/7896d2099dc644e7a6cac218f88fb0d7/onnx/video_data_cap/mask/'
    
#     video_dir='/data/7896d2099dc644e7a6cac218f88fb0d7/onnx/video_data/maskCover/'
#     save_path='/data/7896d2099dc644e7a6cac218f88fb0d7/onnx/video_data_cap/maskCover/'    
    
#     video_dir='/data/7896d2099dc644e7a6cac218f88fb0d7/onnx/video_data/eyeCover/'
#     save_path='/data/7896d2099dc644e7a6cac218f88fb0d7/onnx/video_data_cap/eyeCover/'    
    
    
    
    
    
    
    test_get_frame(video_dir,save_path)
    
    
    
    # make_no_face_landmark_anno()
#     make_list_file()
    # file_list = '/data/datapan1/occlusion/data/test_file.txt'
    # save_list = '/data/datapan1/occlusion/data/test_file_new.txt'
    # replace_list(file_list, save_list)
    # redress_mask_landmark()
    # select_open_mouth_img()

    # show_landmark()
    # copy_landmark_anno()

    # image_name = './957597265737000_close_640_480_rgb_47981.bmp'
    # landmark_txt = image_name.replace('.bmp', '.txt')
    # landmark = get_landmark(landmark_txt)
    # image = cv2.imread(image_name)
    # image = draw_img(image, landmark)
    # cv2.imwrite('/mnt/sdb1/face_with_mask/sungls_mask/mask_landmark/test.jpg', image)