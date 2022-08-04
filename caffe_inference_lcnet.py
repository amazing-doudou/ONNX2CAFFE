import sys
import os
#sys.path.append('/opt/caffe/python')
import numpy as np
import cv2 as cv
import caffe
import random
import math
sys.path.append('/data/7896d2099dc644e7a6cac218f88fb0d7/onnx/occlusion/')
from caffe_inference_occlusion_lcnet import occlusion_inference
def _sigmoid(x):
    #print(x)
    return 1 / (1 + np.exp(-x))

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #print(dets)
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]
    scores = dets[:, 0]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def decode(features, threshold = 0.5, occ_threshold = 0.5, nms_threshold = 0.35, image_shape = [224, 224]):
    stride = 16#cfg.MODEL.HM_STRIDE
    hm = _sigmoid(features[0][0,0,:,:])#.sigmoid_().data.cpu().numpy()[0,0,:,:]
    wh = features[1][0]#.data.cpu().numpy()[0]
    xy = features[2][0]#.data.cpu().numpy()[0]
    if True:
        landmark = features[3][0]#.data.cpu().numpy()[0]
    #if True:
        #occlusion = _sigmoid(features[4][0])#.sigmoid_().data.cpu().numpy()[0]
    #print(hm.shape)
    #print(features[0][0,0,:,:])
    #print(hm)
    c0, c1 = np.where(hm > threshold)
    print('hm > threshold', hm[c0,c1])
    occ_threshold = occ_threshold
    #print(image_shape)
    boxes = []
    if len(c0) > 0:
        for i in range(len(c0)):
            index_0 = c0[i]
            index_1 = c1[i]
            w, h = max(np.exp(wh[0, index_0,index_1]) * stride, stride), \
                   max(np.exp(wh[1, index_0,index_1]) * stride, stride)
            x, y = np.log(xy[0,index_0,index_1]), np.log(xy[1,index_0,index_1])
            ct_x = (index_1 + x) * stride
            ct_y = (index_0 + y) * stride
            print('x,y,ct_x,ct_y: ',x,y,ct_x,ct_y)
            tmp_box = [hm[index_0, index_1],(ct_x - w / 2) / image_shape[1], (ct_y - h / 2) / image_shape[0], \
                (ct_x + w / 2) / image_shape[1], (ct_y + h / 2) / image_shape[0]]
            #print(tmp_box)
            if True:
                landmark_list = []
                for i in range(68):
                    lx = landmark[i*2, index_0, index_1] * stride   + ct_x
                    ly = landmark[i*2+1, index_0, index_1] *   stride   + ct_y
                    landmark_list.append(lx/image_shape[1])
                    landmark_list.append(ly/image_shape[0])
                    
                '''
                lx0 = landmark[0, index_0, index_1] * stride  + ct_x- w / 2
                ly0 = landmark[1, index_0, index_1] * stride  + ct_y- h / 2
                lx1 = landmark[2, index_0, index_1] * stride  + ct_x- w / 2
                ly1 = landmark[3, index_0, index_1] * stride  + ct_y- h / 2
                lx2 = landmark[4, index_0, index_1] * stride  + ct_x- w / 2
                ly2 = landmark[5, index_0, index_1] * stride  + ct_y- h / 2
                lx3 = landmark[6, index_0, index_1] * stride  + ct_x- w / 2
                ly3 = landmark[7, index_0, index_1] * stride  + ct_y- h / 2
                lx4 = landmark[8, index_0, index_1] * stride  + ct_x- w / 2
                ly4 = landmark[9, index_0, index_1] * stride  + ct_y- h / 2
                landmark_list = [lx0/ image_shape[1], ly0/ image_shape[0], lx1/ image_shape[1], ly1/ image_shape[0], \
                                lx2/ image_shape[1], ly2/ image_shape[0], lx3/ image_shape[1], ly3/ image_shape[0], \
                                lx4/ image_shape[1], ly4/ image_shape[0]]
                '''
                #print(landmark_list)
                tmp_box.append(landmark_list)

            boxes.append(tmp_box)
    if len(boxes) == 0:
        return boxes
    boxes =  np.array(boxes)
    keep = py_cpu_nms(boxes, nms_threshold)
    boxes = boxes[keep]
    return boxes
    


def load_model(protofile, model_weights):
    # caffe.set_device(5)
    # caffe.set_mode_gpu()
    caffe.set_mode_cpu()
    net = caffe.Net(protofile, model_weights, caffe.TEST)
    return net

def run_landmark_model(net, img):
    #img = cv.resize(img, (224, 224))
    image = cv.resize(img, (168, 224))
    img = np.zeros((224, 224, 1), dtype = "uint8")
    img[:224, :168,:] = (image[:,:,0:1] / 3 + image[:,:,1:2] / 3 + image[:,:,2:3] / 3)
    
    img = np.float32(img)
    img -= 128
    img /= 128.0
    print(img.shape)
    #img = np.expand_dims(img, axis=-1)
    #print(img.shape)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    #print(img.shape)
    net.blobs['0'].data[...] = img
    #print()
    net.forward()
    #test = net.blobs["backbone.input_block.0.1.pool1"].data
    #test2 = net.blobs["backbone.input_block_9.0.0"].data
    #print(test2)
    #print(test)
    # hm = net.blobs["384"].data
    # wh = net.blobs["391"].data
    # xy = net.blobs["398"].data
    # landmark = net.blobs["411"].data

    hm = net.blobs["heat_map_head"].data
    wh = net.blobs["local_wh_head"].data
    xy = net.blobs["local_xy_head"].data
    landmark = net.blobs["landmark_head"].data

    #occ = net.blobs["occlusion_head"].data
    #features = [hm, wh, xy, landmark, occ]
    features = [hm, wh, xy, landmark]
    # print(features)
    return decode(features)

def get_box_img(image, boxes):
    h, w = image.shape[0], image.shape[1]
    imgs = []
    for i,b in enumerate(boxes):
        #(int(b[1] * w*224/168), int(b[2] * h)), (int(b[3] * w*224/168), int(b[4] * h))
        #(int(b[1] * w*224/168), int(b[2] * h)), (int(b[3] * w*224/168), int(b[4] * h))
        # y x
        # print(int(b[1] * w*224/168), int(b[2] * h),int(b[3] * w*224/168), int(b[4] * h))
        box = np.array([int(b[1] * w*224/168), int(b[2] * h),int(b[3] * w*224/168), int(b[4] * h)])
        box = box * (box>0)
        img = image[box[1]:box[3],box[0]: box[2],:]
        # cv.imwrite('test.bmp', img)
        imgs.append(img)
    return imgs

def draw_img(image, boxes):
    if len(image.shape) == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    h, w = image.shape[0], image.shape[1]
    # print(h, w)
    i = 0
    boxes_num = len(boxes)
    for b in boxes:
        i += 1
        image = cv.rectangle(image, (int(b[1] * w*224/168), int(b[2] * h)), (int(b[3] * w*224/168), int(b[4] * h)), \
            (0, 0, 255), 1)
        

        if True:
            landmark = b[5]
            for j in range(68):


                image = cv.rectangle(image, (int(landmark[2 * j] * w*224/168), int(landmark[2 * j + 1] * h)), \
                (int(landmark[2 * j] * w*224/168), int(landmark[2 * j + 1] * h)), \
                (0, 255, 0), 3)
        
    return image

def put_score_occlusion(image, boxes):
    if len(image.shape) == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    h, w = image.shape[0], image.shape[1]
    b = boxes
    image = cv.rectangle(image, (int(b[1] * w * 224 / 168), int(b[2] * h)), (int(b[3] * w * 224 / 168), int(b[4] * h)), \
                         (0, 0, 255), 1)

def get_list(data_path, part_rate=1):
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
    return save_list

def test_landmark():
    # path = './hand'
    # path = '/mnt/sda1/onnx/landmark.bak/test_img/MLKit_self_collect/20211102112331'
    path = '/mnt/sda1/onnx/landmark.bak/test_img/face_synthetics_data/datasets100000'
    model_name = '0112_1channel_5e-3_wing_loss_adamw_lcnet_landmark_small_v2_add_openmouth_data_mask_scale1'
    protofile = "../model_2d/landmark/" + model_name + ".prototxt"
    model_weights = "../model_2d/landmark/" + model_name + ".caffemodel"
    filename = os.listdir(path)
    net = load_model(protofile, model_weights)
    for i in range(len(filename)):
        img_path = path + '/' + filename[i]
        # img_path = "./20200519_13_51_36_685-st_test_dump_12.bmp"
        print(img_path)

        # img = cv.imread(img_path, 0)
        img = cv.imread(img_path)
        if img is None:
            continue
        copy_image = img.copy()

        result_list = run_landmark_model(net, img)
        if len(result_list) == 0:
            continue
        img = draw_img(copy_image, result_list)
        save_path = os.path.join('./face_synthetics_data', model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv.imwrite(os.path.join(save_path, filename[i]), img)

def test_landmark_occlusion():
    # path = '/mnt/sda1/onnx/landmark.bak/test_img/test_data/normal/'
#     path = '/mnt/sda1/onnx/occlusion/test_data/IMFD/data/'
#     path='/data/7896d2099dc644e7a6cac218f88fb0d7/onnx/video_data_cap/normal/'
#     path='/data/7896d2099dc644e7a6cac218f88fb0d7/onnx/video_data_cap/eyeCover/'
#     path='/data/7896d2099dc644e7a6cac218f88fb0d7/onnx/video_data_cap/maskCover/'
    path='/data/7896d2099dc644e7a6cac218f88fb0d7/onnx/video_data_cap/mask/'
    
    model_dir='./model/'

    landmark_model_name = '0112_1channel_5e-3_wing_loss_adamw_lcnet_landmark_small_v2_add_openmouth_data_mask_scale1'
    landmark_protofile = "/data/7896d2099dc644e7a6cac218f88fb0d7/onnx/model_2d/landmark/" + landmark_model_name + ".prototxt"
    landmark_model_weights = "/data/7896d2099dc644e7a6cac218f88fb0d7/onnx/model_2d/landmark/" + landmark_model_name + ".caffemodel"

#     occlusion_model_name = 'model_002_3800_0306_train255w'
#     occlusion_model_name = 'model_002_01650_0309_train280w_unzip'
#     occlusion_model_name = 'model_002_07400_0306_255w_unzip'
#     occlusion_model_name = 'model_027_00150_0308_train5w_unzip'
#     occlusion_model_name='model_004_01250_0311_train50w_unzip'
    occlusion_model_name='model_002_08000_0309_train280w_unzip'
    
    occlusion_protofile =  model_dir + occlusion_model_name + '.prototxt'
    occlusion_model_weights =  model_dir + occlusion_model_name + '.caffemodel'
    net = load_model(landmark_protofile, landmark_model_weights)
    occlusion_net = load_model(occlusion_protofile, occlusion_model_weights)
#     sunglasses_error_num = 0
#     sunglasses_right_num = 0
#     mask_error_num = 0
#     mask_right_num = 0

    normal_right_num=0
    normal_wrong_num=0
    eyeCover_right_num=0
    eyeCover_wrong_num=0
    maskCover_right_num=0
    maskCover_wrong_num=0
    landmarkNone_num=0

    file_list = get_list(path)
    file_list = sorted(file_list)
    
    save_path='/data/7896d2099dc644e7a6cac218f88fb0d7/onnx/video_data_cap/testResult/{}'.format(occlusion_model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in ('normal_right','normal_wrong','eyeCover_right','eyeCover_wrong','maskCover_right','maskCover_wrong','landmarkNone','mask','sungls'):
        path_temp=os.path.join(save_path,i)
        if not os.path.exists(path_temp):
            os.makedirs(path_temp)
        
    for i in range(len(file_list)):
        img_path = file_list[i]
        # img_path = "/mnt/sda1/onnx/landmark.bak/test_img/MLKit_self_collect/20211102112331/466_111_229_317_0.61_0.01.png"

        # img = cv.imread(img_path, 0)
        img = cv.imread(img_path)
        if img is None:
            continue
        copy_image = img.copy()

        result_landmark_list = run_landmark_model(net, img)
        if len(result_landmark_list) == 0:
            landmarkNone_num+=1
            cv.imwrite(os.path.join(save_path+'/landmarkNone/', str(i) + '_'.join(file_list[i].split('/')[-2:])), copy_image)
            continue
        img = draw_img(copy_image, result_landmark_list)
        face_images = get_box_img(copy_image, result_landmark_list)
        
        for face_image in face_images:
            result_list = occlusion_inference(occlusion_net, occlusion_model_weights,face_image)
            if path.endswith('normal/'):
                if result_list[0][0]<=0.5 and result_list[0][1]<=0.5:
                    normal_right_num+=1
                    print('occlusion_result',result_list)
                    cv.imwrite(os.path.join(save_path+'/normal_right/', str(i) + '_'.join(file_list[i].split('/')[-2:])), copy_image)
                else:
                    normal_wrong_num+=1
                    print('occlusion_result',result_list)
                    cv.imwrite(os.path.join(save_path+'/normal_wrong/', str(round(result_list[0][0],4))+'_'+str(round(result_list[0][1],4))+'_'+str(i) + '_'.join(file_list[i].split('/')[-2:])), copy_image)
                               
            elif path.endswith('eyeCover/') or path.endswith('sungls/'):
                if result_list[0][1]>0.5:
                    eyeCover_right_num+=1
                    print('occlusion_result',result_list)
                    cv.imwrite(os.path.join(save_path+'/eyeCover_right/', str(i) + '_'.join(file_list[i].split('/')[-2:])), copy_image)
                else:
                    eyeCover_wrong_num+=1
                    print('occlusion_result',result_list)
                    cv.imwrite(os.path.join(save_path+'/eyeCover_wrong/', str(round(result_list[0][0],4))+'_'+str(round(result_list[0][1],4))+'_'+str(i) + '_'.join(file_list[i].split('/')[-2:])), copy_image)

                    
            elif path.endswith('maskCover/') or path.endswith('mask/'):
                if result_list[0][0]>0.5 and result_list[0][1]<=0.5:
                    maskCover_right_num+=1
                    print('occlusion_result',result_list)
                    cv.imwrite(os.path.join(save_path+'/maskCover_right/', str(i) + '_'.join(file_list[i].split('/')[-2:])), copy_image)
                else:
                    maskCover_wrong_num+=1
                    print('occlusion_result',result_list)
                    cv.imwrite(os.path.join(save_path+'/maskCover_wrong/', str(round(result_list[0][0],4))+'_'+str(round(result_list[0][1],4))+'_'+str(i) + '_'.join(file_list[i].split('/')[-2:])), copy_image)

            # img = cv2.putText(img, str(i), (int(p1), int(p2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#             if result_list[0][0] < 0.5 and result_list[0][1] < 0.5:
#                 print('occlusion_result', result_list)

#             save_path = os.path.join('./test_img/test_data/sunglasses_error/IMFD/', occlusion_model_name)
#             if not os.path.exists(save_path):
#                 os.makedirs(save_path)
#             if result_list[0][1] <= 0.5:
#                 sunglasses_error_num += 1
#                 print(str(i) + '_'.join(file_list[i].split('/')[-2:]), result_list[0])
#                 cv.imwrite(os.path.join(save_path, str(i) + '_'.join(file_list[i].split('/')[-2:])), copy_image)

#             save_path = os.path.join('./test_img/test_data/sunglasses_right/IMFD/', occlusion_model_name)
#             if not os.path.exists(save_path):
#                 os.makedirs(save_path)
#             if result_list[0][1] > 0.5:
#                 sunglasses_right_num += 1
#                 cv.imwrite(os.path.join(save_path, str(i) + '_'.join(file_list[i].split('/')[-2:])), copy_image)

#             save_path = os.path.join('./test_img/test_data/mask_error/IMFD/', occlusion_model_name)
#             if not os.path.exists(save_path):
#                 os.makedirs(save_path)
#             if result_list[0][0] <= 0.5:
#                 mask_error_num += 1
#                 cv.imwrite(os.path.join(save_path, str(i) + '_'.join(file_list[i].split('/')[-2:])), copy_image)
#             save_path = os.path.join('./test_img/test_data/mask_right/IMFD/', occlusion_model_name)
#             if not os.path.exists(save_path):
#                 os.makedirs(save_path)
#             if result_list[0][0] > 0.5:
#                 mask_right_num += 1
#                 print(str(i) + '_'.join(file_list[i].split('/')[-2:]), result_list[0])
#                 cv.imwrite(os.path.join(save_path, str(i) + '_'.join(file_list[i].split('/')[-2:])), copy_image)
        # print(result_list)
    print('image number: ', len(file_list))
    print('normal_right_num: ',normal_right_num,'normal_wrong_num: ',normal_wrong_num)
    print('eyeCover_right_num: ', eyeCover_right_num, 'eyeCover_wrong_num: ', eyeCover_wrong_num)
    print('maskCover_right_num: ', maskCover_right_num, 'maskCover_wrong_num: ', maskCover_wrong_num)
    print('landmarkNone_num: ',landmarkNone_num)


if __name__ == "__main__":

#     test_landmark()
    test_landmark_occlusion()

