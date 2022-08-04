from __future__ import print_function
import sys
sys.path.append('/data/7896d2099dc644e7a6cac218f88fb0d7/onnx/')
import caffe
import onnx
import numpy as np
from caffe.proto import caffe_pb2
caffe.set_mode_cpu()
from onnx2caffe._transformers import ConvAddFuser,ConstantsToInitializers
from onnx2caffe._graph import Graph

import onnx2caffe._operators as cvt
import onnx2caffe._weightloader as wlr
from onnx2caffe._error_utils import ErrorHandling
from collections import OrderedDict
from onnx import shape_inference
import importlib
import torch.nn as nn

import importlib
import os
import torch.nn.functional as F

from torch.autograd import Variable

import torchvision
import torch
#from utils import meter

import pdb


transformers = [
    ConstantsToInitializers(),
    ConvAddFuser(),
]


def convertToCaffe(graph, prototxt_save_path, caffe_model_save_path):

    exist_edges = []
    layers = []
    exist_nodes = []
    err = ErrorHandling()
    
    print('mao0724 开始遍历onnx的graph：')
    for i in graph.inputs:
        edge_name = i[0]
        input_layer = cvt.make_input(i)
        layers.append(input_layer)
        exist_edges.append(i[0])
        graph.channel_dims[edge_name] = graph.shape_dict[edge_name][1]

    print('mao0724 开始遍历onnx的graph：')
    for id, node in enumerate(graph.nodes):
        node_name = node.name
        op_type = node.op_type
        inputs = node.inputs
        inputs_tensor = node.input_tensors
        input_non_exist_flag = False

        for inp in inputs:
            if inp not in exist_edges and inp not in inputs_tensor:
                input_non_exist_flag = True
                break
        if input_non_exist_flag:
            continue

        if op_type not in cvt._ONNX_NODE_REGISTRY:
            # err.unsupported_op(node)
            continue
        converter_fn = cvt._ONNX_NODE_REGISTRY[op_type]
        layer = converter_fn(node,graph,err)
        if type(layer)==tuple:
            for l in layer:
                layers.append(l)
        else:
            layers.append(layer)
#         print('maom0724 打印当前层layer内容type：', layer.type)
#         print('maom0724 打印当前层layer内容bottom：', layer.bottom)
#         print('maom0724 打印当前层layer内容top：', layer.top)
#         print('maom0724 打印当前层layer内容name：', layer.name)
        outs = node.outputs
        for out in outs:
            exist_edges.append(out)

    net = caffe_pb2.NetParameter()
    for id,layer in enumerate(layers):
        layers[id] = layer._to_proto()
    net.layer.extend(layers)

    with open(prototxt_save_path, 'w') as f:
        print(net,file=f)

    caffe.set_mode_cpu()
    deploy = prototxt_save_path
    net = caffe.Net(deploy,
                    caffe.TEST)

    print('mao0724 开始对op层加载参数：')
    for id, node in enumerate(graph.nodes):
        node_name = node.name
        op_type = node.op_type
        inputs = node.inputs
        inputs_tensor = node.input_tensors
        input_non_exist_flag = False
        if op_type not in wlr._ONNX_NODE_REGISTRY:
            #err.unsupported_op(node)
            continue
        converter_fn = wlr._ONNX_NODE_REGISTRY[op_type]
        
        converter_fn(net, node, graph, err)
        print('mao0724 打印每层加载参数以后的net：', net)
        
    net.save(caffe_model_save_path)
    return net

def getGraph(onnx_path):
    model = onnx.load(onnx_path)
    model = shape_inference.infer_shapes(model)
    model_graph = model.graph
    graph = Graph.from_onnx(model_graph)
    graph = graph.transformed(transformers)
    graph.channel_dims = {}

    return graph
def get_torch_model_input(model_path):
    #module = importlib.import_module("model_generator.resnet_single_online")
    #module = importlib.import_module("model_generator.densenet")
    # module = importlib.import_module("model_generator.LCNet")
    module = importlib.import_module("model_generator.LCNet_2")
    device = torch.device("cuda")
    input_dim = 1
    # pretrained = ''
    #model = module.resnetcaffe18_single_fc_bn(pretrained,last_layers=True)
    # model = module.LCNet_small(input_size=128,input_dim=3)
    model = module.LCNet_small_SE(input_dim=1, input_size=128, num_tasks=2, version=1.3)
    # model = module.LCNet_small_noSE(input_dim=1, input_size=128, num_tasks=2, version=1.3)

    # from model_generator.repvgg import repvgg_model_convert
    # model = repvgg_model_convert(model)
    print('mao0724 model：', model)
    #m conodel = module.DenseNet121()
    '''
    model.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2)
    model.fc_1 = nn.Linear(4608, 64)

    #model.relu = nn.functional.relu(x)
    #model.drop_out = nn.functional.dropout(x,p=0.5,training=self.training)
    model.fc_2 = nn.Linear(64, 2)
    '''
    
    #for densenet model
    #model.avg_pool2d = F.avg_pool2d(7)
    #for densenet model
    model.eval()
    # model.load_state_dict(torch.load(model_path), strict=True)

    #model.cpu()scp
    model.cuda()
    #model = nn.DataParallel(model)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    
    model.load_state_dict(torch.load(model_path),strict=False)
    # for k,v in torch.load(model_path).items():
    #     print(k, v)
#     model.cuda()

    batch_size = 1
    channels = input_dim
    height = 128
    width = 128
    images = Variable(torch.ones(batch_size, channels, height, width))
    images=images.to(device)
    return images,model

def view_caffe_weights(protofile, model_weights):
    net = caffe.Net(protofile, model_weights, caffe.TEST)
    params = net.params
    # print(params)
    for k, v in params.items():
        print(k, v[0].data)


if __name__ == "__main__":
    '''
    onnx_path = sys.argv[1]
    prototxt_path = sys.argv[2]
    caffemodel_path = sys.argv[3]
    ''' 
    #model_path = '/mnt/sda1/onnx/model_2d/model_40.pth'  #model_1.pth  4ttmodel.ptn model_2.pth
#     model_name='0220_100_train_test_normal_sugl_002_01600'
#     model_name='model_001_00000_unzip'
#     model_name='model_002_3800_0306_train255w'
#     model_name='model_002_01650_0309_train280w_unzip'
#     model_name='model_002_07400_0306_255w_unzip'

    model_name='model_002_08000_0309_train280w_unzip'



    model_dir='./model/' #LCNetBackBone_small.pth
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path=os.path.join(model_dir,model_name+'.pth')
    var,model=get_torch_model_input(model_path)
    for k, v in model.state_dict().items():
        print(k, v)
    #
    # #onnx_path='/mnt/sda1/onnx/model_2d/resnet18_bn'+model_name+'.onnx'
    # #onnx_path='/mnt/sda1/onnx/model_2d/densenet_'+model_name+'.onnx'
    onnx_path=model_dir+model_name+'.onnx'
    # torch.onnx.export(model, var,onnx_path, verbose=True)
    torch.onnx.export(model.module,var,onnx_path, verbose=True)
    new_model = onnx.load(onnx_path)
    # for init in new_model.graph.initializer:
    #     data = np.frombuffer(init.raw_data, dtype="f")
    #     print(init.name, data)
    # onnx.checker.check_model(new_model)
    # onnx.helper.printable_graph(new_model.graph)
    # #onnx_path='/mnt/sda1/onnx/model_2d/resnet18_model_5.onnx'
    #
    prototxt_path = model_dir+model_name+'.prototxt'
    caffemodel_path = model_dir+model_name+'.caffemodel'
    # '''
    # prototxt_path = './model_2d/resnet18_bn_'+model_name+'.prototxt'
    # caffemodel_path = './model_2d/resnet18_bn_'+model_name+'.caffemodel'
    # prototxt_path = './model_2d/densenet_'+model_name+'.prototxt'
    # caffemodel_path = './model_2d/densenet_'+model_name+'.caffemodel'
    # '''
    graph = getGraph(onnx_path)
    # # #onnx.helper.printable_graph()
    # # #print('graph is:',graph)
    convertToCaffe(graph, prototxt_path, caffemodel_path)
    # view_caffe_weights(prototxt_path, caffemodel_path)

