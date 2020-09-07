import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph import Conv2D, Pool2D, BatchNorm, Linear, Conv3D, Dropout
from paddle.fluid.dygraph import Sequential
from paddle.fluid.layers import reshape, concat, pool3d

class Inception(fluid.dygraph.Layer):
    def __init__(self, input_num, conv11, conv12, conv13, conv21, conv22, conv32, conv41, mode=1, icep_3a=None, pool='max'):
        super(Inception, self).__init__()
        self.mode = mode
        self.icep = icep_3a
        self.conv1_1 = Conv2D(num_channels=input_num, num_filters=conv11, filter_size=1, padding=0)
        self.bn1_1 = BatchNorm(conv11, act='relu')
        self.conv1_2 = Conv2D(num_channels=conv11, num_filters=conv12, filter_size=3, padding=1)
        self.bn1_2 = BatchNorm(conv12, act='relu')
        self.conv1_3 = Conv2D(num_channels=conv12, num_filters=conv13, filter_size=3, padding=1, stride=self.mode)
        self.bn1_3 = BatchNorm(conv13, act='relu')
        self.conv2_1 = Conv2D(num_channels=input_num, num_filters=conv21, filter_size=1, padding=0)
        self.bn2_1 = BatchNorm(conv21, act='relu')
        self.conv2_2 = Conv2D(num_channels=conv21, num_filters=conv22, filter_size=3, padding=1, stride=self.mode)
        self.bn2_2 = BatchNorm(conv22, act='relu')
        if pool == 'max':
            self.pool = Pool2D(pool_size=3, pool_type='max', pool_stride=self.mode, pool_padding=1)
        elif pool == 'ave':
            self.pool = Pool2D(pool_size=3, pool_type='avg', pool_stride=self.mode, pool_padding=1)
        if self.mode == 2:
            pass
        else:
            self.conv3_2 = Conv2D(num_channels=input_num, num_filters=conv32, filter_size=1, padding=0)
            self.bn3_2 = BatchNorm(conv32, act='relu')
            self.conv4_1 = Conv2D(num_channels=input_num, num_filters=conv41, filter_size=1, padding=0)
            self.bn4_1 = BatchNorm(conv41, act='relu')

    def forward(self, input):
        conv1 = self.conv1_1(input)
        conv1 = self.bn1_1(conv1)
        conv1 = self.conv1_2(conv1)
        conv1_3D = self.bn1_2(conv1)
        conv1 = self.conv1_3(conv1_3D)
        conv1 = self.bn1_3(conv1)
        conv2 = self.conv2_1(input)
        conv2 = self.bn2_1(conv2)
        conv2 = self.conv2_2(conv2)
        conv2 = self.bn2_2(conv2)
        conv3 = self.pool(input)

        if self.mode == 2 and self.icep == None:
            out = concat([conv1, conv2, conv3], axis=1)
            return out
        elif self.mode == 2 and self.icep == True:
            out = concat([conv1, conv2, conv3], axis=1)
            return out, conv1_3D
        else:
            conv3 = self.conv3_2(conv3)
            conv3 = self.bn3_2(conv3)
            conv4 = self.conv4_1(input)
            conv4 = self.bn4_1(conv4)
            out = concat([conv1, conv2, conv3, conv4], axis=1)
            return out

class _3D_conv(fluid.dygraph.Layer):
    def __init__(self, input_num, out_num, dowm=1):
        super(_3D_conv, self).__init__()
        self.down_ = dowm
        self.input_num_ = input_num
        self.conv1 = Conv3D(num_channels=input_num, num_filters=out_num, filter_size=3, stride=dowm, padding=1)
        self.bn1 = BatchNorm(out_num, act='relu')
        self.conv2 = Conv3D(num_channels=out_num, num_filters=out_num, filter_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm(out_num, act='relu')
        if self.down_ == 2:
            self.conv1_1 = Conv3D(num_channels=input_num, num_filters=out_num, filter_size=3, stride=dowm, padding=1)
            self.bn1_1 = BatchNorm(out_num, act='relu')

    def forward(self, input):
        conv1 = self.conv1(input)
        conv1 = self.bn1(conv1)
        conv2 = self.conv2(conv1)
        if self.down_ == 2:
            conv2_1 = self.conv1_1(input)
            conv2 = self.bn1_1(conv2+conv2_1)
        else:
            conv2 = self.bn2(conv2+input)
        return conv2

class _3D_conv_3a(fluid.dygraph.Layer):
    def __init__(self, input_num, out_num):
        super(_3D_conv_3a, self).__init__()
        self.conv1 = Conv3D(num_channels=input_num, num_filters=out_num, filter_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm(out_num, act='relu')
    def forward(self, input):
        conv1 = self.conv1(input)
        conv1 = self.bn1(conv1)
        return conv1

def _3D_block(input_num, out_num, modules, dowm=1):
    layer = [_3D_conv(input_num=input_num, out_num=out_num, dowm=dowm)]
    layer += [_3D_conv(input_num=out_num, out_num=out_num) for _ in range(modules)]
    return Sequential(*layer)

class _2D_net(fluid.dygraph.Layer):
    def __init__(self):
        super(_2D_net, self).__init__()
        self.conv1 = Conv2D(num_channels=3, num_filters=64, filter_size=7, stride=2, padding=3)
        self.bn1 = BatchNorm(64, act='relu')
        self.pool1 = Pool2D(pool_size=3, pool_type='max', pool_stride=2, pool_padding=1)
        self.conv2_1 = Conv2D(num_channels=64, num_filters=64, filter_size=1, stride=1, padding=0)
        self.bn2_1 = BatchNorm(64, act='relu')
        self.conv2 = Conv2D(num_channels=64, num_filters=192, filter_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm(192, act='relu')
        self.pool2 = Pool2D(pool_size=3, pool_type='max', pool_stride=2, pool_padding=1)
        self._3a_1 = Inception(192, 64, 96, 96, 64, 64, 32, 64, pool='ave')
        self._3a_2 = Inception(256, 64, 96, 96, 64, 96, 64, 64, pool='ave')

    def forward(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self._3a_1(x)
        x = self._3a_2(x)
        out = x
        return out

class ECOModel(fluid.dygraph.Layer):
    def __init__(self):
        super(ECOModel, self).__init__()
        self._2D = _2D_net()
        self._2D_3a_up = Inception(320, conv11=64, conv12=96, conv13=96, conv21=128, conv22=160, conv32=None, conv41=None, mode=2, icep_3a=True, pool='max')
        self._3D_3a = _3D_conv_3a(96, 128)
        self._3D_3b = _3D_conv(128, 128)
        self._3D_4 = _3D_block(128, 256, 2, 2)
        self._3D_5 = _3D_block(256, 512, 2, 2)
        self._2Ds_4a = Inception(576, 96, 128, 128, 64, 96, 128, 224, pool='ave')  # 576
        self._2Ds_4b = Inception(576, 96, 128, 128, 96, 128, 128, 192, pool='ave')  # 576
        self._2Ds_4c = Inception(576, 128, 160, 160, 128, 160, 128, 160, pool='ave')  # 608
        self._2Ds_4d = Inception(608, 160, 192, 192, 128, 192, 128, 96,  pool='ave')  # 608
        self._2Ds_4e = Inception(608, 192, 256, 256, 128, 192, None, None, 2, pool='max')  # 1056
        self._2Ds_5a = Inception(1056, 160, 224, 224, 192, 320, 128, 352, pool='ave')
        self._2Ds_5b = Inception(1024, 192, 224, 224, 192, 320, 128, 352, pool='max')
        self.ave_pooling1 = Pool2D(pool_size=7, pool_type='avg', pool_stride=1, global_pooling=True)
        self.ave_pooling2 = Pool2D(pool_size=[1, 24], pool_type='avg', pool_stride=1, global_pooling=True)
        self.drop1 = Dropout(0.6)
        self.drop2 = Dropout(0.5)
        self.lin = Linear(input_dim=1536, output_dim=101, act='softmax')

    def forward(self, image, label):
        x = reshape(image, shape=[-1, 3, 224, 224])
        # 共享特征提取部分
        x = self._2D(x)
        x_0, x_1 = self._2D_3a_up(x)
        # 2D融合部分
        x2 = self._2Ds_4a(x_0)
        x2 = self._2Ds_4b(x2)
        x2 = self._2Ds_4c(x2)
        x2 = self._2Ds_4d(x2)
        x2 = self._2Ds_4e(x2)
        x2 = self._2Ds_5a(x2)
        x2 = self._2Ds_5b(x2)
        x2 = self.ave_pooling1(x2)
        x2 = self.drop1(x2)
        x_sum = reshape(x2, shape=[-1, 1024, 1, 24])
        x_sum = self.ave_pooling2(x_sum)
        x_sum = reshape(x_sum, shape=[-1, 1024])

        # 3D融合部分
        x1 = reshape(x_1, shape=[-1, 96, 28, 28, 24])
        x1 = self._3D_3a(x1)
        x1 = self._3D_3b(x1)
        x1 = self._3D_4(x1)
        x1 = self._3D_5(x1)
        x1 = pool3d(x1, pool_type='avg', global_pooling=True)
        x1 = reshape(x1, shape=[-1, 512])
        x1 = self.drop2(x1)

        # 2D,3D融合
        x_sum = concat([x_sum, x1], axis=1)
        x_sum = self.lin(x_sum)
        if label is not None:
            acc = fluid.layers.accuracy(input=x_sum, label=label)
            return x_sum, acc
        else:
            return x_sum


if __name__ == '__main__':
    with fluid.dygraph.guard():
        network = ECOModel()
        img = np.zeros([10, 24, 3, 224, 224]).astype('float32')
        img = fluid.dygraph.to_variable(img)
        outs = network(img, None).numpy()
        #print('运行成功，没有语法错误')
        #print(outs)