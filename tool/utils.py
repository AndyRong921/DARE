"""utils.py - Helper functions for building the model and for loading model parameters.
   These helper functions are built to mirror those in the official TensorFlow implementation.
"""

# 作者: lukemelas (github用户名)
# Github仓库: https://github.com/lukemelas/EfficientNet-PyTorch  
# 由workingcoder (github用户名)进行调整和添加注释

import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo


################################################################################
### 模型架构的帮助函数
################################################################################

# GlobalParams和BlockArgs：两个命名元组
# Swish和MemoryEfficientSwish：两种Swish激活函数的实现
# round_filters和round_repeats：
#     用于计算模型宽度和深度缩放参数的函数
# get_width_and_height_from_size和calculate_output_image_size
# drop_connect：一种结构设计
# get_same_padding_conv2d：
#     Conv2dDynamicSamePadding
#     Conv2dStaticSamePadding
# get_same_padding_maxPool2d：
#     MaxPool2dDynamicSamePadding
#     MaxPool2dStaticSamePadding
#     这是一个额外的函数，不在EfficientNet中使用，
#     但可以在其他模型（如EfficientDet）中使用。

# 整个模型的参数（stem、所有block和head）
GlobalParams = collections.namedtuple('GlobalParams', [
    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
    'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
    'drop_connect_rate', 'depth_divisor', 'min_depth', 'include_top'])

# 单个模型block的参数
BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

# 设置GlobalParams和BlockArgs的默认值
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


# Swish函数的普通实现
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# Swish函数的内存高效实现
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)  # 保存输入以供反向传播使用
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]  # 获取保存的输入
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)  # 应用内存高效的Swish实现


def round_filters(filters, global_params):
    """根据宽度乘数计算并四舍五入滤波器数量。
       使用global_params中的width_coefficient、depth_divisor和min_depth。
        filters (int): 需要计算的滤波器数量。
        global_params (namedtuple): 模型的全局参数。
        new_filters: 计算后的新滤波器数量。
    """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    # TODO: 修改参数名称。
    #       可能名称(width_divisor,min_width)
    #       比(depth_divisor,min_depth)更合适。
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor # 当使用min_depth时要注意这一行
    # 遵循官方TensorFlow实现中的公式
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters: # 防止四舍五入超过10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """根据深度乘数计算block的重复次数。
       使用global_params中的depth_coefficient。
        repeats (int): 需要计算的num_repeat。
        global_params (namedtuple): 模型的全局参数。
        new repeat: 计算后的新重复次数。
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    # 遵循官方TensorFlow实现中的公式
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """Drop connect操作。
        input (tensor: BCWH): 该结构的输入。
        p (float: 0.0~1.0): drop connection的概率。
        training (bool): 运行模式。
        output: drop connection后的输出。
    """
    assert 0 <= p <= 1, 'p必须在[0,1]范围内'

    if not training:  # 如果不是训练模式，直接返回输入
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # 根据概率生成二进制张量掩码（p概率为0，1-p概率为1）
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor  # 缩放输出以保持期望值
    return output


def get_width_and_height_from_size(x):
    """从x中获取高度和宽度。
        x (int, tuple or list): 数据尺寸
        size: 一个元组或列表(H,W)
    """
    if isinstance(x, int):  # 如果是整数，返回相同的高度和宽度
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):  # 如果是列表或元组，直接返回
        return x
    else:
        raise TypeError()  # 其他类型抛出类型错误


def calculate_output_image_size(input_image_size, stride):
    """当使用带stride的Conv2dSamePadding时计算输出图像尺寸。
       静态填充需要。感谢mannatsingh指出这一点。
        input_image_size (int, tuple or list): 输入图像尺寸
        stride (int, tuple or list): Conv2d操作的stride
        output_image_size: 一个列表[H,W]
    """
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))  # 计算高度
    image_width = int(math.ceil(image_width / stride))  # 计算宽度
    return [image_height, image_width]


# 注意：
# 以下'SamePadding'函数使输出尺寸等于ceil(输入尺寸/stride)。
# 只有当stride等于1时，输出尺寸才能与输入尺寸相同。
# 不要被它们的函数名混淆！

def get_same_padding_conv2d(image_size=None):
    """如果指定了图像尺寸，则选择静态填充，否则选择动态填充。
       静态填充对于模型的ONNX导出是必要的。
        image_size (int or tuple): 图像尺寸
        Conv2dDynamicSamePadding或Conv2dStaticSamePadding
    """
    if image_size is None:  # 如果没有指定图像尺寸，使用动态填充
        return Conv2dDynamicSamePadding
    else:  # 如果指定了图像尺寸，使用静态填充
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """类似TensorFlow的2D卷积，适用于动态图像尺寸。
       填充在forward函数中动态计算。
    """

    # 'SAME'模式填充的提示。
    #     给定以下参数：
    #         i: 宽度或高度
    #         s: stride
    #         k: 核尺寸
    #         d: dilation
    #         p: padding
    #     Conv2d后的输出：
    #         o = floor((i+p-((k-1)*d+1))/s+1)
    # 如果o等于i，i = floor((i+p-((k-1)*d+1))/s+1),
    # => p = (i-1)*s+((k-1)*d+1)-i

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2  # 统一stride格式

    def forward(self, x):
        ih, iw = x.size()[-2:]  # 输入的高度和宽度
        kh, kw = self.weight.size()[-2:]  # 卷积核的高度和宽度
        sh, sw = self.stride  # stride的高度和宽度
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw) # 根据stride改变输出尺寸！
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)  # 计算高度方向填充
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)  # 计算宽度方向填充
        if pad_h > 0 or pad_w > 0:
            # 应用填充
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """类似TensorFlow 'SAME'模式的2D卷积，给定输入图像尺寸。
       填充模块在构造函数中计算，然后在forward中使用。
    """

    # 与Conv2dDynamicSamePadding相同的计算

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2  # 统一stride格式

        # 根据图像尺寸计算填充并保存
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]  # 卷积核尺寸
        sh, sw = self.stride  # stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # 输出尺寸
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)  # 高度填充
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)  # 宽度填充
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()  # 无填充时使用恒等映射

    def forward(self, x):
        x = self.static_padding(x)  # 应用静态填充
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


def get_same_padding_maxPool2d(image_size=None):
    """如果指定了图像尺寸，则选择静态填充，否则选择动态填充。
       静态填充对于模型的ONNX导出是必要的。
        image_size (int or tuple): 图像尺寸
        MaxPool2dDynamicSamePadding或MaxPool2dStaticSamePadding
    """
    if image_size is None:  # 未指定图像尺寸使用动态填充
        return MaxPool2dDynamicSamePadding
    else:  # 指定图像尺寸使用静态填充
        return partial(MaxPool2dStaticSamePadding, image_size=image_size)


class MaxPool2dDynamicSamePadding(nn.MaxPool2d):
    """类似TensorFlow 'SAME'模式的2D最大池化，适用于动态图像尺寸。
       填充在forward函数中动态计算。
    """

    def __init__(self, kernel_size, stride, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride  # 统一stride格式
        self.kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size  # 统一核尺寸格式
        self.dilation = [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation  # 统一dilation格式

    def forward(self, x):
        ih, iw = x.size()[-2:]  # 输入尺寸
        kh, kw = self.kernel_size  # 池化核尺寸
        sh, sw = self.stride  # stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # 输出尺寸
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)  # 高度填充
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)  # 宽度填充
        if pad_h > 0 or pad_w > 0:
            # 应用填充
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding,
                            self.dilation, self.ceil_mode, self.return_indices)

class MaxPool2dStaticSamePadding(nn.MaxPool2d):
    """类似TensorFlow 'SAME'模式的2D最大池化，给定输入图像尺寸。
       填充模块在构造函数中计算，然后在forward中使用。
    """

    def __init__(self, kernel_size, stride, image_size=None, **kwargs):
        super().__init__(kernel_size, stride, **kwargs)
        self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride  # 统一stride格式
        self.kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size  # 统一核尺寸格式
        self.dilation = [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation  # 统一dilation格式

        # 根据图像尺寸计算填充并保存
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.kernel_size  # 池化核尺寸
        sh, sw = self.stride  # stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # 输出尺寸
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)  # 高度填充
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)  # 宽度填充
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()  # 无填充时使用恒等映射

    def forward(self, x):
        x = self.static_padding(x)  # 应用静态填充
        x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding,
                         self.dilation, self.ceil_mode, self.return_indices)
        return x


################################################################################
### 加载模型参数的帮助函数
################################################################################

# BlockDecoder: 用于编码和解码BlockArgs的类
# efficientnet_params: 查询复合系数的函数
# get_model_params和efficientnet:
#     获取EfficientNet的BlockArgs和GlobalParams的函数
# url_map和url_map_advprop: 预训练权重的url映射字典
# load_pretrained_weights: 加载预训练权重的函数

class BlockDecoder(object):
    """Block解码器，为了可读性，
       直接来自官方TensorFlow仓库。
    """

    @staticmethod
    def _decode_block_string(block_string):
        """通过参数字符串表示获取block。
            block_string (str): 参数字符串表示
            BlockArgs: 在本文件顶部定义的命名元组
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # 检查stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            num_repeat=int(options['r']),  # 重复次数
            kernel_size=int(options['k']),  # 卷积核尺寸
            stride=[int(options['s'][0])],  # stride
            expand_ratio=int(options['e']),  # 扩展比率
            input_filters=int(options['i']),  # 输入滤波器数量
            output_filters=int(options['o']),  # 输出滤波器数量
            se_ratio=float(options['se']) if 'se' in options else None,  # SE比率
            id_skip=('noskip' not in block_string))  # 是否跳过连接

    @staticmethod
    def _encode_block_string(block):
        """将block编码为字符串。
            block (namedtuple): BlockArgs类型的参数
            block_string: BlockArgs的字符串形式
        """
        args = [
            'r%d' % block.num_repeat,  # 重复次数
            'k%d' % block.kernel_size,  # 卷积核尺寸
            's%d%d' % (block.strides[0], block.strides[1]),  # stride
            'e%s' % block.expand_ratio,  # 扩展比率
            'i%d' % block.input_filters,  # 输入滤波器
            'o%d' % block.output_filters  # 输出滤波器
        ]
        if 0 < block.se_ratio <= 1:  # 如果有SE比率且合理
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:  # 如果不跳过连接
            args.append('noskip')
        return '_'.join(args)  # 用下划线连接所有参数

    @staticmethod
    def decode(string_list):
        """解码字符串列表以指定网络中的blocks。
            string_list (list[str]): 字符串列表，每个字符串是一个block的表示
            blocks_args: block参数的BlockArgs命名元组列表
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:  # 解码每个block字符串
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """将BlockArgs列表编码为字符串列表。
            blocks_args (list[namedtuples]): block参数的BlockArgs命名元组列表
            block_strings: 字符串列表，每个字符串是一个block的表示
        """
        block_strings = []
        for block in blocks_args:  # 编码每个block
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet_params(model_name):
    """将EfficientNet模型名称映射到参数系数。
        model_name (str): 要查询的模型名称
        params_dict[model_name]: (width,depth,res,dropout)元组
    """
    params_dict = {
        # 系数: width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),  # 基础模型
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),  # 最大模型
    }
    return params_dict[model_name]  # 返回对应模型的参数


def efficientnet(width_coefficient=None, depth_coefficient=None, image_size=None,
                 dropout_rate=0.2, drop_connect_rate=0.2, num_classes=1000, include_top=True):
    """为efficientnet模型创建BlockArgs和GlobalParams。
        width_coefficient (float): 宽度系数
        depth_coefficient (float): 深度系数
        image_size (int): 图像尺寸
        dropout_rate (float): dropout率
        drop_connect_rate (float): drop connect率
        num_classes (int): 类别数
        include_top (bool): 是否包含顶层
        blocks_args, global_params。
    """

    # 整个模型的blocks参数（默认为efficientnet-b0）
    # 在EfficientNet类的构造中会根据模型修改
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25',  # 第一个block
        'r2_k3_s22_e6_i16_o24_se0.25',  # 第二个block
        'r2_k5_s22_e6_i24_o40_se0.25',  # 第三个block
        'r3_k3_s22_e6_i40_o80_se0.25',  # 第四个block
        'r3_k5_s11_e6_i80_o112_se0.25', # 第五个block
        'r4_k5_s22_e6_i112_o192_se0.25', # 第六个block
        'r1_k3_s11_e6_i192_o320_se0.25', # 第七个block
    ]
    blocks_args = BlockDecoder.decode(blocks_args)  # 解码block参数

    global_params = GlobalParams(
        width_coefficient=width_coefficient,  # 宽度系数
        depth_coefficient=depth_coefficient,  # 深度系数
        image_size=image_size,  # 图像尺寸
        dropout_rate=dropout_rate,  # dropout率

        num_classes=num_classes,  # 类别数
        batch_norm_momentum=0.99,  # BN动量
        batch_norm_epsilon=1e-3,  # BN epsilon
        drop_connect_rate=drop_connect_rate,  # drop connect率
        depth_divisor=8,  # 深度除数
        min_depth=None,  # 最小深度
        include_top=include_top,  # 是否包含顶层
    )

    return blocks_args, global_params  # 返回block参数和全局参数


def get_model_params(model_name, override_params):
    """获取给定模型名称的block参数和全局参数。
        model_name (str): 模型名称
        override_params (dict): 用于修改global_params的字典
        blocks_args, global_params
    """
    if model_name.startswith('efficientnet'):  # 如果是EfficientNet模型
        w, d, s, p = efficientnet_params(model_name)  # 获取模型参数
        # 注意：所有模型的drop connect率=0.2
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: {}'.format(model_name))
    if override_params:
        # 如果override_params包含global_params中没有的字段，将引发ValueError
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


# 使用标准方法训练
# 更多细节见论文(EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks)
url_map = {
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth  ',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth  ',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth  ',
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth  ',
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth  ',
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth  ',
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth  ',
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth  ',
}

# 使用对抗样本训练(AdvProp)
# 更多细节见论文(Adversarial Examples Improve Image Recognition)
url_map_advprop = {
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth  ',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth  ',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth  ',
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth  ',
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth  ',
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth  ',
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth  ',
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth  ',
    'efficientnet-b8': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth  ',
}

# TODO: 添加'efficientnet-l2'的预训练权重url映射


def load_pretrained_weights(model, model_name, weights_path=None, load_fc=True, advprop=False):
    """从权重路径加载预训练权重或使用url下载。
        model (Module): efficientnet的整个模型
        model_name (str): efficientnet的模型名称
        weights_path (None or str):
            str: 本地磁盘上预训练权重文件的路径
            None: 使用从互联网下载的预训练权重
        load_fc (bool): 是否加载模型末尾fc层的预训练权重
        advprop (bool): 是否加载使用advprop训练的预训练权重
                        (当weights_path为None时有效)
    """
    if isinstance(weights_path, str):  # 如果提供了权重路径
        state_dict = torch.load(weights_path)
    else:
        # AutoAugment或Advprop（不同的预处理）
        url_map_ = url_map_advprop if advprop else url_map  # 选择url映射
        state_dict = model_zoo.load_url(url_map_[model_name])  # 下载预训练权重

    if load_fc:  # 如果加载全连接层
        ret = model.load_state_dict(state_dict, strict=False)
        assert not ret.missing_keys, '加载预训练权重时缺少的键: {}'.format(ret.missing_keys)
    else:  # 如果不加载全连接层
        state_dict.pop('_fc.weight')  # 移除fc层权重
        state_dict.pop('_fc.bias')    # 移除fc层偏置
        ret = model.load_state_dict(state_dict, strict=False)
        assert set(ret.missing_keys) == set(
            ['_fc.weight', '_fc.bias']), '加载预训练权重时缺少的键: {}'.format(ret.missing_keys)
    assert not ret.unexpected_keys, '加载预训练权重时意外的键: {}'.format(ret.unexpected_keys)

    print('为{}加载了预训练权重'.format(model_name))