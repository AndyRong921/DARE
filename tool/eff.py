"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
"""

# 作者信息
# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

import torch
from torch import nn
from torch.nn import functional as F
from tool.utils import (
    round_filters,  # 调整滤波器数量
    round_repeats,  # 调整模块重复次数
    drop_connect,  # 随机丢弃连接
    get_same_padding_conv2d,  # 获取相同padding的卷积
    get_model_params,  # 获取模型参数
    efficientnet_params,  # 模型配置参数
    load_pretrained_weights,  # 加载预训练权重
    Swish,  # 标准Swish激活函数
    MemoryEfficientSwish,  # 内存优化的Swish
    calculate_output_image_size  # 计算输出图像尺寸
)

# 支持的模型名称列表
VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',

    # 支持构建但不提供预训练权重
    'efficientnet-l2'
)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.
    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].
    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        # 保存块参数和全局参数
        self._block_args = block_args
        # 批归一化参数设置
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch与tensorflow的差异
        self._bn_eps = global_params.batch_norm_epsilon
        # 是否使用SE注意力机制
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        # 是否使用跳跃连接
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # 倒置瓶颈
        inp = self._block_args.input_filters  # 输入通道数
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # 输出通道数
        if self._block_args.expand_ratio != 1:
            # 获取相同padding的卷积
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            # 1x1卷积扩展通道
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            # 归一化
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- 这里不会修改image_size

        # 深度可分离卷积阶段
        k = self._block_args.kernel_size  # 卷积核大小
        s = self._block_args.stride  # 步长
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        # 深度可分离卷积(groups=oup实现深度卷积)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        # 归一化
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        # 计算图像尺寸
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            # 计算压缩后的通道数
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            # SE模块的两个1x1卷积
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # 逐点卷积
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        # 1x1卷积压缩
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        # 归一化
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        # 使用Swish激活函数
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock的前向传播函数"""
        # 保存输入用于跳跃连接
        x = inputs
        
        # 倒置瓶颈
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        # 深度可分离卷积
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation注意力机制
        if self.has_se:
            # 全局平均池化
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            # 压缩通道
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            # 扩展通道
            x_squeezed = self._se_expand(x_squeezed)
            # 使用sigmoid作为注意力权重
            x = torch.sigmoid(x_squeezed) * x

        # 逐点卷积
        x = self._project_conv(x)
        x = self._bn2(x)

        # 跳跃连接和drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        # 满足三个条件才使用跳跃连接:
        # 1. 启用了id_skip
        # 2. 步长为1(
        # 3. 输入输出通道数相同
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # drop connect随机丢弃连接(类似于dropout)
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            # 跳跃连接
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """设置Swish激活函数的版本"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """EfficientNet模型.
       可以通过.from_name或.from_pretrained方法方便地创建
    Args:
        blocks_args (list[namedtuple]): 构造块的BlockArgs列表
        global_params (namedtuple): 块之间共享的GlobalParams
    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)
    Example:
        >>> import torch
        >>> from efficientnet.model import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        # 检查blocks_args是否为非空列表
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        # 保存参数
        self._global_params = global_params
        self._blocks_args = blocks_args

        # 归一化
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # 获取合适的卷积层
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem部分(第一个卷积层)
        in_channels = 3  # rgb输入
        out_channels = round_filters(32, self._global_params)  # 调整输出通道数
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        # 计算输出图像尺寸
        image_size = calculate_output_image_size(image_size, 2)

        # 构建块
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # 调整块的输入输出滤波器
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # 第一个块需要处理步长和滤波器尺寸变化
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1: # 修改block_args保持相同输出尺寸
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # Head部分
        in_channels = block_args.output_filters  # 最后一个块的输出通道数
        out_channels = round_filters(1280, self._global_params)  # 固定为1280
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self.out_channels=out_channels
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        
        # 最后的全连接层
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self._dropout = nn.Dropout(self._global_params.dropout_rate)  # 随机丢弃
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)  # 分类层
        self._swish = MemoryEfficientSwish()  # 激活函数

    def set_swish(self, memory_efficient=True):
        """设置Swish激活函数的版本
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        # 设置所有块的swish函数
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """使用卷积层提取不同尺度的特征
        """
        endpoints = dict()

        # Stem部分
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # 遍历所有块
        for idx, block in enumerate(self._blocks):
            # 计算drop connect率
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # 缩放drop connect率
            x = block(x, drop_connect_rate=drop_connect_rate)
            # 当特征图尺寸变化时保存特征
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head部分
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        # 对每个尺度的特征进行全局平均池化
        f1 = self._avg_pooling(endpoints['reduction_1'])
        f1=f1.flatten(start_dim=1)
        f2 = self._avg_pooling(endpoints['reduction_2'])
        f2=f2.flatten(start_dim=1)
        f3 = self._avg_pooling(endpoints['reduction_3'])
        f3=f3.flatten(start_dim=1)
        f4 = self._avg_pooling(endpoints['reduction_4'])
        f4=f4.flatten(start_dim=1)
        f5 = self._avg_pooling(endpoints['reduction_5'])
        f5=f5.flatten(start_dim=1)

        # 拼接多尺度特征
        feature = f5
        feature = torch.cat((feature,f4),1)
        feature = torch.cat((feature,f3),1)
        feature = torch.cat((feature,f2),1)
        
        return feature.data

    def extract_features(self, inputs):
        """使用卷积层提取特征
        """
        # Stem部分
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        fs = []
        # 遍历所有块
        for idx, block in enumerate(self._blocks):
            # 计算drop connect率
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # 缩放drop connect率
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head部分
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """EfficientNet的前向传播函数"""
        # 提取特征
        x = self.extract_features(inputs)
        # 提取多尺度特征
        infeature = self.extract_endpoints(inputs)
        # 池化和全连接层
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            #x = self._dropout(x)
            x = self._fc(x)
        return infeature, x

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """根据名称创建EfficientNet模型
        
            model_name (str): EfficientNet模型名称
            in_channels (int): 输入数据的通道数
            override_params: 覆盖模型全局参数的其他关键字参数
       
        """
        # 检查模型名称是否有效
        cls._check_model_name_is_valid(model_name)
        # 获取模型参数
        blocks_args, global_params = get_model_params(model_name, override_params)
        # 创建模型
        model = cls(blocks_args, global_params)
        # 调整输入通道数
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False,
                        in_channels=3, num_classes=1000, **override_params):
        """从预训练权重创建EfficientNet模型
       
            model_name (str): EfficientNet模型名称
            weights_path (None or str): 预训练权重路径
            advprop (bool): 是否使用advprop训练的权重
            in_channels (int): 输入数据的通道数
            num_classes (int): 分类类别数
            override_params: 覆盖模型全局参数的其他关键字参数
    
        """
        # 创建模型
        model = cls.from_name(model_name, num_classes=num_classes, **override_params)
        # 加载预训练权重
        load_pretrained_weights(model, model_name, weights_path=weights_path, 
                              load_fc=(num_classes == 1000), advprop=advprop)
        # 调整输入通道数
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """获取给定EfficientNet模型的输入图像尺寸
        """
        # 检查模型名称是否有效
        cls._check_model_name_is_valid(model_name)
        # 获取模型参数
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """验证模型名称是否有效
        """
        # 检查模型名称是否在支持的列表中
        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))

    def _change_in_channels(self, in_channels):
        """调整模型的第一个卷积层的输入通道数
        """
        # 如果输入通道数不是3(RGB)，则调整第一个卷积层
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)