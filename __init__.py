import os
import sys
from . import custom_nodes

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dependencies"))

WEB_DIRECTORY = "web"
NODE_CLASS_MAPPINGS = {
    "ServiceConfigNode": custom_nodes.ServiceConfigNode,
    "CM_InputImage": custom_nodes.InputImageNode,
    "CM_InputString": custom_nodes.InputStringNode,
    "CM_InputEnumString": custom_nodes.InputEnumStringNode,
    "CM_InputInt": custom_nodes.InputIntNode,
    "CM_InputRangeInt": custom_nodes.InputRangeIntNode,
    "CM_InputBoolean": custom_nodes.InputBooleanNode,
    "CM_InputFloat": custom_nodes.InputFloatNode,
    "CM_InputRangeFloat": custom_nodes.InputRangeFloatNode,
    "CM_InputCheckpoint": custom_nodes.InputCheckpointNode,
    "CM_OutputImage": custom_nodes.OutputImageNode,

    # "CM_OutputString": custom_nodes.OutputStringNode,
    # "CM_OutputAudio": custom_nodes.OutputAudioNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ServiceConfigNode": "服务配置节点",
    "CM_InputCheckpoint": "模型输入",
    "CM_InputImage": "图片输入",
    "CM_InputString": "字符串输入",
    "CM_InputEnumString": "枚举输入",
    "CM_InputInt": "整数输入",
    "CM_InputRangeInt": "范围整数输入",
    "CM_InputBoolean": "布尔值输入",
    "CM_InputFloat": "浮点数输入",
    "CM_InputRangeFloat": "范围浮点数输入",
    "CM_OutputImage": "图片输出",
    # "CM_OutputString": "Output String",
    # "CM_OutputAudio": "Output Audio",
}

all = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
