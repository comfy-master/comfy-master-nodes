import os
import sys
from . import custom_nodes

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dependencies"))

WEB_DIRECTORY = "web"
NODE_CLASS_MAPPINGS = {
    "ServiceConfigNode": custom_nodes.ServiceConfigNode,
    "CMaster_InputImage": custom_nodes.InputImageNode,
    "CMaster_InputString": custom_nodes.InputStringNode,
    "CMaster_InputEnumString": custom_nodes.InputEnumStringNode,
    "CMaster_InputInt": custom_nodes.InputIntNode,
    "CMaster_InputRangeInt": custom_nodes.InputRangeIntNode,
    "CMaster_InputBoolean": custom_nodes.InputBooleanNode,
    "CMaster_InputFloat": custom_nodes.InputFloatNode,
    "CMaster_InputRangeFloat": custom_nodes.InputRangeFloatNode,
    "CMaster_InputCheckpoint": custom_nodes.InputCheckpointNode,
    "CMaster_OutputImage": custom_nodes.OutputImageNode,

    # "CMaster_OutputString": custom_nodes.OutputStringNode,
    # "CMaster_OutputAudio": custom_nodes.OutputAudioNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ServiceConfigNode": "服务配置节点",
    "CMaster_InputCheckpoint": "模型输入",
    "CMaster_InputImage": "图片输入",
    "CMaster_InputString": "字符串输入",
    "CMaster_InputEnumString": "枚举输入",
    "CMaster_InputInt": "整数输入",
    "CMaster_InputRangeInt": "范围整数输入",
    "CMaster_InputBoolean": "布尔值输入",
    "CMaster_InputFloat": "浮点数输入",
    "CMaster_InputRangeFloat": "范围浮点数输入",
    "CMaster_OutputImage": "图片输出",
    # "CMaster_OutputString": "Output String",
    # "CMaster_OutputAudio": "Output Audio",
}

all = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
