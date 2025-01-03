import sys

from PIL.ImageFile import ImageFile
from server import PromptServer
import folder_paths
from PIL import Image, ImageOps, ImageSequence
import base64
from io import BytesIO
import torch
import numpy as np
from torch import Tensor

from .encoding import encode_string, encode_image, encode_audio
import os
import torchaudio

import comfy.sd
import comfy.utils
import node_helpers

var_prefix_name = "ComfyMasterVar_"

class ServiceConfigNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": ("STRING", {"multiline": False, "default": "服务名称"}),
            },
            "optional": {
                "description": ("STRING", {"multiline": True, "default": ""}),
                "allowLocalRepair": ("BOOLEAN", {"default": False, "label_on": "属于局部修复工作流"}),
                "allowPreload": ("BOOLEAN", {"default": False, "label_on": "预先加载"}),
                "allowSingleDeploy": ("BOOLEAN", {"default": False, "label_on": "单独部署"}),
                "cpu": ("BOOLEAN", {"default": False, "label_on": "CPU部署"}),
            }
        }


    RETURN_TYPES = ()
    CATEGORY = "comfyui-master"
    FUNCTION = "output_func"

    def output_func(self, name, description = "", allowLocalRepair = False, allowPreload = False, allowSingleDeploy = False, cpu=False):
        return ()

class ImageWorkflowMetadataNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "hidden": {
                "document_width": ("INT", {"default": 0}),
                "document_height": ("INT", {"default": 0}),
                "has_selection": ("BOOLEAN", {"default": False}),
                "selection_x": ("INT", {"default": 0}),
                "selection_y": ("INT", {"default": 0}),
                "selection_width": ("INT", {"default": 0}),
                "selection_height": ("INT", {"default": 0}),
            },
        }


    RETURN_TYPES = ("INT", "INT", "BOOLEAN", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("文档宽度", "文档高度", "是否存在选区", "选区位置X", "选区位置Y", "选区宽度", "选区高度")
    CATEGORY = "comfyui-master/工具"
    FUNCTION = "output_func"

    def output_func(self, document_width = 0, document_height = 0, has_selection=False,
                    selection_x=0, selection_y=0,
                    selection_width = 0, selection_height = 0):
        return (document_width, document_height, has_selection, selection_x, selection_y, selection_width, selection_height)


class ImageWorkflowMetadataTestNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "document_width": ("INT", {"default": 0}),
                "document_height": ("INT", {"default": 0}),
                "has_selection": ("BOOLEAN", {"default": False}),
                "selection_x": ("INT", {"default": 0}),
                "selection_y": ("INT", {"default": 0}),
                "selection_width": ("INT", {"default": 0}),
                "selection_height": ("INT", {"default": 0}),
            },
        }


    RETURN_TYPES = ("INT", "INT", "BOOLEAN", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("文档宽度", "文档高度", "是否存在选区", "选区位置X", "选区位置Y", "选区宽度", "选区高度")
    CATEGORY = "comfyui-master/调试"
    FUNCTION = "output_func"

    def output_func(self, document_width = 0, document_height = 0, has_selection=False,
                    selection_x=0, selection_y=0,
                    selection_width = 0, selection_height = 0):
        return (document_width, document_height, has_selection, selection_x, selection_y, selection_width, selection_height)



class InputCheckpointNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "var_name": ("STRING", {"multiline": False, "default": "InputCheckpoint"}),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "要加载的检查点（模型）的名称。"}),
                "export": ("BOOLEAN", {"default": True}),
                "checkpoints": ("STRING", {"multiline": True, "default": '\n'.join(folder_paths.get_filename_list("checkpoints"))}),
            },
            "optional": {
                "description": ("STRING", {"multiline": False, "default": ""}),
                "order": ("INT", {"default": 0, "min": 0, "max": 0xffffff, "step": 1}),
                "default_generate_algorithm": (["固定值", "随机值"], {"tooltip": "默认生成算法", "default": "固定值"})
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")

    CATEGORY = "comfyui-master/输入"
    FUNCTION = "input_checkpoint"

    def input_checkpoint(self, var_name, ckpt_name, checkpoints, export, description = "", order = 0, default_generate_algorithm= "固定值"):
        try:
            # Split enums by comma or newline, and strip whitespace
            checkpoints = [enum.strip() for enum in checkpoints.replace('\n', ',').split(',') if enum.strip()]
            if ckpt_name not in checkpoints:
                checkpoints.append(ckpt_name)

            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True,
                                                        embedding_directory=folder_paths.get_folder_paths("embeddings"))
            return out[:3]
        except Exception as e:
            print(f"raised exception: {var_name}")
            raise e


class InputLoraNode:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "var_name": ("STRING", {"multiline": False, "default": "InputLora"}),
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "加载LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                                             "tooltip": "如何强烈地修改扩散模型。该值可以是负的。"}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                                            "tooltip": "CLIP模型修改力度有多大。该值可以是负的。"}),
                "export": ("BOOLEAN", {"default": True}),
                "loras": ("STRING", {"multiline": True, "default": '\n'.join(folder_paths.get_filename_list("loras"))}),
            },
            "optional": {
                "description": ("STRING", {"multiline": False, "default": ""}),
                "order": ("INT", {"default": 0, "min": 0, "max": 0xffffff, "step": 1}),
                "default_generate_algorithm": (["固定值", "随机值"], {"tooltip": "默认生成算法", "default": "固定值"})
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"

    CATEGORY = "comfyui-master/输入"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."

    def load_lora(self, var_name, model, clip, lora_name, strength_model, strength_clip, export, loras, description = "", order = 0,
                  default_generate_algorithm = "固定值"):
        try:
            if strength_model == 0 and strength_clip == 0:
                return (model, clip)

            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            lora = None
            if self.loaded_lora is not None:
                if self.loaded_lora[0] == lora_path:
                    lora = self.loaded_lora[1]
                else:
                    self.loaded_lora = None

            if lora is None:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_lora = (lora_path, lora)

            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
            return (model_lora, clip_lora)
        except Exception as e:
            print(f"raised exception: {var_name}")
            raise e


class LoadImageToBase64:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "comfyui-master/调试"

    RETURN_TYPES = ("STRING", )
    FUNCTION = "load_image"

    def load_image(self, image):
        try:
            image_path = folder_paths.get_annotated_filepath(image)

            img: ImageFile = node_helpers.pillow(Image.open, image_path)
            image_data = BytesIO()
            img.save(image_data, format="PNG")
            image_data_bytes = image_data.getvalue()

            return (base64.b64encode(image_data_bytes).decode('utf-8'),)
        except Exception as e:
            print(f"raised exception: LoadImageToBase64")
            raise e

class InputImageNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "var_name": ("STRING", {"multiline": False, "default": "InputImage"}),
                "image": ("STRING", {"multiline": False}),
                "export": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "description": ("STRING", {"multiline": False, "default": ""}),
                "order": ("INT", {"default": 0, "min": 0, "max": 0xffffff, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    CATEGORY = "comfyui-master/输入"
    FUNCTION = "input_image"

    def input_image(self, var_name, image, export, description = "", order = 0):
        try:
            imgdata = base64.b64decode(image)
            img = Image.open(BytesIO(imgdata))

            if "A" in img.getbands():
                mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            img = img.convert("RGB")
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]

            return (img, mask)
        except Exception as e:
            print(f"Exception raised: {var_name}")
            raise e

class ImageInfoNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("宽度", "高度", "通道")
    CATEGORY = "comfyui-master/工具"
    FUNCTION = "input_image"

    def input_image(self, image: Tensor):
        return image.shape[1:]


## 生成遮罩图层
class GenerateImageMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "options": (["固定值", "百分比"], {"default": "固定值"}),
                "inner_edge_width": ("INT", {"default": 5, "tooltip": "固定边缘宽度"}),
                "inner_edge_height": ("INT", {"default": 5, "tooltip": "固定边缘高度" }),
                "inner_edge_width_percent": ("FLOAT", {"default": 0.18, "min": 0, "max": 0.5, "step": 0.01, "tooltip": "固定边缘宽度百分比"}),
                "inner_edge_height_percent": ("FLOAT", {"default": 0.18, "min": 0, "max": 0.5, "step": 0.01, "tooltip": "固定边缘高度百分比"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    CATEGORY = "comfyui-master/工具"
    FUNCTION = "input_image"

    def input_image(self, image: Tensor, options: str, inner_edge_width: int, inner_edge_height: int, inner_edge_width_percent: float,
                    inner_edge_height_percent: float):
        if options == "百分比":
            inner_edge_width = int(inner_edge_width_percent * image.shape[2])
            inner_edge_height = int(inner_edge_height_percent * image.shape[1])
        inner_width = image.shape[2] - inner_edge_width * 2
        inner_height = image.shape[1] - inner_edge_height * 2
        inner_width = max(0, min(image.shape[2], inner_width))
        inner_height = max(0, min(image.shape[1], inner_height))
        if inner_width > 0 and inner_height > 0:
            img = np.zeros((image.shape[1], image.shape[2],), dtype=np.float32)
            img[inner_edge_height:inner_edge_height + inner_height + 1, inner_edge_width:inner_edge_width + inner_width + 1, ] = 1.0
            img = torch.from_numpy(img)
            return (img.unsqueeze(0),)

        else:
            img = np.ones((image.shape[1], image.shape[2], ), dtype=np.float32)
            img = torch.from_numpy(img)
            return (img.unsqueeze(0),)



class InputMaskImageNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "var_name": ("STRING", {"multiline": False, "default": "InputImageMask"}),
                "image": ("STRING", {"multiline": False}),
                "export": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "description": ("STRING", {"multiline": False, "default": ""}),
                "order": ("INT", {"default": 0, "min": 0, "max": 0xffffff, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    CATEGORY = "comfyui-master/输入"
    FUNCTION = "input_image"

    def input_image(self, var_name, image, export, description = "", order = 0):
        try:
            imgdata = base64.b64decode(image)
            img = Image.open(BytesIO(imgdata))
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img)
            if img.dim() == 3:  # RGB(A) input, use red channel
                img = img[:, :, 0]
            return self.read_image(image), img.unsqueeze(0)
        except Exception as e:
            print(f"Exception raised: {var_name}")
            raise e

    def read_image(self, image: str):
        imgdata = base64.b64decode(image)
        img = Image.open(BytesIO(imgdata))

        img = img.convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)[None,]

        return img


class InputStringNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "var_name": ("STRING", {"multiline": False, "default": "InputString"}),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "export": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "description": ("STRING", {"multiline": False, "default": ""}),
                "order": ("INT", {"default": 0, "min": 0, "max": 0xffffff, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("text", )
    CATEGORY = "comfyui-master/输入"
    FUNCTION = "input_string"

    def input_string(self, var_name, text, export, description = "", order = 0):
        return (text, )

class InputEnumStringNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "var_name": ("STRING", {"multiline": False, "default": "InputEnum"}),
                "text": ("STRING", {"multiline": False, "default": ""}),
                "export": ("BOOLEAN", {"default": True}),
                "enums": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "description": ("STRING", {"multiline": False, "default": ""}),
                "order": ("INT", {"default": 0, "min": 0, "max": 0xffffff, "step": 1}),
                "default_generate_algorithm": (["固定值", "随机值"], {"tooltip": "默认生成算法", "default": "固定值"})
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("text", )
    CATEGORY = "comfyui-master/输入"
    FUNCTION = "input_enum_string"

    def input_enum_string(self, var_name, text, enums, export, description = "", order = 0,
                  default_generate_algorithm = "固定值"):
        # Split enums by comma or newline, and strip whitespace
        enums = [enum.strip() for enum in enums.replace('\n', ',').split(',') if enum.strip()]
        if text not in enums:
            enums.append(text)
        return (text, )
    

class InputBooleanNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "var_name": ("STRING", {"multiline": False, "default": "InputBoolean"}),
                "value": ("BOOLEAN", {"default": False}),
                "export": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "description": ("STRING", {"multiline": False, "default": ""}),
                "order": ("INT", {"default": 0, "min": 0, "max": 0xffffff, "step": 1}),
                "default_generate_algorithm": (["固定值", "随机值"], {"tooltip": "默认生成算法", "default": "固定值"})
            }
        }

    RETURN_TYPES = ("BOOLEAN", )
    RETURN_NAMES = ("value", )
    CATEGORY = "comfyui-master/输入"
    FUNCTION = "input_boolean"

    def input_boolean(self, var_name, value, export, description = "", order = 0,
                  default_generate_algorithm = "固定值"):
        return (value, )
    

class InputIntNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "var_name": ("STRING", {"multiline": False, "default": "InputInt"}),
                "number": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "export": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "description": ("STRING", {"multiline": False, "default": ""}),
                "order": ("INT", {"default": 0, "min": 0, "max": 0xffffff, "step": 1}),
                "default_generate_algorithm": (["固定值", "随机值", "递增", "递减"], {"tooltip": "默认生成算法", "default": "固定值"})
            }
        }

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("number", )
    CATEGORY = "comfyui-master/输入"
    FUNCTION = "input_int"

    def input_int(self, var_name, number, export, description = "", order = 0,
                  default_generate_algorithm = "固定值"):
        return (number, )
    

class InputRangeIntNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "var_name": ("STRING", {"multiline": False, "default": "InputRangeInt"}),
                "number": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "export": ("BOOLEAN", {"default": True}),
                "min": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "max": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
            },
            "optional": {
                "description": ("STRING", {"multiline": False, "default": ""}),
                "order": ("INT", {"default": 0, "min": 0, "max": 0xffffff, "step": 1}),
                "default_generate_algorithm": (
                ["固定值", "随机值", "递增", "递减"], {"tooltip": "默认生成算法", "default": "固定值"})
            }
        }

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("number", )
    CATEGORY = "comfyui-master/输入"
    FUNCTION = "input_range_int"

    def input_range_int(self, var_name, number, min, max, export, description = "", order = 0,
                  default_generate_algorithm = "固定值"):
        if min > max:
            min, max = max, min
        if number < min:
            number = min
        if number > max:
            number = max
        return (number,)
    

class InputFloatNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "var_name": ("STRING", {"multiline": False, "default": "InputFloat"}),
                "number": ("FLOAT", {"default": 0, "min": 0, "max": 0xffffffffffff, "step": 0.01}),
                "export": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "description": ("STRING", {"multiline": False, "default": ""}),
                "order": ("INT", {"default": 0, "min": 0, "max": 0xffffff, "step": 1}),
                "default_generate_algorithm": (
                ["固定值", "随机值", "递增", "递减"], {"tooltip": "默认生成算法", "default": "固定值"})
            }
        }

    RETURN_TYPES = ("FLOAT", )
    RETURN_NAMES = ("number", )
    CATEGORY = "comfyui-master/输入"
    FUNCTION = "input_float"

    def input_float(self, var_name, number, export, description = "", order = 0,
                  default_generate_algorithm = "固定值"):
        return (number, )
    

class InputRangeFloatNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "var_name": ("STRING", {"multiline": False, "default": "InputRangeFloat"}),
                "number": ("FLOAT", {"default": 0, "min": 0, "max": 0xffffffffffff, "step": 0.01}),
                "export": ("BOOLEAN", {"default": True}),
                "min": ("FLOAT", {"default": 0, "min": 0, "max": 0xffffffffffff, "step": 0.01}),
                "max": ("FLOAT", {"default": 0, "min": 0, "max": 0xffffffffffff, "step": 0.01}),
            },
            "optional": {
                "description": ("STRING", {"multiline": False, "default": ""}),
                "order": ("INT", {"default": 0, "min": 0, "max": 0xffffff, "step": 1}),
                "default_generate_algorithm": (
                ["固定值", "随机值", "递增", "递减"], {"tooltip": "默认生成算法", "default": "固定值"})
            }
        }

    RETURN_TYPES = ("FLOAT", )
    RETURN_NAMES = ("number", )
    CATEGORY = "comfyui-master/输入"
    FUNCTION = "input_range_float"

    def input_range_float(self, var_name, number, min, max, export, description = "", order = 0,
                  default_generate_algorithm = "固定值"):
        if min > max:
            min, max = max, min
        if number < min:
            number = min
        if number > max:
            number = max
        return (number, )
    

class OutputStringNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "var_name": ("STRING", {"multiline": False, "default": "OutputString"}),
                "text": ("STRING", {"multiline": True, "forceInput": True}),
                "export": ("BOOLEAN", {"default": True, }),
            },
            "optional": {
                "description": ("STRING", {"multiline": False, "default": ""}),
                "order": ("INT", {"default": 0, "min": 0, "max": 0xffffff, "step": 1}),
            }
        }

    RETURN_TYPES = ()
    CATEGORY = "comfyui-master/输出"
    OUTPUT_NODE = True
    FUNCTION = "output_string"

    def output_string(self, var_name, text, export, description = "", order = 0):
        server = PromptServer.instance
        server.send_sync(100001, encode_string(var_prefix_name + var_name, text), server.client_id)

        return {"ui": {"text": [{"var_name": var_prefix_name + var_name, "text": text}]}}


class OutputImageNode:

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.filename_prefix = "ComfyMasterOutput_"
        self.type = "temp"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "var_name": ("STRING", {"multiline": False, "default": "OutputImage"}),
                "images": ("IMAGE", {"forceInput": True}),
                "export": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "description": ("STRING", {"multiline": False, "default": ""}),
                "order": ("INT", {"default": 0, "min": 0, "max": 0xffffff, "step": 1}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_images"
    OUTPUT_NODE = True
    CATEGORY = "comfyui-master/输出"

    def send_images(self, var_name, images, export, description = "", order = 0):
        try:
            var_name = var_prefix_name + var_name
            filename_prefix = self.filename_prefix + var_name
            results = []

            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
            for i, tensor in enumerate(images):
                array = 255.0 * tensor.cpu().numpy()
                image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

                server = PromptServer.instance
                server.send_sync(100002, encode_image(var_prefix_name + var_name, image), server.client_id)
                filename_with_batch_num = filename.replace("%batch_num%", str(i))
                file = f"{filename_with_batch_num}_{counter:05}_.png"
                image.save(os.path.join(full_output_folder, file), pnginfo=None, compress_level=1)
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })

            return {"ui": {"images": results}}
        except Exception as e:
            print(f"Exception: {var_name}")
            raise e


class OutputAudioNode:

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.filename_prefix = "ComfyMasterOutput_"
        self.type = "temp"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "var_name": ("STRING", {"multiline": False, "default": "OutputAudio"}),
                "audio": ("AUDIO",),
                "export": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "description": ("STRING", {"multiline": False, "default": ""}),
                "order": ("INT", {"default": 0, "min": 0, "max": 0xffffff, "step": 1}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_audio"
    OUTPUT_NODE = True
    CATEGORY = "comfyui-master/输出"

    def send_audio(self, var_name, audio, export, description = "", order = 0):
        var_name = var_prefix_name + var_name
        filename_prefix = self.filename_prefix + var_name
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            os.path.join("audio", filename_prefix), self.output_dir)
        results = list()

        for (batch_number, waveform) in enumerate(audio["waveform"].cpu()):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.wav"

            buff = BytesIO()
            torchaudio.save(buff, waveform, audio["sample_rate"], format="WAV")
            server = PromptServer.instance
            server.send_sync(100003, encode_audio(var_prefix_name + var_name, waveform, audio["sample_rate"]),
                             server.client_id)

            with open(os.path.join(full_output_folder, file), 'wb') as f:
                f.write(buff.getbuffer())

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"audio": results}}
