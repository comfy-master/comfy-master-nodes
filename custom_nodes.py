from server import PromptServer
import folder_paths
from PIL import Image
import base64
from io import BytesIO
import torch
import numpy as np
from .encoding import encode_string, encode_image, encode_audio
import os
import torchaudio
from datetime import datetime
import comfy.sd
import comfy.utils

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
                "allowAutoGenerate": ("BOOLEAN", {"default": False}),
                "allowLocalRepair": ("BOOLEAN", {"default": False}),
                "order": ("INT", {"default": 0, "min": 0, "max": 0xffffff, "step": 1}),
            }
        }


    RETURN_TYPES = ()
    CATEGORY = "comfyui-master"
    FUNCTION = "output_func"

    def output_func(self, name, description):
        return ()


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
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")

    CATEGORY = "comfyui-master"
    FUNCTION = "input_checkpoint"

    def input_checkpoint(self, var_name, ckpt_name, checkpoints, export, description, order):
        # Split enums by comma or newline, and strip whitespace
        checkpoints = [enum.strip() for enum in checkpoints.replace('\n', ',').split(',') if enum.strip()]
        if ckpt_name not in checkpoints:
            checkpoints.append(text)

        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True,
                                                    embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[:3]


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
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"

    CATEGORY = "comfyui-master"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."

    def load_lora(self, var_name, model, clip, lora_name, strength_model, strength_clip, export, loras, description, order):
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
    CATEGORY = "comfyui-master"
    FUNCTION = "input_image"

    def input_image(self, var_name, image, export, description, order):
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

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("text", "export")
    CATEGORY = "comfyui-master"
    FUNCTION = "input_string"

    def input_string(self, var_name, text, export, description, order):
        return (text, export)

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
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("text", "export")
    CATEGORY = "comfyui-master"
    FUNCTION = "input_enum_string"

    def input_enum_string(self, var_name, text, enums, export, description, order):
        # Split enums by comma or newline, and strip whitespace
        enums = [enum.strip() for enum in enums.replace('\n', ',').split(',') if enum.strip()]
        if text not in enums:
            enums.append(text)
        return (text, export)
    

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
            }
        }

    RETURN_TYPES = ("BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("value", "export")
    CATEGORY = "comfyui-master"
    FUNCTION = "input_boolean"

    def input_boolean(self, var_name, value, export, description, order):
        return (value, export)
    

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
            }
        }

    RETURN_TYPES = ("INT", "BOOLEAN")
    RETURN_NAMES = ("number", "export")
    CATEGORY = "comfyui-master"
    FUNCTION = "input_int"

    def input_int(self, var_name, number, export, description, order):
        return (number, export)
    

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
            }
        }

    RETURN_TYPES = ("INT", "BOOLEAN")
    RETURN_NAMES = ("number", "export")
    CATEGORY = "comfyui-master"
    FUNCTION = "input_range_int"

    def input_range_int(self, var_name, number, min, max, export, description, order):
        if min > max:
            min, max = max, min
        if number < min:
            number = min
        if number > max:
            number = max
        return (number,export)
    

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
            }
        }

    RETURN_TYPES = ("FLOAT", "BOOLEAN")
    RETURN_NAMES = ("number", "export")
    CATEGORY = "comfyui-master"
    FUNCTION = "input_float"

    def input_float(self, var_name, number, export, description, order):
        return (number, export)
    

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
            }
        }

    RETURN_TYPES = ("FLOAT", "BOOLEAN")
    RETURN_NAMES = ("number", "export")
    CATEGORY = "comfyui-master"
    FUNCTION = "input_range_float"

    def input_range_float(self, var_name, number, min, max, export, description, order):
        if min > max:
            min, max = max, min
        if number < min:
            number = min
        if number > max:
            number = max
        return (number, export)
    

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
    CATEGORY = "comfyui-master"
    OUTPUT_NODE = True
    FUNCTION = "output_string"

    def output_string(self, var_name, text, export, description, order):
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
    CATEGORY = "comfyui-master"

    def send_images(self, var_name, images, export, description, order):
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
    CATEGORY = "comfyui-master"

    def send_audio(self, var_name, audio, export, description, order):
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
