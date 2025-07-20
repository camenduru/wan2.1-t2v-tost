import os, json, requests, random, time, cv2, ffmpeg, runpod
from urllib.parse import urlsplit

import torch
from PIL import Image
import numpy as np

from nodes import NODE_CLASS_MAPPINGS
from comfy_extras import nodes_model_advanced, nodes_hunyuan

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
LoraLoaderModelOnly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
EmptyHunyuanLatentVideo = nodes_hunyuan.NODE_CLASS_MAPPINGS["EmptyHunyuanLatentVideo"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
ModelSamplingSD3 = nodes_model_advanced.NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

with torch.inference_mode():
    unet = UNETLoader.load_unet("wan2.1_t2v_14B_fp8_e4m3fn.safetensors", "default")[0]
    clip = CLIPLoader.load_clip("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan")[0]
    lora1 = LoraLoaderModelOnly.load_lora_model_only(unet, "FusionX/Wan2.1_T2V_14B_FusionX_LoRA.safetensors", 1.0)[0]
    lora2 = LoraLoaderModelOnly.load_lora_model_only(lora1, "FusionX/WAN2.1_1990sOldschoolMovieScreencapTheCrow_v1_by-AI_Characters.safetensors", 1.0)[0]
    vae = VAELoader.load_vae("wan_2.1_vae.safetensors")[0]

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

def images_to_mp4(images, output_path, fps=24):
    try:
        frames = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = np.clip(i, 0, 255).astype(np.uint8)
            if img.shape[0] in [1, 3, 4]:
                img = np.transpose(img, (1, 2, 0))
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            frames.append(img)
        temp_files = [f"temp_{i:04d}.png" for i in range(len(frames))]
        for i, frame in enumerate(frames):
            success = cv2.imwrite(temp_files[i], frame[:, :, ::-1])
            if not success:
                raise ValueError(f"Failed to write {temp_files[i]}")
        if not os.path.exists(temp_files[0]):
            raise FileNotFoundError("Temporary PNG files were not created")
        stream = ffmpeg.input('temp_%04d.png', framerate=fps)
        stream = ffmpeg.output(stream, output_path, vcodec='libx264', pix_fmt='yuv420p')
        ffmpeg.run(stream, overwrite_output=True)
        for temp_file in temp_files:
            os.remove(temp_file)
    except Exception as e:
        print(f"Error: {e}")

@torch.inference_mode()
def generate(input):
    try:
        values = input["input"]

        positive_prompt = values['positive_prompt'] # In a dim tea house lit by lanterns, an Asian girl carefully pours tea in slow motion. The camera dollies in on her calm expression, highlighting the steam curling upward in elegant spirals. Warm ambient lighting flickers on wooden walls, creating a cinematic chiaroscuro effect. The atmosphere is intimate and meditative, steeped in cultural grace.
        negative_prompt = values['negative_prompt'] # 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿
        width = values['width'] # 1024
        height = values['height'] # 576
        length = values['length'] # 121
        batch_size = values['batch_size'] # 1
        shift = values['shift'] # 8.0
        cfg = values['cfg'] # 1.0
        sampler_name = values['sampler_name'] # uni_pc
        scheduler = values['scheduler'] # simple
        steps = values['steps'] # 8
        seed = values['seed'] # 1.0
        if seed == 0:
            random.seed(int(time.time()))
            seed = random.randint(0, 18446744073709551615)
        fps = values['fps'] # 24
        filename_prefix = values['filename_prefix'] # wan_t2i
        is_cinematic = values['is_cinematic'] # True

        positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
        negative = CLIPTextEncode.encode(clip, negative_prompt)[0]
        lora = lora2 if is_cinematic else lora1
        model = ModelSamplingSD3.patch(lora, shift)[0]
        latent_image = EmptyHunyuanLatentVideo.generate(width, height, length, batch_size=batch_size)[0]
        samples = KSampler.sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image)[0]
        decoded_images = VAEDecode.decode(vae, samples)[0].detach()
        images_to_mp4(decoded_images, f"/content/{filename_prefix}.mp4", fps)

        result = f"/content/{filename_prefix}.mp4"

        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        with open(result, 'rb') as file:
            response = requests.post("https://upload.tost.ai/api/v1", files={'file': file})
        response.raise_for_status()
        result_url = response.text
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})