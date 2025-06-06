import os
import torch
import random
import diffusers
import torch.utils
import unet.utils as utils
from unet.unet_controller import UNetController
import argparse
from datetime import datetime
from diffusers.utils import load_image

diffusers.utils.logging.set_verbosity_error()

def load_unet_controller(pipe, device):
    unet_controller = UNetController()
    unet_controller.device = device
    unet_controller.tokenizer = pipe.tokenizer

    return unet_controller

from transformers import DPTImageProcessor, DPTForDepthEstimation

depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
from PIL import Image

def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)
    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

control_image = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
    "/kandinsky/cat.png"
).resize((1024, 1024))

def generate_images(unet_controller: UNetController, pipe, id_prompt, frame_prompt_list, save_dir, window_length, seed, verbose=True):
    generate = torch.Generator().manual_seed(seed)
    if unet_controller.Use_ipca is True:
        unet_controller.Store_qkv = True
        original_prompt_embeds_mode = unet_controller.Prompt_embeds_mode
        unet_controller.Prompt_embeds_mode = "original"
        #_ = pipe(id_prompt, generator=generate, unet_controller=unet_controller).images
        _ = pipe(id_prompt, image=control_image, generator=generate, unet_controller=unet_controller, ).images
        unet_controller.Prompt_embeds_mode = original_prompt_embeds_mode


    unet_controller.Store_qkv = False
    images, story_image = utils.movement_gen_story_slide_windows(
        id_prompt, frame_prompt_list, pipe, window_length, seed, unet_controller, save_dir, verbose=verbose, control_image=control_image
    )

    return images, story_image


def main(device, model_path, save_dir, id_prompt, frame_prompt_list, precision, seed, window_length):
    pipe, _ = utils.load_pipe_from_path(model_path, device, torch.float16 if precision == "fp16" else torch.float32, precision)
    
    unet_controller = load_unet_controller(pipe, device)          
    images, story_image = generate_images(unet_controller, pipe, id_prompt, frame_prompt_list, save_dir, window_length, seed)

    return images, story_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using a specific device.")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for computation (e.g., cuda:0, cpu)')
    parser.add_argument('--model_path', type=str, default='stabilityai/stable-diffusion-xl-base-1.0', help='Path to the model')
    parser.add_argument('--project_base_path', type=str, default='.', help='Path to save the generated images')
    parser.add_argument('--id_prompt', type=str, default="A photo of a red fox with coat", help='Initial prompt for image generation')
    parser.add_argument('--frame_prompt_list', type=str, nargs='+', default=[
        "wearing a scarf in a meadow",
        "playing in the snow",
        "at the edge of a village with river",
    ], help='List of frame prompts')
    parser.add_argument('--precision', type=str, choices=["fp16", "fp32"], default="fp16", help='Model precision')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generation')
    parser.add_argument('--window_length', type=int, default=10, help='Window length for story generation')
    parser.add_argument('--save_padding', type=str, default='test', help='Padding for save directory')
    parser.add_argument('--random_seed', action='store_true', help='Use random seed')
    parser.add_argument('--json_path', type=str,)
    
    args = parser.parse_args()
    if args.random_seed:
        args.seed = random.randint(0, 1000000)

    current_time = datetime.now().strftime("%Y%m%d%H")
    current_time_ = datetime.now().strftime("%M%S")
    save_dir = os.path.join(args.project_base_path, f'result/{current_time}/{current_time_}_{args.save_padding}_seed{args.seed}')
    os.makedirs(save_dir, exist_ok=True)

    if args.json_path is None:
        main(args.device, args.model_path, save_dir, args.id_prompt, args.frame_prompt_list, args.precision, args.seed, args.window_length)
    else:
        import json
        with open(args.json_path, "r") as file:
            data = json.load(file)

        combinations = data["combinations"]

        for combo in combinations:
            main(args.device, args.model_path, save_dir, combo['id_prompt'], combo['frame_prompt_list'], args.precision, args.seed, args.window_length)
