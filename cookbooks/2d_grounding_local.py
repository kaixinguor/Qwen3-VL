"""
Spatial Understanding with Qwen3-VL (Local Model)

This script showcases Qwen3-VL's advanced spatial localization abilities using a local model.
It replaces API calls with local model inference for the same functionality.
"""

import json
import ast
import os
import torch
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
from transformers import AutoModelForImageTextToText, AutoProcessor

# 可视化工具函数
additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def decode_json_points(text: str):
    """Parse coordinate points from text format"""
    try:
        # 清理markdown标记
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        
        # 解析JSON
        data = json.loads(text)
        points = []
        labels = []
        
        for item in data:
            if "point_2d" in item:
                x, y = item["point_2d"]
                points.append([x, y])
                
                # 获取label，如果没有则使用默认值
                label = item.get("label", f"point_{len(points)}")
                labels.append(label)
        
        return points, labels
        
    except Exception as e:
        print(f"Error: {e}")
        return [], []


def parse_json(json_output):
    """Parsing out the markdown fencing"""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output


def plot_bounding_boxes(im, bounding_boxes):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        im: PIL Image object
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """
    # Load the image
    img = im
    width, height = img.size
    print(f"Image size: {img.size}")
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
        'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
        'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)

    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    except:
        font = ImageFont.load_default()

    try:
        json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
        end_idx = bounding_boxes.rfind('"}') + len('"}')
        truncated_text = bounding_boxes[:end_idx] + "]"
        json_output = ast.literal_eval(truncated_text)

    if not isinstance(json_output, list):
        json_output = [json_output]

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json_output):
        # Select a color from the list
        color = colors[i % len(colors)]

        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box["bbox_2d"][1] / 1000 * height)
        abs_x1 = int(bounding_box["bbox_2d"][0] / 1000 * width)
        abs_y2 = int(bounding_box["bbox_2d"][3] / 1000 * height)
        abs_x2 = int(bounding_box["bbox_2d"][2] / 1000 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Draw the bounding box
        draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=3
        )

        # Draw the text
        if "label" in bounding_box:
            draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    # Display the image
    # img.show()
    
    # With matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()

    # Save permanently instead
    img.save('output.png')


def plot_points(im, text):
    """Plot points on an image"""
    img = im
    width, height = img.size
    draw = ImageDraw.Draw(img)
    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
        'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
        'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
    ] + additional_colors

    points, descriptions = decode_json_points(text)
    print("Parsed points: ", points)
    print("Parsed descriptions: ", descriptions)
    if points is None or len(points) == 0:
        img.show()
        return

    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    except:
        font = ImageFont.load_default()

    for i, point in enumerate(points):
        color = colors[i % len(colors)]
        abs_x1 = int(point[0])/1000 * width
        abs_y1 = int(point[1])/1000 * height
        radius = 2
        draw.ellipse([(abs_x1 - radius, abs_y1 - radius), (abs_x1 + radius, abs_y1 + radius)], fill=color)
        draw.text((abs_x1 - 20, abs_y1 + 6), descriptions[i], fill=color, font=font)
    
    img.show()


def plot_points_json(im, text):
    """Plot points from JSON format"""
    img = im
    width, height = img.size
    draw = ImageDraw.Draw(img)
    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
        'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
        'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
    ] + additional_colors
    
    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    except:
        font = ImageFont.load_default()

    text = text.replace('```json', '')
    text = text.replace('```', '')
    data = json.loads(text)
    for item in data:
        point_2d = item['point_2d']
        label = item['label']
        x, y = int(point_2d[0] / 1000 * width), int(point_2d[1] / 1000 * height)
        radius = 2
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=colors[0])
        draw.text((x + 2*radius, y + 2*radius), label, fill=colors[0], font=font)
    
    img.show()


class Qwen3VLLocalInference:
    """本地模型推理类"""
    
    def __init__(self, model_path="models/qwen3-vl-4b-instruct", device="auto", dtype="auto"):
        """
        初始化本地模型
        
        Args:
            model_path: 模型路径，可以是本地路径或 HuggingFace 模型名称
            device: 设备，默认为 "auto" 自动选择
            dtype: 数据类型，默认为 "auto" 自动选择
        """
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device,
            torch_dtype=torch.bfloat16 if dtype == "auto" else dtype,
        )
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        print("Model loaded successfully!")
    
    def inference(self, img_url, prompt, min_pixels=64 * 32 * 32, max_pixels=9800 * 32 * 32):
        """
        使用本地模型进行推理
        
        Args:
            img_url: 图片路径（本地文件路径或 URL）
            prompt: 提示文本
            min_pixels: 最小像素数
            max_pixels: 最大像素数
            
        Returns:
            模型生成的文本响应
        """
        # 处理图片路径
        # 如果 img_url 已经是 PIL Image 对象，直接使用
        if isinstance(img_url, Image.Image):
            image_path = img_url
        elif os.path.exists(img_url):
            # 本地文件，直接使用绝对路径（transformers 支持直接路径，不需要 file:// 前缀）
            image_path = os.path.abspath(img_url)
        elif img_url.startswith("http://") or img_url.startswith("https://"):
            # URL 直接使用
            image_path = img_url
        else:
            # 尝试使用绝对路径
            abs_path = os.path.abspath(img_url)
            if os.path.exists(abs_path):
                image_path = abs_path
            else:
                # 如果路径不存在，尝试直接使用原路径（可能是相对路径）
                image_path = img_url
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # 准备输入
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        # 推理：生成输出
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        return output_text[0] if output_text else ""


# 全局模型实例（可选，用于兼容函数）
_global_inference_engine = None

def inference_with_local_model(img_url, prompt, min_pixels=64 * 32 * 32, max_pixels=9800 * 32 * 32, model=None):
    """
    兼容函数：与原来的 inference_with_openai_api 函数接口兼容
    
    使用方式：
        # 方式1：使用全局模型实例
        inference_engine = Qwen3VLLocalInference(model_path="models/qwen3-vl-4b-instruct")
        _global_inference_engine = inference_engine
        result = inference_with_local_model(img_url, prompt)
        
        # 方式2：传入模型实例
        inference_engine = Qwen3VLLocalInference(model_path="models/qwen3-vl-4b-instruct")
        result = inference_with_local_model(img_url, prompt, model=inference_engine)
    
    Args:
        img_url: 图片路径
        prompt: 提示文本
        min_pixels: 最小像素数
        max_pixels: 最大像素数
        model: Qwen3VLLocalInference 实例，如果为 None 则使用全局实例
        
    Returns:
        模型生成的文本响应
    """
    global _global_inference_engine
    inference_engine = model if model is not None else _global_inference_engine
    
    if inference_engine is None:
        raise ValueError("Model not initialized. Please initialize Qwen3VLLocalInference first or pass model parameter.")
    
    return inference_engine.inference(img_url, prompt, min_pixels, max_pixels)


# 示例使用
if __name__ == "__main__":
    # 初始化本地模型推理器
    # 注意：请根据实际情况修改模型路径
    model_path = "models/qwen3-vl-4b-instruct"  # 可以是本地路径或 "Qwen/Qwen3-VL-4B-Instruct"
    
    # 如果模型路径不存在，尝试使用 HuggingFace 模型名称
    if not os.path.exists(model_path):
        model_path = "Qwen/Qwen3-VL-4B-Instruct"
    
    print(f"Initializing model from: {model_path}")
    inference_engine = Qwen3VLLocalInference(model_path=model_path)
    
    # 设置全局模型实例（用于兼容函数）
    _global_inference_engine = inference_engine
    
    # 示例 1: 检测餐桌上的不同物体
    print("\n=== Example 1: Detecting different objects on a dining table ===")
    prompt = 'locate every instance that belongs to the following categories: "plate/dish, scallop, wine bottle, tv, bowl, spoon, air conditioner, coconut drink, cup, chopsticks, person". Report bbox coordinates in JSON format.'
    img_url = "./cookbooks/assets/spatial_understanding/dining_table.png"
    
    if os.path.exists(img_url):
        # 使用直接调用方式
        model_response = inference_engine.inference(img_url, prompt)
        # 或者使用兼容函数：model_response = inference_with_local_model(img_url, prompt)
        print("Model response:\n", model_response)
        
        image = Image.open(img_url)
        image.thumbnail([640, 640], Image.Resampling.LANCZOS)
        plot_bounding_boxes(image, model_response)
    else:
        print(f"Image not found: {img_url}")
    
    # # 示例 2: 检测特定物体
    # print("\n=== Example 2: Detecting specific objects ===")
    # prompt = "Outline the position of each small cake and output all the coordinates in JSON format."
    # img_url = "./assets/spatial_understanding/spatio_case1.jpg"
    
    # if os.path.exists(img_url):
    #     # 使用兼容函数方式
    #     model_response = inference_with_local_model(img_url, prompt)
    #     print("Model response:\n", model_response)
        
    #     image = Image.open(img_url)
    #     image.thumbnail([640, 640], Image.Resampling.LANCZOS)
    #     plot_bounding_boxes(image, model_response)
    # else:
    #     print(f"Image not found: {img_url}")

