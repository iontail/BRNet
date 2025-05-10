import os
import random
from PIL import Image
from utils.DarkISP import Low_Illumination_Degrading
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch


WIDERFACE_VAL_DIR = "'./dataset/WiderFace/Wider_train"
OUTPUT_DIR = "output_images"  


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    data_path = 'dataset/WiderFace/WIDER_train/images/12--Group/'
    img_name = '12_Group_Group_12_Group_Group_12_51.jpg'
    img_path = os.path.join(data_path, img_name)

    img = Image.open(img_path)
    img.show()
    img = img.convert('RGB')
    img_tensor = to_tensor(img).to(DEVICE)


    darkisp_tensor, _ = Low_Illumination_Degrading(img_tensor)

    # darkisp_image는 텐서 형태이므로, 다시 PIL 이미지로 변환
    darkisp_image_pil = to_pil_image(darkisp_tensor.cpu()) 

    darkisp_save_path = os.path.join(OUTPUT_DIR, f"{img_name[:-4]}_darkisp.png")

    # 디렉토리가 존재하지 않으면 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    darkisp_image_pil.save(darkisp_save_path)

if __name__ == "__main__":
    main()