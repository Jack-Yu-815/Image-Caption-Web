import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from utils import get_device, load_model
from models import Img2Cap
import re
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")


def save_image_caption(img, caption, file_path=None):
    """
    Parameters
    ----------
    img: np.ndarray
        (3xHxW)
    caption: str
    """
    count = sum(1 for _ in Path("../figs").glob("fig_*_*.png"))
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406

    img = img.transpose((1, 2, 0))

    plt.ioff()
    fig = plt.figure()  # figsize=(5, 3)
    ax = fig.add_subplot(111)
    ax.imshow(img)
    # ax.text(10, 220, caption)
    ax.set_title(caption, wrap=True)
    if file_path is not None:
        fig.savefig(file_path)
    plt.ion()


def beautify(sentence: str) -> str:
    ans = re.sub(r' <EOS>( <PAD>)*', '.', sentence)
    ans = re.sub(r'<SOS> ', '', ans)
    ans = ans[0].upper() + ans[1:]
    return ans


def infer_from_file(img_path):
    weights_path = "8K_caption_24.torch"

    img_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    with open("tokenizer.pickle", "rb") as file:
        tokenizer = pickle.load(file)
    device = get_device()
    model = Img2Cap(tokenizer, 400, torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))["model_state_dict"])
    model = model.to(device)
    model.eval()

    try:
        img = Image.open(img_path).convert("RGB")
        actual_img = ImageOps.exif_transpose(img)  # pass this to visualization, but not model's input because model is trained on images without rotation correction
        img = img_transform(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2])).to(device)
        actual_img = img_transform(actual_img)
        actual_img = actual_img.reshape((1, actual_img.shape[0], actual_img.shape[1], actual_img.shape[2])).to(device)

        seq = model.predict(img)
        sentence = beautify(tokenizer.sequences_to_texts([seq])[0])
        return sentence
        # save_image_caption(actual_img[0].cpu().detach().numpy(), sentence, f"../figs/figures_8K_{img_path.stem}.png")
    except Exception as err:
        print(err)
    finally:
        p = Path(img_path)
        if p.exists() and p.is_file():
            p.unlink()


def infer_from_image(img):

    weights_path = "8K_caption_24.torch"

    img_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    with open("tokenizer.pickle", "rb") as file:
        tokenizer = pickle.load(file)
    device = get_device()
    model = Img2Cap(tokenizer, 400, torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))["model_state_dict"])
    model = model.to(device)
    model.eval()

    try:
        img = img.convert("RGB")
        actual_img = ImageOps.exif_transpose(img)  # pass this to visualization, but not model's input because model is trained on images without rotation correction
        img = img_transform(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2])).to(device)
        actual_img = img_transform(actual_img)
        actual_img = actual_img.reshape((1, actual_img.shape[0], actual_img.shape[1], actual_img.shape[2])).to(device)

        seq = model.predict(img)
        sentence = beautify(tokenizer.sequences_to_texts([seq])[0])
        return sentence
        # save_image_caption(actual_img[0].cpu().detach().numpy(), sentence, f"../figs/figures_8K_{img_path.stem}.png")
    except Exception as err:
        print(err)


if __name__ == "__main__":
    infer_from_file("/Users/jackyu/Desktop/images")
    pass
