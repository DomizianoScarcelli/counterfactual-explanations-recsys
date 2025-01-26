import torch
import fire
from config import ConfigParams

def print_pth(pth_path:str):
    pth_file = torch.load(pth_path, map_location=ConfigParams.DEVICE)
    print(pth_file)

if __name__ == "__main__":
    fire.Fire(print_pth)

