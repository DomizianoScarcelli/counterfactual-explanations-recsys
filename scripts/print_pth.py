import torch
import fire

def print_pth(pth_path:str):
    pth_file = torch.load(pth_path)
    print(pth_file)

if __name__ == "__main__":
    fire.Fire(print_pth)

