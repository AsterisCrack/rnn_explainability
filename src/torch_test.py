import torch

if __name__ == "__main__":
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"CUDA is available with {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available")