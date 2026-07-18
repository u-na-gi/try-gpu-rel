import torch
from kernel import add
from const import DEVICE

if torch.cuda.is_available():
    print("cude", True)
else:
    print("cude", False)


def main():
    torch.manual_seed(0)
    size = 9843200
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output_torch = x + y
    print(output_torch)

    output_triton = add(x, y)

    print(output_torch)
    print(output_triton)
    print(
        f"The maximum difference between torch and triton is "
        f"{torch.max(torch.abs(output_torch - output_triton))}"
    )


if __name__ == "__main__":
    main()
