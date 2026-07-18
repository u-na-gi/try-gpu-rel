import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

if torch.cuda.is_available:
    print("cude", True)
else:
    print("cude", False)


@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    # NOTE: `constexpr` so it can be used as a shape value.
):
    # ここでいうprogramとはなんなのか
    pid = tl.program_id(axis=0)
    tl.device_print("pid", pid)

    block_start = pid * BLOCK_SIZE
    tl.device_print("block_start", block_start)

    offsets = block_start + tl.arange(0, BLOCK_SIZE)


def add(x: torch.Tensor, y: torch.Tensor):
    # あらかじめアロケータをもっておく
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)


def main():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output_torch = x + y
    print(output_torch)

    output_triton = add(x, y)
    pass


if __name__ == "__main__":
    main()
