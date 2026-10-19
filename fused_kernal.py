@triton.jit
def add_relu_kernel(a_ptr, b_ptr, out_ptr, n,
                    BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    a = tl.load(a_ptr + offsets, mask=mask, other=0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0)
    result = tl.maximum(a + b, 0)

    tl.store(out_ptr + offsets, result, mask=mask)
a = torch.randn(1_000_000, device="cuda")
b = torch.randn_like(a)
out = torch.empty_like(a)
grid = (triton.cdiv(a.numel(), 1024),)
def run_triton():
    add_relu_kernel[grid](a, b, out, a.numel(), BLOCK_SIZE=1024)
def run_torch():
    torch.add(a, b, out=out)
    out.relu_()
run_triton()
torch.testing.assert_close(out, torch.relu(a + b))

triton_ms = triton.testing.do_bench(run_triton)
torch_ms = triton.testing.do_bench(run_torch)

print(f"Triton:  {triton_ms:.6f} ms")
print(f"PyTorch: {torch_ms:.6f} ms")
print(f"Speedup: {torch_ms / triton_ms:.2f}x")