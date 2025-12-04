import torch, time

for i in range(10):
    a = torch.randn(25000, 25000, device="cuda", dtype=torch.float16)
    b = torch.randn(25000, 25000, device="cuda", dtype=torch.float16)

    torch.cuda.synchronize()
    start = time.time()
    c = a @ b
    torch.cuda.synchronize()

    print("GPU FP16 matmul time:", time.time() - start)    
    print(i)


