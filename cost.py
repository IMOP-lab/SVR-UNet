### script for test model parameters and FLOPs

import torch
import time
import os
from model import get3dmodel
from thop import profile, clever_format
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'0,1,2,3,4,5,6,7'

network_name = 'CAFSANet'

device = torch.device("cuda:0")
model = get3dmodel(network_name, 1, 5).to(device)  # Make sure this line initializes your model correctly

model.eval()
print('network_name',network_name)



dummy_input = torch.randn(1, 1, 96, 96, 96, device=device)
flops, params = profile(model, (dummy_input,))
with torch.no_grad():
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
# flops, params = clever_format([flops, params], "%.2f")
print('params: ', params, 'flops: ', flops)
print('params: %.2f M, flops: %.2f G' % (params / 1000000.0, flops / 1000000000.0))



# # ---------------------------
# # 2) Speed Benchmark
# # ---------------------------
# def benchmark_inference(
#     model,
#     inp,
#     warmup=30,
#     iters=200,
#     use_fp16=False,
# ):
#     # fp16: 推荐用 autocast（对 conv / matmul 等有效；某些算子可能不支持）
#     # 注意：如果模型里有不支持 fp16 的 op，可能报错或回退
#     starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#     times_ms = []

#     # warmup
#     with torch.inference_mode():
#         for _ in range(warmup):
#             if use_fp16:
#                 with torch.cuda.amp.autocast(dtype=torch.float16):
#                     _ = model(inp)
#             else:
#                 _ = model(inp)

#     torch.cuda.synchronize()

#     # measure
#     with torch.inference_mode():
#         for _ in range(iters):
#             starter.record()
#             if use_fp16:
#                 with torch.cuda.amp.autocast(dtype=torch.float16):
#                     _ = model(inp)
#             else:
#                 _ = model(inp)
#             ender.record()
#             torch.cuda.synchronize()  # 等 GPU 真跑完
#             times_ms.append(starter.elapsed_time(ender))

#     times_ms = np.array(times_ms, dtype=np.float64)
#     stats = {
#         "mean_ms": float(times_ms.mean()),
#         "std_ms": float(times_ms.std(ddof=1)) if len(times_ms) > 1 else 0.0,
#         "p50_ms": float(np.percentile(times_ms, 50)),
#         "p90_ms": float(np.percentile(times_ms, 90)),
#         "p95_ms": float(np.percentile(times_ms, 95)),
#         "min_ms": float(times_ms.min()),
#         "max_ms": float(times_ms.max()),
#     }
#     # throughput: batch / sec
#     bsz = inp.shape[0]
#     stats["throughput_samples_s"] = float(bsz * 1000.0 / stats["mean_ms"])
#     return stats

# # FP32
# fp32_stats = benchmark_inference(model, dummy_input, warmup=30, iters=200, use_fp16=False)
# print("\n[FP32] latency(ms): mean={mean_ms:.3f}, p50={p50_ms:.3f}, p90={p90_ms:.3f}, p95={p95_ms:.3f}, min={min_ms:.3f}, max={max_ms:.3f}".format(**fp32_stats))
# print("[FP32] throughput(samples/s): {throughput_samples_s:.2f}".format(**fp32_stats))

# # FP16（可选）
# fp16_stats = benchmark_inference(model, dummy_input, warmup=30, iters=200, use_fp16=True)
# print("\n[FP16 autocast] latency(ms): mean={mean_ms:.3f}, p50={p50_ms:.3f}, p90={p90_ms:.3f}, p95={p95_ms:.3f}, min={min_ms:.3f}, max={max_ms:.3f}".format(**fp16_stats))
# print("[FP16 autocast] throughput(samples/s): {throughput_samples_s:.2f}".format(**fp16_stats))

# device = torch.device("cuda:7")  # pick one GPU for profiling

# model = get3dmodel(network_name, 1, 5).to(device).eval()
# dummy_input = torch.randn(1, 1, 96, 96, 96, device=device)

# with torch.no_grad():
#     flops, params = profile(model, inputs=(dummy_input,), verbose=False)

# # Transfer the model to the appropriate device (GPU or CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device:",device)
# model.to(device)

# model.eval()
# # Generate a data samples
# inputs = torch.randn(1, 1, 96, 96, 96).to(device)  # Adjust the size according to your model input
# ass = model(inputs)
# # Record start time to measure model inference speed
# start_time = time.time()

# # Processing samples
# for i in range(100):  # Adding an extra batch dimension
#     ass = model(inputs)
# # Calculate total inference time
# end_time = time.time()
# total_time = end_time - start_time

# print(f"Time required to process 1 samples:{total_time*10}ms")
# print(f"Time required to process 100 samples:{total_time}s")
