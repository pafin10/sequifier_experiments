import torch
import time
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from torch import Tensor
from sequifier.train_old import RegionAttention as RegionAttentionV1
from sequifier.train_old import RegionEncoder as RegionEncoderV1
from sequifier.train import ParallelRegionEncoderLayer as ParallelRegionEncoderLayerV2
from sequifier.train import RegionAttention as RegionAttentionV2
from sequifier.train import RegionEncoder as RegionEncoderV2


def check_equivalence():
    """
    Checks the mathematical equivalence of RegionEncoderV1 and RegionEncoderV2.

    This test verifies that the refactored, parallelized RegionEncoderV2 produces
    the exact same output as the original RegionEncoderV1 when given the same
    inputs and weights.

    It works by:
    1. Initializing an instance of the V1 encoder.
    2. Creating a V2 encoder by programmatically transferring the weights from the V1 instance.
    3. Passing a random tensor through both encoders.
    4. Asserting that their outputs are numerically very close.
    """
    # 1. Define hyperparameters for the test models
    B, T, R, D_EMBED, N_LAYERS, DROP = 10, 30, 16, 64, 4, 0.1
    NUM_HEADS = R
    D_MODEL = R * D_EMBED
    D_HEAD = D_EMBED  # In this architecture, d_head equals d_embed

    # 2. Instantiate V1 Encoder with random weights
    layers_v1 = nn.ModuleList([
        nn.ModuleList([
            RegionAttentionV1(num_regions=R, d_embed=D_EMBED, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROP)
            for _ in range(R)
        ])
        for _ in range(N_LAYERS)
    ])
    encoder_v1 = RegionEncoderV1(layers=layers_v1, num_regions=R, d_model=D_MODEL, d_head=D_HEAD, drop=DROP)

    # 3. Instantiate V2 Encoder and transfer weights from V1
    parallel_layers_v2 = nn.ModuleList()
    for l in range(N_LAYERS):
        # Create a temporary list of V2-style attention modules to hold V1's weights
        layer_list_v2_temp = nn.ModuleList()
        for i in range(R):
            attn_v1 = encoder_v1.layers[l][i]
            attn_v2_temp_module = RegionAttentionV2(num_regions=R, d_embed=D_EMBED, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROP)

            # Manually copy and reshape weights from V1's format to V2's format
            w_q_stacked = torch.stack([p.data for p in attn_v1.W_Q], dim=0)
            attn_v2_temp_module.W_Q.data.copy_(w_q_stacked)
            
            w_k_stacked = torch.stack([p.data for p in attn_v1.W_K], dim=0)
            w_v_stacked = torch.stack([p.data for p in attn_v1.W_V], dim=0)
            attn_v2_temp_module.W_KV.data.copy_(torch.cat([w_k_stacked, w_v_stacked], dim=-1))
            
            attn_v2_temp_module.alpha.data.copy_(attn_v1.alpha.data)
            layer_list_v2_temp.append(attn_v2_temp_module)
            
        # Create the parallel layer for V2 using the weight-copied V2 temp modules
        parallel_layer = ParallelRegionEncoderLayerV2(layer_list_v2_temp, encoder_v1.out_proj)
        parallel_layers_v2.append(parallel_layer)
        
    encoder_v2 = RegionEncoderV2(parallel_layers=parallel_layers_v2, d_model=D_MODEL, drop=DROP)
    
    # Copy remaining weights (FFN, LayerNorms)
    encoder_v2.ffn.load_state_dict(encoder_v1.ffn.state_dict())
    encoder_v2.ln1.load_state_dict(encoder_v1.ln1.state_dict())
    encoder_v2.ln2.load_state_dict(encoder_v1.ln2.state_dict())

    # 4. Set models to evaluation mode and create a random input tensor
    encoder_v1.eval()
    encoder_v2.eval()
    src = torch.randn(B, T, D_MODEL)


    # 5. Get outputs and assert equivalence
    with torch.no_grad():
        out_v1 = encoder_v1(src)
        out_v2 = encoder_v2(src)

    are_equal = torch.allclose(out_v1, out_v2, atol=1e-5)
    print(f"Equivalence Test Passed: {are_equal}")
    print(f"Max absolute difference: {(out_v1 - out_v2).abs().max().item()}")
    assert are_equal, "The outputs of RegionEncoderV1 and RegionEncoderV2 are not equivalent."


    print("Starting a proper performance benchmark...")
    num_runs = 50
    warmup_runs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        # --- V1 Benchmark ---
        print("Warming up V1...")
        for _ in range(warmup_runs):
            _ = encoder_v1(src)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        print("Benchmarking V1...")
        start_time_v1 = time.perf_counter()
        for _ in range(num_runs):
            _ = encoder_v1(src)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time_v1 = time.perf_counter()
        avg_time_v1 = (end_time_v1 - start_time_v1) / num_runs
        print(f"V1 Encoder :: Average time: {avg_time_v1:.6f} seconds")

        # --- V2 Benchmark ---
        print("\nWarming up V2...")
        for _ in range(warmup_runs):
            _ = encoder_v2(src)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        print("Benchmarking V2...")
        start_time_v2 = time.perf_counter()
        for _ in range(num_runs):
            _ = encoder_v2(src)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time_v2 = time.perf_counter()
        avg_time_v2 = (end_time_v2 - start_time_v2) / num_runs
        print(f"V2 Encoder :: Average time: {avg_time_v2:.6f} seconds")


if __name__ == '__main__':
    check_equivalence()