import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from models.noise_aware_dncnn_v2 import NoiseAwareDnCNNV2


def main():
    """Test Noise-Aware DnCNN V2 architecture and verify tensor shapes."""

    print("=" * 70)
    print("NOISE-AWARE DnCNN V2 SHAPE TEST")
    print("=" * 70)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Instantiate model
    print("\nInstantiating NoiseAwareDnCNNV2...")
    model = NoiseAwareDnCNNV2(num_features=64, num_blocks=10).to(device)
    model.eval()

    print(f"Model architecture:")
    print(f"  - Input channels: 2")
    print(f"  - Features: 64")
    print(f"  - Residual blocks: 10")
    print(f"  - Output channels: 1")
    print(f"  - Upsampling: 2x PixelShuffle")

    # Create test input
    print("\nCreating test input...")
    test_input = torch.randn(2, 2, 128, 128).to(device)
    print(f"  Input shape: {test_input.shape}")

    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        output = model(test_input)

    print(f"  Output shape: {output.shape}")

    # Verify shapes
    print("\n" + "=" * 70)
    print("SHAPE VERIFICATION")
    print("=" * 70)

    expected_input_shape = torch.Size([2, 2, 128, 128])
    expected_output_shape = torch.Size([2, 1, 256, 256])

    print(f"\nExpected input shape:  {expected_input_shape}")
    print(f"Actual input shape:    {test_input.shape}")
    input_ok = test_input.shape == expected_input_shape
    print(f"Input shape match: {'✓ PASS' if input_ok else '✗ FAIL'}")

    print(f"\nExpected output shape: {expected_output_shape}")
    print(f"Actual output shape:   {output.shape}")
    output_ok = output.shape == expected_output_shape

    print(f"Output shape match: {'✓ PASS' if output_ok else '✗ FAIL'}")

    # Final result
    print("\n" + "=" * 70)
    if input_ok and output_ok:
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        return 0
    else:
        print("✗ TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit(main())
