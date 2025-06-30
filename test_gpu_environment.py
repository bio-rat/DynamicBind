#!/usr/bin/env python3
"""
Quick GPU environment test for DynamicBind
"""

import os
import subprocess
import sys

def setup_cuda_environment(device_id):
    """Setup CUDA environment with proper library paths for HPC systems"""
    cuda_env = os.environ.copy()
    cuda_env['CUDA_VISIBLE_DEVICES'] = str(device_id)
    
    # Common CUDA library paths on different HPC systems
    cuda_paths = [
        '/usr/local/cuda/lib64',
        '/usr/local/cuda-12.1/lib64',
        '/usr/local/cuda-12/lib64',
        '/usr/local/cuda-11/lib64',
        '/opt/cuda/lib64',
        '/apps/cuda/lib64',
        '/sw/cuda/lib64',
        '/cluster/cuda/lib64'
    ]
    
    # Check for CUDA installation via environment modules
    cuda_home = cuda_env.get('CUDA_HOME')
    if cuda_home:
        cuda_paths.insert(0, os.path.join(cuda_home, 'lib64'))
    
    # Check for existing LD_LIBRARY_PATH
    existing_ld_path = cuda_env.get('LD_LIBRARY_PATH', '')
    
    # Find existing CUDA paths
    valid_cuda_paths = [p for p in cuda_paths if os.path.exists(p)]
    
    # Combine all paths
    if valid_cuda_paths:
        if existing_ld_path:
            cuda_env['LD_LIBRARY_PATH'] = ':'.join(valid_cuda_paths + [existing_ld_path])
        else:
            cuda_env['LD_LIBRARY_PATH'] = ':'.join(valid_cuda_paths)
        print(f"Found CUDA libraries in: {valid_cuda_paths}")
    else:
        print("Warning: No CUDA library paths found. GPU may not work.")
    
    return cuda_env

def test_gpu_environment():
    print("=== GPU Environment Test ===")
    
    device = 0
    python_path = '/nfs/hpc/share/nguyhun2/.conda/envs/dynamicbind/bin/python'
    
    # Set up CUDA environment
    cuda_env = setup_cuda_environment(device)
    
    print(f"CUDA_VISIBLE_DEVICES: {cuda_env.get('CUDA_VISIBLE_DEVICES')}")
    print(f"LD_LIBRARY_PATH: {cuda_env.get('LD_LIBRARY_PATH', 'Not set')}")
    
    # Test PyTorch CUDA
    test_cmd = f'{python_path} -c "import torch; print(f\'PyTorch version: {{torch.__version__}}\'); print(f\'CUDA available: {{torch.cuda.is_available()}}\'); print(f\'CUDA version: {{torch.version.cuda}}\'); print(f\'Number of GPUs: {{torch.cuda.device_count()}}\'); [print(f\'GPU {{i}}: {{torch.cuda.get_device_name(i)}}\') for i in range(torch.cuda.device_count()) if torch.cuda.is_available()]"'
    
    print("\nTesting PyTorch CUDA availability...")
    result = subprocess.run(test_cmd, shell=True, env=cuda_env, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    
    print("=== Test Complete ===")
    
    if "CUDA available: True" in result.stdout:
        print("✅ GPU is properly detected!")
        return True
    else:
        print("❌ GPU is not detected. Check your CUDA installation.")
        return False

if __name__ == "__main__":
    test_gpu_environment() 