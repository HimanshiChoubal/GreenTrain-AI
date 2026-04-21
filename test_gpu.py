import pynvml
try:
    pynvml.nvmlInit()
    print("Success: NVIDIA Management Library initialized!")
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"Found {device_count} GPU(s).")
except Exception as e:
    print(f"Error: {e}")