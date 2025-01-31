import torch
import time
import extension_cpp
# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create two random 1024x1024 matrices
a = torch.randn(1024, 1024, dtype=torch.float32, device=device)
b = torch.randn(1024, 1024, dtype=torch.float32, device=device)

# Measure execution time
start_time = time.time()

# Matrix multiplication
# result = extension_cpp.ops.mm_new_8(a, b)
result = extension_cpp.mm_new_8(a, b)
#result = torch.mm(a, b)  # Or: result = a @ b

# Synchronize if using GPU to measure correct time
if device.type == "cuda":
    torch.cuda.synchronize()

end_time = time.time()

# Print execution time
print(f"Matrix multiplication completed in {end_time - start_time:.6f} seconds")

# Optional: Print a small part of the result
print(result[:5, :5])  # Print top-left 5x5 submatrix

