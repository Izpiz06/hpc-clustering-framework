import time
import numpy as np
from sklearn.datasets import make_blobs

# Notice how clean this import is because of your __init__.py!
from src.cpu_backend import KMeansCPU 

def run_cpu_benchmark():
    print("🚀 Starting CPU K-Means Benchmark...")
    print("-" * 40)

    # 1. Define dataset parameters (Large enough to make the CPU sweat)
    N_SAMPLES = 10_000
    N_FEATURES = 50
    N_CLUSTERS = 5
    
    print(f"📊 Generating dataset: {N_SAMPLES:,} samples, {N_FEATURES} features...")
    
    # Generate synthetic data
    X, _ = make_blobs(
        n_samples=N_SAMPLES, 
        centers=N_CLUSTERS, 
        n_features=N_FEATURES, 
        random_state=42
    )
    
    print("✅ Dataset generated.\n")
    
    # 2. Initialize the K-Means Model
    kmeans = KMeansCPU(n_clusters=N_CLUSTERS, max_iter=300, random_state=42)
    
    # 3. Benchmark the training process
    print("⏳ Running K-Means (NumPy/CPU Backend)...")
    
    start_time = time.perf_counter()
    kmeans.fit(X)
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    
    # 4. Report Results
    print("-" * 40)
    print("🎯 BENCHMARK RESULTS")
    print("-" * 40)
    print(f"Backend      : CPU (NumPy)")
    print(f"Iterations   : {kmeans.n_iters_}")
    print(f"Time Taken   : {execution_time:.4f} seconds")
    print("-" * 40)
    print("To beat this score, contributors must implement a GPU backend!")

if __name__ == "__main__":
    run_cpu_benchmark()