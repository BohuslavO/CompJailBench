from dataset import BenchmarkDataset
from benchmark import CompJailBench
from logger import TrajectoryLogger

dataset = BenchmarkDataset()
benchmark = CompJailBench()
logger = TrajectoryLogger()

for sample in dataset:
    print(f"\nRunning Sample {sample['id']}")

    result = benchmark.run(sample)

    logger.save(sample, result)

print("\Benchmark Complete.")