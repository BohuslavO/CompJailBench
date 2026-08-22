import json
import os
from datetime import datetime


class TrajectoryLogger:

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save(self, sample, result):
        """result is the dict returned by CompJailBench.run(), i.e.
        {"execution_trace": ..., "monitor": ...} — saved as-is rather than
        re-wrapped, so the on-disk shape matches what benchmark.py produced.
        """

        filename = f"{sample['id']}.json"
        path = os.path.join(self.output_dir, filename)

        log = {
            "timestamp": datetime.now().isoformat(),
            "sample": sample,
            "execution_trace": result["execution_trace"],
            "monitor": result["monitor"],
        }

        with open(path, "w") as f:
            json.dump(log, f, indent=4)

        return path