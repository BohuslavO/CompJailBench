import json

REQUIRED_FIELDS = ("id", "task")


class BenchmarkDataset:
    def __init__(self, filename="datasets/sample.json"):
        with open(filename, "r") as f:
            self.samples = json.load(f)

        self._validate()

    def _validate(self):
        for i, sample in enumerate(self.samples):
            missing = [field for field in REQUIRED_FIELDS if field not in sample]
            if missing:
                raise ValueError(
                    f"Sample at index {i} is missing required field(s): {missing}"
                )

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]