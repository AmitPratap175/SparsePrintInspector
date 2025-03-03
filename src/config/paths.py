from pathlib import Path

class PathConfig:
    def __init__(self):
        self.root = Path(__file__).parents[2]
        self.data = self.root / "data"
        self.checkpoints = self.root / "checkpoints"
        self.outputs = self.root / "outputs"
        
    @property
    def train_good(self):
        return self.data / "train" / "good"
    
    @property
    def test_data(self):
        return self.data / "test"
    
    @property
    def anomaly_maps(self):
        return self.outputs / "anomaly_maps"
    
    def get_checkpoint(self, name):
        return self.checkpoints / name

# Singleton instance
paths = PathConfig()