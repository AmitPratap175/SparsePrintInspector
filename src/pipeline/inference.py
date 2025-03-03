from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import numpy as np
import torch

class InferencePipeline:
    def __init__(self, backbone, transform, paths):
        self.backbone = backbone
        self.transform = transform
        self.paths = paths
        
    def build_memory_bank(self, sampling_fraction: float = 0.1):
        """Construct core set memory bank with parallel processing"""
        files = list(self.paths.train_good.glob("*.*"))
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._process_file, f) for f in files]
            features = [f.result() for f in tqdm(futures, desc="Building Memory Bank")]
            
        memory_bank = torch.cat(features, dim=0)
        indices = np.random.choice(
            len(memory_bank), 
            int(len(memory_bank) * sampling_fraction), 
            False
        )
        return memory_bank[indices]
    
    def _process_file(self, file_path):
        image = self.transform(Image.open(file_path).convert("RGB"))
        return self.backbone(image.unsqueeze(0)).cpu()