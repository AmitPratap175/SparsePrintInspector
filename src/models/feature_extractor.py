import torch
import torchvision

class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.model = self._build_model(checkpoint_path)
        
    def _build_model(self, checkpoint_path):
        model = torchvision.models.resnet50(pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        # Modify final layer
        model.fc = torch.nn.Linear(model.fc.in_features, 3)
        model.load_state_dict(state_dict)
        
        # Extract feature layers
        return torch.nn.Sequential(*list(model.children())[:-2])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.model(x)
            return features.mean(dim=(2, 3))  # Global average pooling