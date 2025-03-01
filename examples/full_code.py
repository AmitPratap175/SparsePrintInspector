from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
import torch
import torchvision
from torchvision.transforms import transforms
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score
)
from concurrent.futures import ThreadPoolExecutor

# Global Variables
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Model Definition
class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self, checkpoint_path):
        """Extracts feature maps from a sparsified ResNet model."""
        super().__init__()
        model = torchvision.models.resnet50()
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)

        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.fc = torch.nn.Linear(model.fc.in_features, 3)
        model.load_state_dict(state_dict)

        self.model = torch.nn.Sequential(*(list(model.children())[:-2]))  # Remove FC and AvgPool layers

        # Disable gradient computations
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_tensor):
        """Forward pass to extract feature maps."""
        with torch.no_grad():
            feature_map = self.model(input_tensor)
            feature_map = feature_map.mean(dim=(2, 3))  # Global Average Pooling
        return feature_map

# Utility Functions
def load_image(file_path):
    """Load and preprocess an image."""
    return TRANSFORM(Image.open(file_path).convert('RGB')).unsqueeze(0)

def build_memory_bank(folder_path, backbone, sampling_fraction=0.1):
    """Build a memory bank of features from training images."""
    all_features = []

    def process_image(file):
        data = load_image(file)
        return backbone(data).cpu()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, file) for file in Path(folder_path).iterdir()]
        for future in tqdm(futures, leave=False, desc="Building Memory Bank"):
            all_features.append(future.result())

    memory_bank = torch.cat(all_features, dim=0)
    selected_indices = np.random.choice(len(memory_bank), size=int(len(memory_bank) * sampling_fraction), replace=False)
    return memory_bank[selected_indices]

def evaluate_images(folder_path, backbone, memory_bank):
    """Evaluate images for anomaly detection."""
    y_score, y_true = [], []

    def process_image(file, true_label):
        test_image = load_image(file)
        features = backbone(test_image)
        distances = torch.cdist(features, memory_bank, p=2.0)
        dist_score, _ = torch.min(distances, dim=1)
        return torch.max(dist_score).item(), true_label

    with ThreadPoolExecutor() as executor:
        for class_folder in Path(folder_path).iterdir():
            if class_folder.is_dir():
                true_label = 0 if class_folder.name == 'good' else 1
                futures = [
                    executor.submit(process_image, file, true_label)
                    for file in class_folder.iterdir() if file.is_file()
                ]
                for future in tqdm(futures, leave=False, desc=f"Evaluating {class_folder.name}"):
                    score, label = future.result()
                    y_score.append(score)
                    y_true.append(label)
    
    return y_score, y_true

def calculate_threshold(y_scores, std_multiplier=2):
    """Calculate anomaly threshold based on training data scores."""
    return np.mean(y_scores) + std_multiplier * np.std(y_scores) if y_scores else 0

def visualize_results(y_score, y_true, best_threshold):
    """Visualize histogram and metrics for anomaly detection."""
    plt.hist(y_score, bins=50, alpha=0.7, label="Anomaly Scores")
    plt.axvline(best_threshold, color='red', linestyle='--', label='Threshold')
    plt.legend()
    plt.title("Anomaly Scores Distribution")
    plt.show()

    if y_true:
        auc_roc = roc_auc_score(y_true, y_score)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_roc:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.title('ROC Curve')
        plt.show()

        best_f1 = np.argmax([f1_score(y_true, y_score >= t) for t in thresholds])
        confusion = confusion_matrix(y_true, np.array(y_score) >= thresholds[best_f1])
        ConfusionMatrixDisplay(confusion).plot()
        plt.show()

def generate_and_save_anomaly_map(image_path, backbone, memory_bank, output_dir):
    """Generate and save anomaly map for a given image."""
    # try:
    test_image = TRANSFORM(Image.open(image_path).convert('RGB')).unsqueeze(0)
    features = backbone(test_image)
    distances = torch.cdist(features, memory_bank, p=2.0)
    dist_score, _ = torch.min(distances, dim=1)
    segm_map = dist_score.view(1, 1, 7, 7)
    segm_map = torch.nn.functional.interpolate(segm_map, size=(224, 224), mode='bilinear').squeeze()

    plt.imshow(segm_map, cmap='jet')
    plt.axis('off')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_anomaly_map.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Anomaly map saved to {output_path}")
    # except Exception as e:
    #     print(f"Error processing {image_path}: {e}")

# Main Execution
if __name__ == "__main__":
    sparse_checkpoint_path = "../checkpoints/resnet-50-sparse-beans.pth"
    backbone = ResNetFeatureExtractor(sparse_checkpoint_path)

    train_folder = "../data/carpet/train"
    memory_bank = build_memory_bank(f"{train_folder}/good", backbone)

    y_scores_ok, _ = evaluate_images(train_folder, backbone, memory_bank)
    best_threshold = calculate_threshold(y_scores_ok)
    visualize_results(y_scores_ok, None, best_threshold)

    test_folder = "../data/carpet/test"
    y_scores, y_true = evaluate_images(test_folder, backbone, memory_bank)
    visualize_results(y_scores, y_true, best_threshold)

    output_dir = "../output/carpet/output"
    for class_folder in Path(test_folder).iterdir():
        if class_folder.is_dir():
            for file in class_folder.iterdir():
                if file.is_file():
                    generate_and_save_anomaly_map(file, backbone, memory_bank, output_dir)