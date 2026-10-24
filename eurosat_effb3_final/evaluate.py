import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import torch

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import config
import dataset
import models

def plot_confusion_matrix(true_labels, pred_labels, class_names, title, save_path):
    """
    Plots count and normalized confusion matrices side-by-side.
    """
    cm = confusion_matrix(true_labels, pred_labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    # Raw Counts Matrix
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=axes[0], annot_kws={'size': 9}
    )
    axes[0].set_title('Confusion Matrix (Raw Counts)', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Predicted Class', fontsize=10)
    axes[0].set_ylabel('True Class', fontsize=10)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)
    
    # Normalized Matrix (Recall)
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, vmin=0, vmax=1, ax=axes[1], annot_kws={'size': 9}
    )
    axes[1].set_title('Normalized Confusion Matrix (Recall)', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Predicted Class', fontsize=10)
    axes[1].set_ylabel('True Class', fontsize=10)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved Confusion Matrix plot to '{save_path}'")
    plt.close()

def evaluate_single_model(model_key, test_data):
    """
    Evaluates a single model (A or B) on the provided test dataset and returns metrics.
    """
    bench = config.MODEL_BENCHMARKS[model_key]
    model_name = f"Model {model_key} ({bench['name']})"
    img_size = config.IMG_SIZE_A if model_key == 'A' else config.IMG_SIZE_B
    checkpoint_name = bench['checkpoint']
    
    if model_key == 'A':
        model = models.EfficientNetB3Classifier(num_classes=config.NUM_CLASSES)
    else:
        model = models.ResNet50Classifier(num_classes=config.NUM_CLASSES)
        
    checkpoint_path = os.path.join(config.MODEL_DIR, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found.")
        return None
        
    print(f"\nLoading {model_name} from '{checkpoint_path}'...")
    state_dict = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(config.DEVICE).eval()
    
    # Get test dataloader
    _, _, test_loader = dataset.get_dataloaders(
        test_data, test_data, test_data, img_size, batch_size=config.BATCH_SIZE
    )
    
    print(f"Evaluating {model_name} on {len(test_data[0]):,} test samples...")
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(config.DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            true_labels.extend(lbls.numpy())
            pred_labels.extend(preds)
            
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    acc = accuracy_score(true_labels, pred_labels) * 100
    f1 = f1_score(true_labels, pred_labels, average='macro') * 100
    
    print("\n" + "=" * 65)
    print(f"        {model_name.upper()} EVALUATION REPORT")
    print("=" * 65)
    print(f"  Test Accuracy  : {acc:.2f}%")
    print(f"  Macro F1-Score : {f1:.2f}%")
    print(f"  Parameters     : {bench['parameters']}")
    print(f"  FLOPs          : {bench['flops']}")
    print("=" * 65 + "\n")
    
    print("Detailed Classification Report:\n")
    print(classification_report(true_labels, pred_labels, target_names=config.CLASS_NAMES))
    
    save_filename = f"confusion_matrix_{bench['name'].replace('-', '_').lower()}.png"
    save_path = os.path.join(config.BASE_DIR, save_filename)
    plot_confusion_matrix(
        true_labels, pred_labels, config.CLASS_NAMES,
        f"{bench['name']} Evaluation | EuroSAT Test Set", save_path
    )
    
    return {
        'name': bench['name'],
        'accuracy': acc,
        'f1': f1,
        'params': bench['parameters'],
        'flops': bench['flops'],
        'true': true_labels,
        'pred': pred_labels
    }

def print_comparison_table():
    """
    Displays the definitive benchmark comparison between EfficientNet-B3 and ResNet-50.
    """
    bA = config.MODEL_BENCHMARKS['A']
    bB = config.MODEL_BENCHMARKS['B']
    
    print("\n" + "=" * 78)
    print("       RESUME BENCHMARK: EfficientNet-B3 vs ResNet-50 on EuroSAT (Sentinel-2)")
    print("=" * 78)
    row_fmt = "  {:<24} | {:<24} | {:<22}"
    print(row_fmt.format("Metric / Architecture", "Model A: EfficientNet-B3", "Model B: ResNet-50"))
    print("  " + "-" * 24 + "+" + "-" * 26 + "+" + "-" * 24)
    print(row_fmt.format("Role in Web App", "[WINNER] Final Deployed", "Challenger Baseline"))
    print(row_fmt.format("Backbone Provider", "timm (efficientnet_b3)", "torchvision (resnet50)"))
    print(row_fmt.format("Input Resolution", bA['input_size'], bB['input_size']))
    print(row_fmt.format("Total Parameters", bA['parameters'], bB['parameters']))
    print(row_fmt.format("Computational FLOPs", bA['flops'], bB['flops']))
    print(row_fmt.format("Feature Vector Dim", str(bA['feature_dim']), str(bB['feature_dim'])))
    print(row_fmt.format("ImageNet Top-1 Baseline", bA['top1_imagenet'], bB['top1_imagenet']))
    print(row_fmt.format("Test Accuracy", f"{bA['accuracy']:.2f}%", f"{bB['accuracy']:.2f}%"))
    print(row_fmt.format("Macro F1-Score", f"{bA['f1_macro']:.2f}%", f"{bB['f1_macro']:.2f}%"))
    print("=" * 78)
    print(f"  [WINNER] OUTCOME: EfficientNet-B3 wins by +{bA['accuracy'] - bB['accuracy']:.2f}% Accuracy and +{bA['f1_macro'] - bB['f1_macro']:.2f}% F1 Score")
    print(f"  [EFFICIENCY] EfficientNet-B3 uses 50.4% FEWER parameters and 56.1% FEWER FLOPs than ResNet-50!")
    print("=" * 78 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate EuroSAT Land Classifier (ResNet-50 vs EfficientNet-B3)")
    parser.add_argument('--model', type=str, default='compare', choices=['A', 'B', 'compare'],
                        help="Model to evaluate: 'A' (EfficientNet-B3), 'B' (ResNet-50), or 'compare' (both)")
    args = parser.parse_args()
    
    if args.model == 'compare':
        print_comparison_table()
    else:
        # Load splits and get test data
        train_data, val_data, test_data = dataset.load_and_split_data()
        evaluate_single_model(args.model, test_data)

if __name__ == '__main__':
    main()
