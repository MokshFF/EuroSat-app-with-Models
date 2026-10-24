import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import config
import dataset
import models
import utils

def main():
    parser = argparse.ArgumentParser(description="Train EuroSAT Land Classifier")
    parser.add_argument('--model', type=str, default='A', choices=['A', 'B'],
                        help="Model to train: 'A' (EfficientNet-B3) or 'B' (ResNet-50)")
    parser.add_argument('--epochs', type=int, default=None,
                        help="Override default number of epochs")
    parser.add_argument('--lr', type=float, default=None,
                        help="Override default learning rate")
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help="Batch size for training")
    args = parser.parse_args()

    # Determine configuration based on selection
    if args.model == 'A':
        model_name = "Model A (EfficientNet-B3 - Production Winner)"
        img_size = config.IMG_SIZE_A
        epochs = args.epochs if args.epochs is not None else config.EPOCHS_A
        lr = args.lr if args.lr is not None else config.LR_A
        model = models.EfficientNetB3Classifier(num_classes=config.NUM_CLASSES)
        checkpoint_name = "model_A_efficientnet_b3_optical.pth"
    else:
        model_name = "Model B (ResNet-50 - Challenger Baseline)"
        img_size = config.IMG_SIZE_B
        epochs = args.epochs if args.epochs is not None else config.EPOCHS_B
        lr = args.lr if args.lr is not None else config.LR_B
        model = models.ResNet50Classifier(num_classes=config.NUM_CLASSES)
        checkpoint_name = "model_B_resnet50_optical.pth"

    model = model.to(config.DEVICE)
    print(f"Starting training for {model_name}...")
    print(f"   Device: {config.DEVICE}")
    print(f"   Epochs: {epochs} | Learning Rate: {lr} | Batch Size: {args.batch_size}")
    
    # Load and split data
    train_data, val_data, test_data = dataset.load_and_split_data()
    train_loader, val_loader, _ = dataset.get_dataloaders(
        train_data, val_data, test_data, img_size, batch_size=args.batch_size
    )

    # TensorBoard setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(config.LOG_DIR, f"{model_name.replace(' ', '_')}_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging active. Run: tensorboard --logdir={config.LOG_DIR}")

    # Loss function (CrossEntropy with Label Smoothing) and Optimizer (AdamW)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    
    scheduler = utils.get_lr_scheduler(optimizer, config.WARMUP_EPOCHS, epochs)
    stopper = utils.EarlyStopping(patience=config.PATIENCE)

    for epoch in range(1, epochs + 1):
        # ── TRAINING PHASE ────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(config.DEVICE), lbls.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            
            # Clip gradient norm to 1.0 to prevent gradient explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(dim=1) == lbls).sum().item()
            train_total += imgs.size(0)
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100

        # ── VALIDATION PHASE ──────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(config.DEVICE), lbls.to(config.DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                
                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(dim=1) == lbls).sum().item()
                val_total += imgs.size(0)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/Val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/Val', epoch_val_acc, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs:2d} | "
                  f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:5.2f}% | "
                  f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:5.2f}% | "
                  f"LR: {current_lr:.2e}")

        # Check early stopping (saves the best model weights inside helper)
        if stopper.step(epoch_val_loss, model):
            print(f"Early stopping triggered at epoch {epoch}. Restoring best weights.")
            break

    # Save final best model to configuration models directory
    save_path = os.path.join(config.MODEL_DIR, checkpoint_name)
    torch.save(stopper.best_weights if stopper.best_weights is not None else model.state_dict(), save_path)
    print(f"Best weights successfully saved to '{save_path}'")
    writer.close()

if __name__ == '__main__':
    main()
