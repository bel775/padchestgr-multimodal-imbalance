import torch
from evaluation import evaluate_model
import torch.nn as nn
from sklearn.metrics import f1_score
import copy

import matplotlib.pyplot as plt
import numpy as np

def save_loss_plot(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()


def train_model(model, optimizer,criterion, training_mode, train_loader, val_loader, freezeText, save_path, num_epochs=300, early_Stopping = 60):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',       
        factor=0.65,      
        patience=15,       
        min_lr=1e-6       
    )

    train_losses = []
    val_losses = []
    best_f1 = 0.0
    best_model = None

    patience_counter = 0

    print("Start the loop Training ...")
    test = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_preds = []
        epoch_labels = []

        for batch in train_loader:

            optimizer.zero_grad()
            if training_mode == 0:
                images = batch['image_feat'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images)

            elif training_mode == 1:
                if freezeText:
                    sentences = batch['sentence'].to(device)
                else:
                    sentences = batch['sentence']

                labels = batch['label'].to(device)

                outputs = model(sentences)

            elif training_mode == 2:
                if freezeText:
                    sentences = batch['sentence'].to(device)
                else:
                    sentences = batch['sentence']

                images = batch['image_feat'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images, sentences)  
            else:
                raise ValueError(f"Unknown mode: {training_mode}")

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            epoch_labels.append(labels.cpu())
            epoch_preds.append(preds.cpu())

        train_loss = running_loss / len(train_loader)
        epoch_preds = torch.cat(epoch_preds).numpy()
        epoch_labels = torch.cat(epoch_labels).numpy()

        training_f1 = f1_score(epoch_labels, epoch_preds, average="macro", zero_division=0)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f} - Training F1: {training_f1:.4f}")


        # ---- Validation ----
        val_loss, f1 = evaluate_model(model, val_loader,training_mode,freezeText, criterion = criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if f1 > best_f1:
            best_f1 = f1
            best_model = copy.deepcopy(model)
            patience_counter = 0
        else: 
            patience_counter += 1

        if patience_counter >= early_Stopping:
            print(f"Stopping early -------------- ")
            break  # Stop training early

    save_loss_plot(train_losses, val_losses, save_path)
    return best_model