"""
train.py
Trains the AisleEye EfficientNet-B0 model on the tray-table dataset.
"""

import torch
from torch import nn
import torchvision
from tqdm.auto import tqdm
from timeit import default_timer as timer

import data_setup, model_builder

def main():
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Hyperparameters
    NUM_EPOCHS = 10
    BATCH_SIZE = 16 # was 32 — better for Mac CPU
    LEARNING_RATE = 0.001

    # Directories
    train_dir = "data/train"
    test_dir = "data/test"

    # Get the transforms from the pre-trained weights
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    auto_transforms = weights.transforms()

    # Create DataLoaders (num_workers=0 is important on Mac)
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=auto_transforms,
        batch_size=BATCH_SIZE,
        num_workers=0          # ← critical fix for macOS
    )

    print(f"Class names: {class_names}")

    # Create model
    model, _ = model_builder.create_aisleye_model(num_classes=len(class_names))
    model = model.to(device)

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop helpers
    def train_step(model, dataloader, loss_fn, optimizer, device):
        model.train()
        train_loss, train_acc = 0, 0

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        train_loss /= len(dataloader)
        train_acc /= len(dataloader)
        return train_loss, train_acc

    def test_step(model, dataloader, loss_fn, device):
        model.eval()
        test_loss, test_acc = 0, 0

        with torch.inference_mode():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)

                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                test_loss += loss.item()

                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                test_acc += (y_pred_class == y).sum().item() / len(y_pred)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        return test_loss, test_acc

    def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device):
        results = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }

        best_test_acc = 0.0

        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
            test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

            print(
                f"Epoch: {epoch+1}/{epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            # Save the best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), "models/aisleye_efficientnet_b0.pth")
                print(f"→ Saved best model (test_acc: {test_acc:.4f})")

        return results

    # Start training
    start_time = timer()

    results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device
    )

    end_time = timer()
    print(f"\nTotal training time: {end_time - start_time:.2f} seconds")
    print(f"Best model saved to: models/aisleye_efficientnet_b0.pth")


if __name__ == "__main__":
    main()