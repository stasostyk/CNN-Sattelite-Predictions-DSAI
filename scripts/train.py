import torch
import torch.nn as nn
import torch.optim as optim


def train(model, train_loader, val_loader, device, best_model_path, num_epochs=20, lr=0.001):
    # Define loss function, optimizer and training epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Initialize lists to store losses and accuracies
    training_losses = []
    validation_losses = []
    validation_accuracies = []

    # We keep track of the best validation accuracy and save the best model
    best_val_accuracy = 0.0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        # Store average training loss for the epoch
        training_losses.append(training_loss / len(train_loader))

        # Validation loop
        model.eval()
        validation_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Store average training loss and accuracy for the epoch
        validation_losses.append(validation_loss / len(val_loader))
        val_accuracy = 100 * correct / total
        validation_accuracies.append(val_accuracy)

        # save best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch + 1}: Improved validation accuracy to {best_val_accuracy:.2f}%. Model saved.")

        print(f'Epoch {epoch + 1}/{num_epochs}', f'Train Loss: {training_losses[-1]:.4f}, '
                                                 f'Validation Loss: {validation_losses[-1]:.4f}, '
                                                 f'Accuracy: {validation_accuracies[-1]:.2f}%')

    return training_losses, validation_losses, validation_accuracies
