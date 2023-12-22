import torch
import torch.nn as nn

def test(model, test_loader64, device):
    # Test loop
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in test_loader64:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Update progress bar with stats
    print(f'Test Loss: {test_loss:.4f}', f'Accuracy: {100 * correct / total:.2f}%')