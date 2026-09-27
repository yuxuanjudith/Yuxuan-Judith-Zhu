import torch


def evaluate_loss(model, criterion, dataloader, device):
    # TODO: set model to eval mode
    model.eval()  # disables dropout/batchnorm updates during evaluation

    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            # TODO: load inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)
            # TODO: Call model and get outputs
            outputs = model(inputs)
            # TODO: Calculate the loss using loss function
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss


def evaluate_accuracy(model, dataloader, device):
    # TODO: set model to eval mode
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            # TODO: load inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)
            # TODO: Call model and get outputs
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (correct / total) * 100
    return accuracy


def train(model, optimizer, criterion, trainloader, testloader, epochs, device, scheduler=None):
    """
    Part 1.a: complete the training loop
    """
    train_losses = []  # For recording train losses
    test_losses = []  # For recording test losses

    # TODO: move model to device here
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        # TODO: Set the model to train mode
        model.train()

        for inputs, labels in trainloader:
            optimizer.zero_grad()

            # TODO: load inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)
            # TODO: Call model and get outputs
            outputs = model(inputs)
            # TODO: Calculate the loss using loss function
            loss = criterion(outputs, labels)
            # TODO: call backward on loss
            loss.backward()
            # TODO: Add optimizer step
            optimizer.step()

            running_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss}")

        test_loss = evaluate_loss(model, criterion, testloader, device)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}/{epochs} - Test Loss: {test_loss}")

        test_accuracy = evaluate_accuracy(model, testloader, device)
        print(f"Epoch {epoch+1}/{epochs} - Test Accuracy: {test_accuracy:.2f}%")

    return train_losses, test_losses