import datetime

import matplotlib.pyplot as plt
import numpy as np

import student_profile
import torch


def visualize_loss(train_losses, test_losses, title):
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d")
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(test_losses, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"{title} - Train and Test Losses\n {student_profile.andrew_id}: {timestamp_str}"
    )
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()


def visualize_images(model, testset, device, title):
    # TODO: [Optional] feel free to pass in more arguments for the function if needed
    model.eval()
    model = model.to(device)
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d")

    # Visualize the final results
    plt.figure(figsize=(12, 6))
    random_inds = np.random.choice(len(testset), 10)

    for i, image_idx in enumerate(random_inds):
        test_image, test_label = testset[image_idx]
        test_image = test_image.unsqueeze(0)

        plt.subplot(2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        # Move the test_image tensor back to CPU for visualization
        test_image_np = test_image.cpu().squeeze().numpy()
        # TODO: Reverse transforms and show the image using plt.imshow
        # Hint1: You need to handle grayscale and RGB images differently
        # Hint2: You may need to unnormalize the image before displaying
        if test_image_np.ndim == 2:
            img_disp = test_image_np *0.5 + 0.5
            img_disp = np.clip(img_disp,0,1)
            plt.imshow(img_disp, cmap="gray")
        else:
            img_disp = np.transpose(test_image_np,(1,2,0))
            img_disp = img_disp*0.5+0.5
            img_disp = np.clip(img_disp,0,1)
            plt.imshow(img_disp)

        with torch.no_grad():
            # Move the test_image tensor to the same device as the model (cuda or cpu)
            test_image = test_image.to(device)
            output = model(test_image)
            predicted_label = torch.argmax(output).item()

        plt.xlabel(f"True: {test_label}\nPred: {predicted_label}")

    plt.suptitle(
        f"Final Results - {title}\n {student_profile.andrew_id}: {timestamp_str}",
        fontsize=16,
    )
    plt.show()
