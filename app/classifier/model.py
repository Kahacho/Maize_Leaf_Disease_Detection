from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import transforms


class Net(nn.Module):
    """ Create a neural net class """

    def __init__(self, num_classes=3):
        # num_classes is 3 because we have 3 classes: Health, MLN, and MSV
        super(Net, self).__init__()

        # Our images are RGB, so input_channels=3.
        # We'll apply 12 filters in the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=12,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        # We'll apply max pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)

        # A second convolution layer takes 12 input channels, and generates 12 outputs
        self.conv2 = nn.Conv2d(in_channels=12,
                               out_channels=12,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        # A third convolutional layers takes 12 inputs and generates 24 outputs
        self.conv3 = nn.Conv2d(in_channels=12,
                               out_channels=24,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        # A drop layer deletes 20% of the features to help prevent overfitting
        self.drop = nn.Dropout(p=0.2)

        # Our 128*128 image tensors will be pooled twice with a kernel size 0f 2.
        # 128/2/2 is 32. So our feature tensors are now 32 * 32, and we have generated 24 of them.
        # We need to flatten these and feed them to a fully-connected layer to map them to the probability of each class
        self.fc = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes)

    def forward(self, x):
        # Use a ReLu activation function after layer 1 (convolution 1 and pool)
        x = F.relu(self.pool(self.conv1(x)))

        # Use a ReLu activation function after layer 2 (convolution 2 and pool)
        x = F.relu(self.pool(self.conv2(x)))

        # Select some features to drop after the 3rd convolution to prevent overfitting
        x = F.relu(self.drop(self.conv3(x)))

        # Only drop the features if this is a training pass.
        # By default, self.training flag is set to True, i.e. modules are in train mode by default
        x = F.dropout(x, training=self.training)

        # Flatten
        x = x.view(-1, 32 * 32 * 24)

        # Feed to fully-connected layer to predict class
        x = self.fc(x)

        # Return log_softmax tensor
        return F.log_softmax(x, dim=1)


@lru_cache(1)
def load_model():
    """
    Load the trained model/classifier
    """

    # Create a new model class and load the saved weights
    model_file = Path(__file__).parent.parent.parent / "models/maize_classifier.pt"
    model = Net()
    model.load_state_dict(torch.load(model_file))

    return model


def predict_image(image) -> str:
    """
    Classify/predict the disease status of maize
    """

    # Load the model/classifier
    classifier = load_model()

    # Switch the classifier to evaluation mode
    classifier.eval()

    # Apply the same transformations as we did for the training images
    transformation = transforms.Compose([
        # Resize to a common 128 * 128 image size
        transforms.Resize(size=[128, 128]),
        # Transform to tensors
        transforms.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Preprocess the image
    image_tensor = transformation(image).float()

    # Add an extra batch dimension since PyTorch treats all inputs as batches
    image_tensor = image_tensor.unsqueeze_(0)

    # Turn the input into a variable
    input_features = Variable(image_tensor)

    # Predict the class of an image
    output = classifier(input_features)
    index = output.data.numpy().argmax()

    disease_status = {0: "HEALTHY", 1: "MLN", 2: "MSV"}

    return disease_status.get(index)
