# Image Classification with ResNet-50
This project demonstrates how to train a ResNet-50 model for image classification using PyTorch. The script includes data loading, model training, evaluation, and checkpoint saving.

## Dependencies
- PyTorch
- torchvision
- numpy
- natsort

# Code Overview 
# train.py
## Data Loading and Splitting
The load_split_test function loads images from a specified directory and splits them into training and testing sets. The images are transformed into tensors.

## Model Evaluation
The eval_on_test_set function evaluates the model's performance on the test set and calculates the accuracy.

## Saving Checkpoints
The save_checkpoint function saves the model's state, including the epoch number, model parameters, optimizer state, and best accuracy achieved.

## Training the Model
The train_nn function trains the model for a specified number of epochs, evaluating its performance after each epoch and saving the best model.

## Main Function
The main function sets up the dataset path, loads the data, defines the device (GPU or CPU), initializes the ResNet-50 model, and modifies the final layer to match the number of classes. It then sets the loss function and optimizer, and trains the model.

The script will train the model, evaluate it, and save the best-performing model as best_model.pth. 

# prediction.py
## Device Configuration
The script checks if a CUDA-compatible GPU is available and sets the device accordingly.
## Model Loading
The trained ResNet-50 model is loaded from a file (best_model.pth).
## Image Transformations
Images are transformed into tensors using the specified transformations.
## Directory and File Sorting
The script reads image file names from the specified directory and sorts them in natural order.
## Prediction and Output
The script iterates over the sorted image files, applies the trained model to each image, and writes the predictions to a text file (prediction.txt).
The script will generate predictions for 10,000 images in the test directory and save the results to prediction.txt.

# Note
This repo is private, please let me know if you would like to see it.
