Federated Learning for Privacy-Preserving AI
## Project Overview
This project demonstrates **Federated Learning (FL)** using a Convolutional Neural Network (CNN) on the CIFAR-10 dataset.

Instead of sending raw data to a central server, multiple clients train the model locally and only share model weights. This helps in preserving data privacy.
The project also compares:
- Federated Learning (FL)
- Centralized Learning (CL)

## Objectives
- Implement Federated Learning from scratch
- Simulate Non-IID client data distribution
- Compare performance with centralized training
- Visualize accuracy differences
 
## Model Architecture
A CNN model is used with:
- 3 Convolution layers
- Batch Normalization
- Max Pooling
- Dropout (for regularization)
- Fully Connected layers
   
## Dataset
- CIFAR-10 dataset
- 60,000 images (32x32 color images)
- 10 classes:
  - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Technologies Used
- Python
- PyTorch
- NumPy
- Matplotlib

## Working Process

🔹Federated Learning
1. Split dataset into multiple clients (Non-IID)
2. Each client trains locally
3. Send weights to central server
4. Apply Federated Averaging
5. Update global model

🔹 Centralized Learning
- Train entire dataset in one place
- Compare results with FL
