<strong>Edit: </strong> Use 0.0001 learning rate and after 10 epochs you can have accuracy upto 99.5% accuracy
# Distracted Driver Detection

**Domain**: Computer Vision & Artificial Neural Networks (ANN)

## Project Summary
This project aims to enhance road safety by automating the monitoring and detection of distracted driver behaviors using a Convolutional Neural Network (CNN) model. The solution leverages computer vision techniques to classify different types of driver distractions based on real-time image data.

## Problem Statement
Manual observation and reporting of driver behavior are inefficient and lack real-time capabilities. Current systems often depend on facial expressions and suffer from challenges related to data diversity. This project addresses these issues by proposing an ANN-based model that automates the classification of driver behaviors, ensuring more effective and timely monitoring.

## Proposed Solution
The solution involves training a CNN model on a diverse dataset of driver images. The model learns to identify and classify different types of distracted driving behaviors. The key stages of the project include dataset loading, normalization, model implementation, training, testing, prediction, and deployment.

### Detailed Description

**1. Dataset Loading**
- The dataset consists of images labeled with different classes of driver behavior. The images are loaded and preprocessed to ensure consistency in size and format.

**2. Data Normalization**
- Images are normalized to improve the efficiency and performance of the model during training. Normalization involves scaling the pixel values to a range of 0 to 1.

**3. Model Implementation**
- The core of the project is the implementation of a Vanilla CNN model. The architecture comprises several types of layers, each contributing to the model's ability to learn and generalize from the data.

**Model Architecture**
- **Convolutional 2D Layers**: These layers extract local features from the images, capturing patterns such as edges, textures, and shapes.
- **Max Pooling Layers**: These layers reduce the spatial dimensions by downsampling, leading to a reduced computational load while preserving the dominant features.
- **Flatten Layer**: Converts the 2D feature maps into a 1D vector.
- **Dense Layers**: Fully connected layers that apply weights and biases to the feature vector, using an activation function to make final class predictions.
- **Dropout Layers**: These layers randomly set a fraction of input units to zero during training to prevent overfitting.

**4. Model Training**
- The model is trained using gradient descent optimizers to minimize the loss function. The training process involves feeding the normalized images into the model, adjusting weights, and iterating until the model achieves satisfactory performance on the validation set.

**5. Model Testing**
- The trained model is evaluated on a separate test set to measure its accuracy and generalization capability. Performance metrics such as accuracy, precision, recall, and f1-score are used to assess the model.

**6. Prediction and Deployment**
- Once validated, the model is deployed for real-time classification of driver behaviors. The deployment involves integrating the model into a system that can process live video feeds or image uploads and provide predictions.

## Project Deliverables

**Repository Contents**
- **Proposal**: A detailed project proposal outlining the objectives, methodology, and expected outcomes.
- **Presentation**: Slides summarizing the project, problem statement, proposed solution, model architecture, results, and future work.
- **Project Frontend**: A streamlit based frontend where we can upload files and photo is displayed & then it is classified.
- **Report**: A comprehensive report documenting the entire project lifecycle, including data preprocessing, model training, evaluation, and deployment.
  

## Conclusion
This project successfully demonstrates the use of a Vanilla CNN model for the classification of distracted driver behaviors. The automated system enhances road safety by providing real-time monitoring and effective classification of various distractions. Future work may include incorporating night vision capability, facial recognition, and integration with dynamic camera detection systems.

**Future Work**
- **Night Vision Capability**: Enhancing the model to work effectively under low-light conditions.
- **Facial Recognition**: Integrating facial recognition to improve accuracy in identifying specific distractions.
- **Dynamic Camera Detection**: Adapting the system to work with moving cameras for more robust monitoring.

By leveraging the power of computer vision and ANN, this project sets a foundation for advanced driver monitoring systems that can significantly reduce road accidents caused by distracted driving.
