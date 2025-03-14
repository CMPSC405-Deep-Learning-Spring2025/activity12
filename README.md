# Activity 12
## Due: 10am on March 13, 2025

## Objectives
- Understand an architecture of the fully functional CNN
- Learn how to define and implement a CNN model (LeNet-5) in PyTorch 
- Observe the role of activation functions (Sigmoid) and pooling layers in feature transformation and dimensionality reduction

## Tasks
1. Replace average pooling with max-pooling.
   - `nn.AvgPool2d(kernel_size=2, stride=2)  →  nn.MaxPool2d(kernel_size=2, stride=2)`
2. Replace the softmax layer with ReLU.
   - `nn.Sigmoid()  →  nn.ReLU()`
3. Add an extra convolutional layer.
4. Experiment with Kernel Size.
   - `nn.Conv2d(1, 6, kernel_size=3, padding=1)`
5. Modify Stride (default is 1).
   - `nn.Conv2d(1, 6, kernel_size=5, stride=2, padding=2)`

## Observations
TODO: for each task above describe your observations of the output. It might be helpful to display the image outputs from the first or second layer.
1. TODO
2. TODO
3. TODO
4. TODO
5. TODO
