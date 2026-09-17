import numpy as np

class Module:

    def forward(self, x):
        return x

    def backward(self, grad):
        return grad

    def parameters(self):
        return []

    def __call__(self, *args):
        return self.forward(*args)

class Parameter:

    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)

class Conv2D(Module):

    def __init__(self, in_channels, out_channels, kernel_size):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # weight
        self.weights = Parameter(np.random.randn(out_channels, in_channels, kernel_size, kernel_size))

        # bais
        self.bias = Parameter(np.zeros(out_channels))
        
    def forward(self, x):

        # save the original
        self.input = x

        batch_size, channels, height, width = x.shape

        output_height = height - self.kernel_size + 1
        output_width = width - self.kernel_size + 1

        output = np.zeros((batch_size, self.out_channels, output_height, output_width))

        # each image
        for batch in range(batch_size): # 10 images

            # each out_channel
            for out_channel in range(self.out_channels): # 8 out channels

                # height
                for row in range(output_height):

                    # width    
                    for column in range(output_width):

                        sum_of_inchannels = 0

                        # each in_channel
                        for in_channel in range(self.in_channels): # 3 RGB in channels

                            # convolve 
                            sum_of_inchannels += np.sum(x[batch, in_channel, row: row + self.kernel_size, column : column + self.kernel_size] * self.weights.data[out_channel, in_channel])

                        # after combining channel's. add a bais
                        sum_of_inchannels += self.bias.data[out_channel]

                        # Store the it output
                        output[batch, out_channel, row, column] = sum_of_inchannels

                    

        return output


    def backward(self, grad):

        # gradient for weights
        dW = np.zeros_like(self.weights.data)

        # gradient for bias
        dB = np.zeros_like(self.bias.data)

        # gradient to return to the previous value
        dX = np.zeros_like(self.input)

        batch_size, out_channels, height, width = grad.shape

        for batch in range(batch_size):

            for out_channel in range(out_channels):

                for row in range(height):

                    for column in range(width):

                        # conside this as dL/dy
                        g = grad[batch, out_channel, row, column]

                        # Bias update 
                        # dy/db
                        dB[out_channel] += g

                        for in_channel in range(self.in_channels):

                            x = self.input[batch, in_channel, row:row + self.kernel_size, column:column + self.kernel_size]

                            # Weight update
                            # dy/dw
                            dW[out_channel, in_channel] += g * x

                            # dy/dx
                            # Accumulating the gradient for pixel 5 four times. solves flipping the kernel check the image below
                            dX[batch, in_channel, row:row + self.kernel_size, column:column + self.kernel_size] += g * self.weights.data[out_channel, in_channel]

        self.weights.grad += dW
        self.bias.grad += dB

        return dX

    def parameters(self):
        return [self.weights, self.bias]

class MSELoss(Module):
    def forward(self, predict, target):
        self.predict = predict
        self.target = target
        self.n = target.size

        loss = np.mean((predict - target) ** 2)

        return loss

    def backward(self):
        # dL/dy_cap
        return (2/self.n) * (self.predict - self.target)
    

class Activation_ReLU(Module):

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):

        return grad * (self.x > 0)

class MaxPool2D(Module):

    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):

        # Save input for backward()
        self.input = x

        batch_size, channels, height, width = x.shape

        # Calculate output size
        out_height = ((height - self.pool_size) // self.stride) + 1
        out_width  = ((width  - self.pool_size) // self.stride) + 1

        # Output after pooling
        self.output = np.zeros((batch_size, channels, out_height, out_width))

        # coordinates of the maximum
        self.max_row = np.zeros((batch_size, channels, out_height, out_width), dtype=int)
        self.max_col = np.zeros((batch_size, channels, out_height, out_width), dtype=int)

        for batch in range(batch_size):
            for channel in range(channels):
                for row in range(0, height - self.pool_size + 1, self.stride):
                    for column in range(0, width - self.pool_size + 1, self.stride):
                        # output r, c
                        out_row = row // self.stride
                        out_column = column // self.stride

                        # window
                        window = x[batch, channel, row: row + self.pool_size, column : column + self.pool_size]

                        # Maximum value becomes output
                        self.output[batch, channel, out_row, out_column] = np.max(window)

                        r, c = np.unravel_index(np.argmax(window), window.shape)
                        
                        self.max_row[batch, channel, out_row, out_column] = row + r
                        self.max_col[batch, channel, out_row, out_column] = column + c


        return self.output

    def backward(self, grad):

        dx = np.zeros_like(self.input)

        batch_size, channels, height, width = self.output.shape

        for batch in range(batch_size):
            for channel in range(channels):
                for row in range(height):
                    for column in range(width):

                        r = self.max_row[batch, channel, row, column]
                        c = self.max_col[batch, channel, row, column]

                        dx[batch, channel, r, c] += grad[batch, channel, row, column]

        return dx


class Flatten(Module):

    def forward(self, x):

        self.input_shape = x.shape

        x = x.reshape(self.input_shape[0], -1)

        return x

    def backward(self, grad):

        return grad.reshape(self.input_shape)

class Linear(Module): 

    def __init__(self, in_features, out_features):

        self.in_features = in_features
        self.out_features = out_features

        # weight
        self.weights = Parameter(np.random.randn(self.in_features, self.out_features))

        # bais
        self.bias = Parameter(np.zeros(self.out_features))
        

    def forward(self, x):

        self.input = x

        self.output = (x @ self.weights.data) + self.bias.data

        return self.output

    def backward(self, grad):
        # dL/dy = grad
        
        # dy/dx
        self.dx = np.zeros_like(self.input)
        self.dx += grad @ (self.weights.data.T)

        # dy/dw
        self.weights.grad += (self.input.T @ grad)

        # dy/db
        self.bias.grad += np.sum(grad, axis=0)

        return self.dx

    def parameters(self):
        return [self.weights, self.bias]


class Sequential(Module):

    def __init__(self, *args):
        self.layers = list(args)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x

    def backward(self, grad):

        for layer in self.layers[::-1]:
            grad = layer.backward(grad)

        return grad

    def parameters(self):

        p = []

        for layer in self.layers:
            p.extend(layer.parameters())

        return p

class Softmax(Module):

    def forward(self, x):

        self.input = x

        x = x - np.max(x, axis = 1, keepdims = True)

        x = np.exp(x)

        total = np.sum(x, axis = 1, keepdims = True)

        self.output = x / total

        return self.output

    def backward(self, grad):

        batch_size, n = self.output.shape

        dx = np.zeros_like(grad)

        for b in range(batch_size):

            jacobian = np.zeros((n, n))

            for i in range(n):
                for j in range(n):

                    if i == j:
                        jacobian[i, j] = self.output[b, i] * (1 - self.output[b, i])

                    else:
                        jacobian[i, j] = -self.output[b, i] * self.output[b, j]

            dx[b] = grad[b] @ jacobian

        return dx

class CrossEntropyLoss(Module):

    def forward(self, predicted, target):

        self.predicted = predicted
        self.target = target

        self.loss = -np.mean(np.sum(target * np.log(predicted + 1e-12), axis=1))

        return self.loss

    def backward(self):

        batch_size = self.target.shape[0]
        return -(self.target / (self.predicted + 1e-12)) / batch_size

    
class SGD:

    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for parameter in self.parameters:
            parameter.data -= self.lr * parameter.grad

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad.fill(0)