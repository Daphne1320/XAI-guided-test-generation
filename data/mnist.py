import tensorflow as tf
from keras.utils import to_categorical


def mnist_data(reshape=True, categorical=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if reshape:
        # Reshape data
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    if categorical:
        # Convert class vectors to binary class matrices
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


"""
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Lambda(lambda x: x.view(28, 28, 1))  # Reshape the tensor
])
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_dataloader = DataLoader(mnist_dataset, batch_size=32, shuffle=True)
mnist_dataloaderx = OnlyX(mnist_dataloader)
my_mnist = ALDataTF(pool_data=mnist_dataloaderx, expert_data=mnist_dataloader, test_data=(x_test, y_test),
                    batch_size=4)
"""
