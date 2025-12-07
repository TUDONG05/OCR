import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5, 6, 7, 8,9]
# loss = [10.0503, 8.4629, 7.9895, 7.5291, 7.2383, 7.0745, 6.9181, 6.5203]
loss = [ 7.6279,
7.3075,
7.1030,
6.9503,
6.7424,
6.5793,
6.3782,
6.2075,
6.0887]

plt.plot(epochs, loss, marker='o')
plt.title("Fine tune 9 epoch với learning rate = 1e-5")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
