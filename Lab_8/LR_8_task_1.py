import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
m = 1000
X_data = np.random.rand(m, 1).astype(np.float32)
epsilon = np.random.normal(0, 0.5, (m, 1)).astype(np.float32)
y_data = 2 * X_data + 1 + epsilon

k = tf.Variable(tf.random.normal([1, 1]))
b = tf.Variable(tf.zeros([1]))

learning_rate = 0.0005
epochs = 20000
batch_size = 200

optimizer = tf.optimizers.SGD(learning_rate)

for epoch in range(1, epochs + 1):
    indices = np.random.choice(m, batch_size)
    X_batch = X_data[indices]
    y_batch = y_data[indices]

    with tf.GradientTape() as tape:
        y_pred = tf.matmul(X_batch, k) + b
        loss = tf.reduce_sum(tf.square(y_pred - y_batch))

    grads = tape.gradient(loss, [k, b])
    optimizer.apply_gradients(zip(grads, [k, b]))

    if epoch % 1000 == 0:
        print(f"Епоха {epoch}: {loss.numpy():.6f}, k={k.numpy()[0][0]:.4f}, b={b.numpy()[0]:.4f}")

plt.scatter(X_data, y_data, alpha=0.3, label="Дані")
plt.plot(X_data, X_data * k.numpy()[0][0] + b.numpy()[0], color='red', label="Передбачена лінія")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
