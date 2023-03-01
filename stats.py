import numpy as np

A = np.load("softmax_0.99_4_4_0.3_transformerscodegen-16B-mono_early_layers.npy")
print(np.mean(A))