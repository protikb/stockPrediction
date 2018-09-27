from matplotlib import pyplot as plt
import numpy as np
import dtcwt

# Generate a 300x2 array of a random walk
vecs = np.cumsum(np.random.rand(300,2) - 0.5, 0)

# Show input
#figure()
#plot(vecs)
#title('Input')
fig = plt.figure()
plt.plot(vecs)
plt.show()
fig.suptitle('test title', fontsize=20)
#plt.title("input")

# 1D transform, 5 levels
transform = dtcwt.Transform1d()
vecs_t = transform.forward(vecs, nlevels=5)

# Show level 2 highpass coefficient magnitudes
#figure()
#plot(np.abs(vecs_t.highpasses[1]))
#title('Level 2 wavelet coefficient magnitudes')
plt.plot(np.abs(vecs_t.highpasses[1]))
plt.show()

# Show last level lowpass image
#figure()
#plot(vecs_t.lowpass)
#title('Lowpass signals')
plt.plot(vecs_t.lowpass)
plt.show()

# Inverse
vecs_recon = transform.inverse(vecs_t)

# Show output
#figure()
#plot(vecs_recon)
#title('Output')
plt.plot(vecs_recon)
plt.show()

# Show error
#figure()
#plot(vecs_recon - vecs)
#title('Reconstruction error')
plt.plot(vecs_recon - vecs)
plt.show()
print('Maximum reconstruction error: {0}'.format(np.max(np.abs(vecs - vecs_recon))))
