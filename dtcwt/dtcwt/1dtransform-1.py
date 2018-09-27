from matplotlib.pylab import *
import dtcwt

# Generate a 300x2 array of a random walk
vecs = np.cumsum(np.random.rand(300,2) - 0.5, 0)

# Show input
figure()
plot(vecs)
title('Input')

# 1D transform, 5 levels
transform = dtcwt.Transform1d()
vecs_t = transform.forward(vecs, nlevels=5)

# Show level 2 highpass coefficient magnitudes
figure()
plot(np.abs(vecs_t.highpasses[1]))
title('Level 2 wavelet coefficient magnitudes')
plot.show()
# Show last level lowpass image
figure()
plot(vecs_t.lowpass)
title('Lowpass signals')
plot.show()
# Inverse
vecs_recon = transform.inverse(vecs_t)

# Show output
figure()
plot(vecs_recon)
title('Output')
plot.show()
# Show error
figure()
plot(vecs_recon - vecs)
title('Reconstruction error')
plot.show()
print('Maximum reconstruction error: {0}'.format(np.max(np.abs(vecs - vecs_recon))))
