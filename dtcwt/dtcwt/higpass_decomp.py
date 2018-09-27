from matplotlib import pyplot as plt
import numpy as np
import dtcwt
import copy 

# Generate a 300x2 array of a random walk
vecs = np.cumsum(np.random.rand(300,1) - 0.5, 0)
#print vecs
# 1D transform, 5 levels
transform = dtcwt.Transform1d()
vecs_t = transform.forward(vecs, nlevels=6)
#print vecs_t
# Make Copies
vecs_t1 = copy.deepcopy(vecs_t)
vecs_t2 = copy.deepcopy(vecs_t)
vecs_t3 = copy.deepcopy(vecs_t)
vecs_t4 = copy.deepcopy(vecs_t)
vecs_t5 = copy.deepcopy(vecs_t)

# Inverse
vecs_recon = transform.inverse(vecs_t)

# Inverse with first component removed
for jj in range(len(vecs_t1.highpasses[0])):
    vecs_t1.highpasses[0][jj] = 0     

vecs_recon1 = transform.inverse(vecs_t1)

# Inverse with first and second component removed
for jj in range(len(vecs_t1.highpasses[1])):
    vecs_t1.highpasses[1][jj] = 0 

vecs_recon2 = transform.inverse(vecs_t1)

# Inverse with first, second and third component removed
for jj in range(len(vecs_t1.highpasses[2])):
    vecs_t1.highpasses[2][jj] = 0 

vecs_recon3 = transform.inverse(vecs_t1)

# Inverse with first, second, third and fourth component removed
for jj in range(len(vecs_t1.highpasses[3])):
    vecs_t1.highpasses[3][jj] = 0 

vecs_recon4 = transform.inverse(vecs_t1)

# Inverse with first, second, third, fourth and fifth component removed
for jj in range(len(vecs_t1.highpasses[4])):
    vecs_t1.highpasses[4][jj] = 0 

vecs_recon5 = transform.inverse(vecs_t1)

plt.subplot(2, 3, 1)
plt.title('Input')
plt.plot(vecs)

# Show the component removed
plt.subplot(2, 3, 2)
plt.title('High Pass')
plt.plot(vecs_recon1)

plt.subplot(2, 3, 3)
plt.title('Reconstruction')
plt.plot(vecs_recon2)

plt.subplot(2, 3, 4)
plt.title('Reconstruction')
plt.plot(vecs_recon3)

plt.subplot(2, 3, 5)
plt.title('Reconstruction')
plt.plot(vecs_recon4)

plt.subplot(2, 3, 6)
plt.title('Reconstruction')
plt.plot(vecs_recon5)

plt.show()

print('Maximum reconstruction error: {0}'.format(np.max(np.abs(vecs - vecs_recon))))
for i in range (0,5):
    # 1D transform, 5 levels
    transform = dtcwt.Transform1d()
    if len(vecs_t1.highpasses[i])%2!=0:
       continue
    #vecs = np.cumsum(np.random.rand(300,1) - 0.5, 0)
    vecs_t_h = transform.forward(vecs_t1.highpasses[i], nlevels=6-i-1)
    vecs_t_h = transform.forward(vecs, nlevels=6-i-1)
    # Make Copies
    vecs_t1_h = copy.deepcopy(vecs_t_h)
    vecs_t2_h = copy.deepcopy(vecs_t_h)
    vecs_t3_h = copy.deepcopy(vecs_t_h)
    vecs_t4_h = copy.deepcopy(vecs_t_h)
    vecs_t5_h = copy.deepcopy(vecs_t_h)

    # Inverse
    vecs_recon_h = transform.inverse(vecs_t_h)
    plt.subplot(2,3,1)
    plt.title('Input')
    #plt.plot(vecs_t1.highpasses[i])#instead of plt.plot(vecs)
    plt.plot(vecs)
    #plt.show()
    for no_of_comp in range(0,6-i-1):
    
        for jj in range(len(vecs_t1_h.highpasses[no_of_comp])):
            vecs_t1_h.highpasses[no_of_comp][jj] = 0     

        vecs_recon1_h = transform.inverse(vecs_t1_h)
        title = "higpass ",i
        plt.subplot(2,3,no_of_comp+2)
        plt.title(title)
        plt.plot(vecs_recon1_h)
    plt.show()
#     # Inverse with first and second component removed
#     for jj in range(len(vecs_t1_h.highpasses[1])):
#         vecs_t1_h.highpasses[1][jj] = 0 

#     vecs_recon2_h = transform.inverse(vecs_t1_h)

#     # Inverse with first, second and third component removed
#     for jj in range(len(vecs_t1_h.highpasses[2])):
#         vecs_t1_h.highpasses[2][jj] = 0 

#     vecs_recon3_h = transform.inverse(vecs_t1_h)

#     # Inverse with first, second, third and fourth component removed
#     for jj in range(len(vecs_t1_h.highpasses[3])):
#         vecs_t1_h.highpasses[3][jj] = 0 

#     vecs_recon4_h = transform.inverse(vecs_t1_h)

#     # Inverse with first, second, third, fourth and fifth component removed
#     for jj in range(len(vecs_t1_h.highpasses[4])):
#         vecs_t1_h.highpasses[4][jj] = 0 

#     vecs_recon5_h = transform.inverse(vecs_t1_h)

#     plt.subplot(2, 3, 1)
#     plt.title('Input')
#     plt.plot(vecs_t1.highpasses[0])#instead of plt.plot(vecs)

#     # Show the component removed
#     plt.subplot(2, 3, 2)
#     plt.title('High Pass')
#     plt.plot(vecs_recon1_h)

#     plt.subplot(2, 3, 3)
#     plt.title('Reconstruction')
#     plt.plot(vecs_recon2_h)

#     plt.subplot(2, 3, 4)
#     plt.title('Reconstruction')
#     plt.plot(vecs_recon3_h)

#     plt.subplot(2, 3, 5)
#     plt.title('Reconstruction')
#     plt.plot(vecs_recon4_h)

#     plt.subplot(2, 3, 6)
#     plt.title('Reconstruction')
#     plt.plot(vecs_recon5_h)

#     plt.show()

    print('Maximum reconstruction error: {0}'.format(np.max(np.abs(vecs_t1.highpasses[i] - vecs_recon_h))))