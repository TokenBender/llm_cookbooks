from numba import cuda
device = cuda.get_current_devices()
device.reset()