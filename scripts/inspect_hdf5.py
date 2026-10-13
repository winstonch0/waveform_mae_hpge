import h5py

path = "data/raw/MJD_Train_0.hdf5"

with h5py.File(path, "r") as f:
    print("HDF5 contents:\n")

    for key in f.keys():
        x = f[key]
        print(f"{key:25} shape={x.shape} dtype={x.dtype}")

