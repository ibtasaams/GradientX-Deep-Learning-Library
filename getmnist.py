import urllib.request, gzip, struct, numpy as np

def load_images(path):
    with gzip.open(path) as f:
        f.read(4)
        n = struct.unpack('>I', f.read(4))[0]
        struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), np.uint8).reshape(n, 784).astype(np.float32) / 255.0

def load_labels(path):
    with gzip.open(path) as f:
        f.read(4)
        n = struct.unpack('>I', f.read(4))[0]
        return np.frombuffer(f.read(), np.uint8).astype(np.float32)

base = "https://ossci-datasets.s3.amazonaws.com/mnist/"
files = [
    ("train-images-idx3-ubyte.gz", "tr_img.gz"),
    ("train-labels-idx1-ubyte.gz", "tr_lbl.gz"),
    ("t10k-images-idx3-ubyte.gz",  "te_img.gz"),
    ("t10k-labels-idx1-ubyte.gz",  "te_lbl.gz"),
]
for name, dest in files:
    print(f"Downloading {name}...")
    urllib.request.urlretrieve(base + name, dest)

load_images("tr_img.gz").tofile("train_images.mat")
load_labels("tr_lbl.gz").tofile("train_labels.mat")
load_images("te_img.gz").tofile("test_images.mat")
load_labels("te_lbl.gz").tofile("test_labels.mat")
print("Done!")