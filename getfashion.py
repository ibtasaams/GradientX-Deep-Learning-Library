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

base = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
alt_base = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"

files = [
    ("train-images-idx3-ubyte.gz", "fashion_tr_img.gz"),
    ("train-labels-idx1-ubyte.gz", "fashion_tr_lbl.gz"),
    ("t10k-images-idx3-ubyte.gz",  "fashion_te_img.gz"),
    ("t10k-labels-idx1-ubyte.gz",  "fashion_te_lbl.gz"),
]

for name, dest in files:
    print(f"Downloading {name}...")
    try:
        req = urllib.request.Request(base + name, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(dest, 'wb') as out_file:
            out_file.write(response.read())
    except Exception as e:
        print(f"Primary source failed ({e}), trying alternative...")
        req = urllib.request.Request(alt_base + name, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(dest, 'wb') as out_file:
            out_file.write(response.read())

print("Extracting and formatting Fashion MNIST matrices...")
load_images("fashion_tr_img.gz").tofile("fashion_train_images.mat")
load_labels("fashion_tr_lbl.gz").tofile("fashion_train_labels.mat")
load_images("fashion_te_img.gz").tofile("fashion_test_images.mat")
load_labels("fashion_te_lbl.gz").tofile("fashion_test_labels.mat")
print("Done! Fashion MNIST is ready.")
