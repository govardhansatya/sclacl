#!/usr/bin/env python3
"""Re-download the datasets used by the experiments into the repository-root data/ folder.
ELEC2 (OpenML id 151) and Covertype (OpenML id 1596). SEA is generated in code.
Usage (from the repository root): python3 code/fetch_data.py
"""
import hashlib, os, urllib.request

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")  # repository-root data/
# SHA-256 of the copies used for the paper; a mismatch is reported but not fatal.
SHA256 = {
    "elec.arff":      "2d86fbc74c69a5c04bcde0119bce280dd6d8efa0115987edc75b77ccf024d7f2",
    "covertype.arff": "689d65323b4c24659de6116bb3d6f2250c9c1dfcfe4f348f07688583dea75edf",
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

FILES = {
    "elec.arff":      "https://openml.org/data/v1/download/2419/electricity.arff",
    "covertype.arff": "https://openml.org/data/v1/download/1601911/covertype.arff",
}

def main():
    os.makedirs(DATA, exist_ok=True)
    for name, url in FILES.items():
        dest = os.path.join(DATA, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"[skip] {name} already present ({os.path.getsize(dest)/1e6:.1f} MB)")
        else:
            print(f"[get ] {name} <- {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
                f.write(r.read())
            print(f"[ok  ] {name} ({os.path.getsize(dest)/1e6:.1f} MB)")
        digest = sha256(dest)
        if digest == SHA256[name]:
            print(f"[hash] {name} matches the copy used for the paper")
        else:
            print(f"[WARN] {name} SHA-256 {digest} differs from the paper's copy {SHA256[name]}")
    print("\nDone. Now run: python3 code/sclacl_experiments.py")

if __name__ == "__main__":
    main()
