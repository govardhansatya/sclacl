# Datasets

The datasets are not stored in this repository. Download them with

```bash
python code/fetch_data.py
```

which saves the two files below into `data/` and checks their SHA-256 hashes against the
copies used for the paper.

| File | Dataset | Source | Used as | SHA-256 |
|---|---|---|---|---|
| `elec.arff` | ELEC2 (electricity pricing), 45,312 time-ordered records (Harries, 1999) | OpenML dataset 151 | full stream | `2d86fbc74c69a5c04bcde0119bce280dd6d8efa0115987edc75b77ccf024d7f2` |
| `covertype.arff` | Covertype, 581,012 rows (Blackard and Dean, 1999) | OpenML dataset 1596 (UCI, CC BY 4.0) | contiguous 40,000-row block starting at row 60,000, features standardized | `689d65323b4c24659de6116bb3d6f2250c9c1dfcfe4f348f07688583dea75edf` |

SEA (Street and Kim, 2001) is generated in code (`gen_sea()` in `code/sclacl_experiments.py`):
40,000 samples, 3 features, abrupt concept changes at samples 10k, 20k and 30k, one stream per seed.

If the OpenML download fails or the hash differs, download the same datasets manually from
OpenML (ids 151 and 1596) and place them here under the file names above.
