*default cares18; `()` means seres18*

Market1501 

Baseline -- @Acc1: 91.7%; @mAP: 77.8%

| Continual | Metric    | Triplet Margin | Acc@1           | mAP             |
|-----------|-----------|----------------|-----------------|-----------------|
| ❌         | Euclidean | 0.3            | 0.9240          | 0.7915          |
| ❌         | Euclidean | Soft           | 0.9243 (0.9231) | 0.8030 (0.8203) |
| ❌         | Jaccard   | 0.3            | 0.9267          | 0.8852          |
| ❌         | Jaccard   | Soft           | 0.9279 (0.9311) | 0.8893 (0.8923) |
| ✔️        | Jaccard   | 0.3            | 0.9463          | 0.9210          |
| ✔️        | Jaccard   | Soft           | (0.9525)        | (0.9235)        |

DukeMTMC

Baseline -- DukeMTMC @Acc1: 82.5%; @mAP: 68.8%

| Continual | Metric    | Triplet Margin | Acc@1           | mAP             |
|-----------|-----------|----------------|-----------------|-----------------|
| ❌         | Euclidean | 0.3            | 0.8182          | 0.6967          |
| ❌         | Euclidean | Soft           | 0.8223 (0.8110) | 0.7134 (0.7315) |
| ❌         | Jaccard   | 0.3            | 0.8532          | 0.8041          |
| ❌         | Jaccard   | Soft           | 0.8618 (0.8703) | 0.8108 (0.8273) |
| ✔️        | Jaccard   | 0.3            | 0.8757          | 0.8312          |
| ✔️        | Jaccard   | Soft           | 0.8811 (0.8905) | 0.8422 (0.8502) |

| Use Side (Co-eff=1.0) | Metric    | Acc@1  | mAP    |
|-----------------------|-----------|--------|--------|
| ❌                     | Euclidean | 0.8223 | 0.7134 |
| ✔️                    | Euclidean | 0.8061 | 0.7273 |
| ❌                     | Jaccard   | 0.8618 | 0.8108 |
| ✔️                    | Jaccard   | 0.8694 | 0.8127 |

Veri776 on SeRes18, optionally apply re-detection [Cropped Detector](https://drive.google.com/file/d/1SYwGRfH9fSAt_keZahbDFMVhjscD5kZ9/view?usp=drive_link)

Baseline -- Veri776 @Acc1: 87.6%; @mAP: 59.0%

| Use Side (Co-eff=-1.0) | Metric    | Acc@1    | mAP      |
|------------------------|-----------|----------|----------|
| ✔️                     | Euclidean | (0.8737) | (0.6923) |
| ❌                      | Jaccard   | (0.9255) | (0.7394) |
| ✔️                     | Jaccard   | (0.9249) | (0.7420) |
| ✔️w/. continual        | Jaccard   | (0.9344) | (0.7436) |

* Domain Transfer

Market1501 -> DukeMTMC

Baseline -- @Acc1: 40.0%; @mAP: 25.1%

| Continual | Metric       | Acc@1  | mAP    |
|-----------|--------------|--------|--------|
| ❌         | Cross-domain | 0.4969 | 0.3334 |
| ✔️        | Cross-domain | 0.5368 | 0.3896 |

DukeMTMC -> Market1501

Baseline -- @Acc1: 52.9%; @mAP: 25.1%

| Continual | Metric       | Acc@1  | mAP    |
|-----------|--------------|--------|--------|
| ❌         | Cross-domain | 0.6256 | 0.3933 |
| ✔️        | Cross-domain | 0.6580 | 0.4130 |