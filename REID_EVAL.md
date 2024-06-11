[Checkpoints](https://drive.google.com/drive/folders/1GC8CXXfMbfCR_-hJaMKGTNUx2-XU9Gp4?usp=sharing)
* Metric-learning

Market1501 

Baseline -- @Acc1: 91.7%; @mAP: 77.8%

| Action          | Acc@1  | mAP    |
|-----------------|--------|--------|
| Baseline        | 0.9181 | 0.7994 |
| Inference Boost | 0.92+  | 0.82+  |
| Backbone Boost️ | 0.9451 | 0.8877 |
| Metric Boost    | 0.9406 | 0.9172 |
| Training Boost️ | 0.9477 | 0.9222 |
| Continual Boost | 0.9572 | 0.9406 |

DukeMTMC

Baseline -- DukeMTMC @Acc1: 82.5%; @mAP: 68.8%

| Metric    | Acc@1  | Acc@5  | Acc@10 | mAP    |
|-----------|--------|--------|--------|--------|
| Euclidean | 0.8613 | 0.9336 | 0.9502 | 0.7506 |
| Jaccard   | 0.8873 | 0.9327 | 0.9430 | 0.8464 |

Veri776 on SeRes18, optionally apply re-detection [Cropped Detector](https://drive.google.com/file/d/1SYwGRfH9fSAt_keZahbDFMVhjscD5kZ9/view?usp=drive_link)

Baseline -- Veri776 @Acc1: 87.6%; @mAP: 59.0%

| Metric    | Acc@1  | Acc@5  | Acc@10 | mAP    |
|-----------|--------|--------|--------|--------|
| Euclidean | 0.9439 | 0.9750 | 0.9815 | 0.7486 |
| Jaccard   | 0.9523 | 0.9666 | 0.9738 | 0.7988 |

* Domain Transfer

Market1501 -> DukeMTMC

Baseline -- @Acc1: 40.0%; @mAP: 25.1%

| Continual | Metric       | Acc@1  | mAP    |
|-----------|--------------|--------|--------|
| ❌         | Cross-domain | 0.5794 | 0.4454 |
| ✔️        | Cross-domain | 0.6198 | 0.4939 |

DukeMTMC -> Market1501

Baseline -- @Acc1: 52.9%; @mAP: 25.1%

| Continual | Metric       | Acc@1  | mAP    |
|-----------|--------------|--------|--------|
| ❌         | Cross-domain | 0.6452 | 0.4809 |
| ✔️        | Cross-domain | 0.7072 | 0.5476 |