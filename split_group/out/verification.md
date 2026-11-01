# Group partition verification

Images: train 10245, val 2185, test 2207. Violations: **0**.

Shared keys between partitions (0 means no key crosses partitions).

| Check | train-val | train-test | val-test |
|---|---|---|---|
| normalised name keys | 0 | 0 | 0 |
| CleanSea dive + camera | 0 | 0 | 0 |
| CleanSea dive | 0 | 0 | 0 |
| group ids | 0 | 0 | 0 |

Content-similar pairs across partitions.

| Check | train-val | train-test | val-test |
|---|---|---|---|
| partition criterion | 0 | 0 | 0 |
| looser 1: dhash<=28, cos>=0.92 (CleanSea), cos>=0.84 (Roboflow) | 888 | 541 | 140 |
| looser 2: dhash<=32, cos>=0.90 (CleanSea), cos>=0.82 (Roboflow) | 4678 | 2279 | 380 |

Nearest neighbour in another partition.

| Partition | Max cosine | Min dHash (Roboflow) |
|---|---|---|
| train | 0.948 | 26 |
| val | 0.9246 | 29 |
| test | 0.948 | 40 |
