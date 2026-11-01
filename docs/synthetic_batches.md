# Synthetic batches

The synthetic images are produced with the
[synthetic-dataset-generator](https://github.com/asferrer/synthetic-dataset-generator) (called "the generator"
below) and arranged under `DATA_ROOT/synthetic/` so that `prepare/` can read them. The two batches of the study
and the batches of the incremental series are produced as follows. Statistics, crop libraries and the generator
options are described in the generator's `docs/SYNTHESIS_PROFILE.md`.

| Label | Folder under `DATA_ROOT/synthetic/` | Content | Images |
|---|---|---|---:|
| `v9` | `clean_background/` | clean-background batch | 10,771 |
| `v11` | `copy_paste/` | copy-paste batch onto real train images | 12,000 |
| `v11_r25` | `copy_paste_r25/` | copy-paste onto the 25 % real subset | 3,000 |
| `v11_r50` | `copy_paste_r50/` | copy-paste onto the 50 % real subset | 6,000 |
| `p_pos_tight`, `q_lab_same` | `proxy_p_pos_tight/`, `proxy_q_lab_same/` | copy-paste batches of the transfer check (10 % real subset) | 2,000 each |

Each folder holds `synthetic_dataset.json` (COCO) and the images. `prepare/prepare_strategies.py` reads
`file_name` relative to `<folder>/images/` for `clean_background` and the two transfer-check folders, and
relative to `<folder>/` for the merged copy-paste folders (`chunk_<k>/images/<file>`), and stores
`synthetic/<folder>/...` in the training JSON.

The commands use these variables (the generator mounts `DATASET_PATH` as `/app/datasets` and its `output/`
folder as `/app/output`):

```bash
export GEN=../synthetic-dataset-generator        # checkout of the generator, services running (docker compose up)
export DATASET_PATH=<folder mounted as /app/datasets>   # Backgrounds_filtered, crop libraries, real_train*
export DATA_ROOT=<DenSea-Aggregated root>        # annotations/, images/ (CleanSea images restored)
export PREPARE_OUT=prepare/out                   # training sets written by prepare/
```

## Real backgrounds of a split

The copy-paste batches paste onto real train images. The generator builds the folder from a train COCO file;
`--ids-from` restricts it to the images of an experiment (only train images are accepted):

```bash
python $GEN/scripts/build_real_train.py --train $DATA_ROOT/annotations/train.json --images-root $DATA_ROOT \
    --out $DATASET_PATH/real_train
```

## Clean-background batch (`v9`)

```bash
# 1. Keep only backgrounds without a relative in val or test (writes allowed_backgrounds.txt and BACKGROUNDS.md)
python synthetic/check_backgrounds.py --backgrounds-dir $DATASET_PATH/Backgrounds_filtered --out out/backgrounds
#    copy the allowed files to the folder used as backgrounds_dir

# 2. Request the batch (parameters and justification: synthetic/PARAMETERS.md)
curl -X POST http://localhost:8004/compose-batch -H "Content-Type: application/json" \
     -d @synthetic/clean_background_request.json

# 3. Copy the job folder (images/ and synthetic_dataset.json) to the batch folder
cp -r $GEN/output/synthetic_clean_background/<job_id> $DATA_ROOT/synthetic/clean_background
```

An interrupted job continues with `POST http://localhost:8004/jobs/<job_id>/resume`.

## Copy-paste batch (`v11`)

Six chunks of 2,000 images, seeds 21 to 26, variant `s_drop`, request `configs/batch_request.json` of the
generator (12,000 images, per-class targets Bottle 1,500, Can 4,500, Fishing_Net 2,500, Glove 2,500, Mask 1,500,
Metal_Debris 4,500, Plastic_Debris 1,500, Tire 1,500):

```bash
GENERATOR_DIR=$GEN OUT_NAME=synthetic_copy_paste CHUNKS=6 IMAGES=2000 FIRST_SEED=21 \
    bash synthetic/generate_chunks.sh

python synthetic/merge_chunks.py --src $GEN/output/synthetic_copy_paste \
    --dst $DATA_ROOT/synthetic/copy_paste --chunks 6 --request $GEN/configs/batch_request.json \
    --name copy_paste --summary out/copy_paste_batch_summary.json
```

`out/copy_paste_batch_summary.json` has to match `synthetic/reference/copy_paste_batch_summary.json` in the
images, boxes and pasted boxes per class.

## Incremental series (25 % and 50 % real)

For each real fraction there is a copy-paste batch of about 1.17 synthetic images per real image (the ratio of
12,000 images to the 10,245 train images), pasted only onto the images of that fraction.

| Fraction | Real images | Chunks | Images per chunk | Seeds | Synthetic images |
|---|---:|---:|---:|---|---:|
| 25 % | 2,561 | 2 | 1,500 | 31, 32 | 3,000 |
| 50 % | 5,122 | 3 | 2,000 | 41, 42, 43 | 6,000 |

```bash
# 1. Real subsets of each fraction (class-stratified, seed 21; no synthetic batch needed)
python prepare/prepare_strategies.py --experiments exp12_real25 exp13_real50

# 2. Real-background folders of each fraction: --real-dir points to a folder built from that fraction's train images
python $GEN/scripts/build_real_train.py --train $DATA_ROOT/annotations/train.json --images-root $DATA_ROOT \
    --ids-from $PREPARE_OUT/exp12_real25/train.json --out $DATASET_PATH/real_train_r25
python $GEN/scripts/build_real_train.py --train $DATA_ROOT/annotations/train.json --images-root $DATA_ROOT \
    --ids-from $PREPARE_OUT/exp13_real50/train.json --out $DATASET_PATH/real_train_r50

# 3. Chunks of each fraction
GENERATOR_DIR=$GEN OUT_NAME=synthetic_copy_paste_r25 CHUNKS=2 IMAGES=1500 FIRST_SEED=31 \
    REAL_DIR=/app/datasets/real_train_r25 bash synthetic/generate_chunks.sh
GENERATOR_DIR=$GEN OUT_NAME=synthetic_copy_paste_r50 CHUNKS=3 IMAGES=2000 FIRST_SEED=41 \
    REAL_DIR=/app/datasets/real_train_r50 bash synthetic/generate_chunks.sh

# 4. Merge
python synthetic/merge_chunks.py --src $GEN/output/synthetic_copy_paste_r25 --dst $DATA_ROOT/synthetic/copy_paste_r25 \
    --chunks 2 --request $GEN/configs/batch_request.json --name copy_paste_r25
python synthetic/merge_chunks.py --src $GEN/output/synthetic_copy_paste_r50 --dst $DATA_ROOT/synthetic/copy_paste_r50 \
    --chunks 3 --request $GEN/configs/batch_request.json --name copy_paste_r50

# 5. Training sets of the series
python prepare/prepare_series.py
```

`prepare_series.py` stops if a copy-paste image uses a background outside its fraction (the `background` field is
`<image id:05d>_<name>`, the naming of `build_real_train.py`). The reference summaries are
`synthetic/reference/copy_paste_r25_summary.json` and `copy_paste_r50_summary.json`.

## Transfer-check batches (10 % real)

Two copy-paste variants, 2,000 images each, seed 21, pasted onto the 10 % real subset of `exp11_real10`:

```bash
python prepare/prepare_strategies.py --experiments exp11_real10
python $GEN/scripts/build_real_train.py --train $DATA_ROOT/annotations/train.json --images-root $DATA_ROOT \
    --ids-from $PREPARE_OUT/exp11_real10/train.json --out $DATASET_PATH/real_train_r10
for v in p_pos_tight q_lab_same; do
  GENERATOR_DIR=$GEN OUT_NAME=synthetic_proxy_$v CHUNKS=1 IMAGES=2000 FIRST_SEED=21 VARIANT=$v \
      REAL_DIR=/app/datasets/real_train_r10 bash synthetic/generate_chunks.sh
  mkdir -p $DATA_ROOT/synthetic/proxy_$v
  cp -r $GEN/output/synthetic_proxy_$v/chunk_1/$v/. $DATA_ROOT/synthetic/proxy_$v/
done
python prepare/prepare_proxy.py
```

## Quality check against real references

`synthetic/posthoc_validate.py` validates a sample of a batch against real train images with the reference-based
endpoint of the augmentor (see the last section of `synthetic/PARAMETERS.md`).
