#!/usr/bin/env bash
# Generate the chunks of a copy-paste batch with the generator's tools/run_variants.py, inside the augmentor
# container. Chunk k (1..CHUNKS) uses seed FIRST_SEED + k - 1 and writes
#   /app/output/<OUT_NAME>/chunk_<k>/<VARIANT>/{images/, synthetic_dataset.json, variant.json}
# which the generator's compose file mounts as <GENERATOR_DIR>/output/<OUT_NAME>/chunk_<k>/...
# The depth and segmentation services of the generator must be running. A chunk that is already complete is
# skipped by run_variants.py, so the script can be run again to resume.
#
# Environment variables (required: GENERATOR_DIR, OUT_NAME, CHUNKS):
#   GENERATOR_DIR  checkout of the synthetic-dataset-generator repository
#   OUT_NAME       output folder name under /app/output (for example synthetic_copy_paste)
#   CHUNKS         number of chunks
#   IMAGES         images per chunk                         (default 2000)
#   FIRST_SEED     seed of the first chunk                  (default 21)
#   REAL_DIR       real backgrounds with annotations.json, as seen in the container (default /app/datasets/real_train)
#   VARIANT        run_variants.py variant                  (default s_drop)
#   REQUEST        job request JSON                         (default $GENERATOR_DIR/configs/batch_request.json)
#   CONTAINER      augmentor container name                 (default synth-augmentor)
#   SYNC_APP       1 to copy services/augmentor/app into the container first (default 0)
#
# Example (6 chunks of 2,000 images, seeds 21-26):
#   GENERATOR_DIR=../synthetic-dataset-generator OUT_NAME=synthetic_copy_paste CHUNKS=6 bash synthetic/generate_chunks.sh
set -euo pipefail
export MSYS_NO_PATHCONV=1    # Git Bash on Windows: keep /tmp and /app paths unchanged

: "${GENERATOR_DIR:?set GENERATOR_DIR}" "${OUT_NAME:?set OUT_NAME}" "${CHUNKS:?set CHUNKS}"
IMAGES=${IMAGES:-2000}
FIRST_SEED=${FIRST_SEED:-21}
REAL_DIR=${REAL_DIR:-/app/datasets/real_train}
VARIANT=${VARIANT:-s_drop}
REQUEST=${REQUEST:-$GENERATOR_DIR/configs/batch_request.json}
CONTAINER=${CONTAINER:-synth-augmentor}
SYNC_APP=${SYNC_APP:-0}

[ -f "$REQUEST" ] || { echo "job request not found: $REQUEST" >&2; exit 1; }
if [ "$SYNC_APP" = 1 ]; then
  docker cp "$GENERATOR_DIR/services/augmentor/app/." "$CONTAINER:/app/app/"
fi
docker cp "$GENERATOR_DIR/tools/run_variants.py" "$CONTAINER:/tmp/run_variants.py"
docker cp "$REQUEST" "$CONTAINER:/tmp/batch_request.json"

for k in $(seq 1 "$CHUNKS"); do
  seed=$((FIRST_SEED + k - 1))
  echo "chunk $k/$CHUNKS: $IMAGES images, seed $seed"
  docker exec "$CONTAINER" python /tmp/run_variants.py --request /tmp/batch_request.json \
    --n "$IMAGES" --seed "$seed" --out "/app/output/$OUT_NAME/chunk_$k" \
    --real-dir "$REAL_DIR" --save-masks --variants "$VARIANT"
done
echo "done: $GENERATOR_DIR/output/$OUT_NAME"
