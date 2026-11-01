# Clean-background batch: parameters

The clean-background batch (label `v9`) pastes object crops onto background images that contain no objects.
It is requested from the augmentor of the
[synthetic-dataset-generator](https://github.com/asferrer/synthetic-dataset-generator) with
`clean_background_request.json`:

```bash
curl -X POST http://localhost:8004/compose-batch -H "Content-Type: application/json" \
     -d @synthetic/clean_background_request.json
```

The job writes `<output_dir>/<job_id>/{images/, synthetic_dataset.json}`. Copy that folder to
`DATA_ROOT/synthetic/clean_background/` (see `docs/synthetic_batches.md`). The augmentor is called directly
(port 8004) because the gateway does not forward every field of the request, for example `depth_service_url`.

## Inputs

| Parameter | Value | Justification |
|---|---|---|
| `objects_dir` | `Extracted_objects_train` | RGBA crops of annotated objects of the train split only; no crop comes from val or test. |
| `backgrounds_dir` | `Backgrounds_filtered` | Candidate backgrounds filtered with `synthetic/check_backgrounds.py`: a background that has a content-similar image in val or test (group-partition metric) is excluded. Of 424 candidates, 4 were excluded and 420 used. |

## Volume and placement

| Parameter | Value | Justification |
|---|---|---|
| `targets_per_class` | 5,000 for each of the 8 classes | Targets count pasted instances, not images. The batch holds 10,771 images and 40,004 boxes (`prepare/reference/SUMMARY.json`). |
| `num_images` | 40,000 | Upper bound on the number of images; generation ends when the per-class targets are reached. |
| `max_objects_per_image` | 8 | At most one pasted object per class (8 classes) in an image. |
| `depth_aware` | true | Object scale follows the depth map of the background (depth service). |
| `semantic_placement` | true | Position of each object follows the behaviour of its class (`placement.py`). |

## Effects

The effect list is `color_correction`, `blur_matching`, `underwater`, `caustics`, `shadows`,
`edge_smoothing`. `lighting` and `motion_blur` are not used. Every value of `effects_config` is the default of
`EffectsConfig` (`services/augmentor/app/models/schemas.py`) and is written explicitly so that it is recorded
in the job.

| Parameter | Value | Effect |
|---|---|---|
| `color_intensity` | 0.25 | LAB colour transfer from the background to the object. |
| `blur_strength` | 0.65 | Blur of the object matched to the background. |
| `underwater_intensity` | 0.33 | Underwater tint. |
| `caustics_intensity` | 0.12 | Caustics pattern, seeded per image. |
| `shadow_opacity`, `shadow_blur` | 0.12, 28 | Diffuse shadow under the object. |
| `edge_feather` | 12 | Edge smoothing; the radius adapts to the contrast. |
| `water_color`, `water_clarity` | (180, 120, 30) BGR, `clear` | Colour of the water tint. |
| `underwater_physical`, `caustics_depth_aware`, `depth_occlusion`, `validate_identity` | false | Optional physical options and identity check, off. |
| `use_binary_alpha`, `alpha_feather_radius` | true, 2 | Binary alpha with a 2 px anti-aliased edge. |
| `max_blur_budget` | 10.0 | Maximum cumulative blur in pixels for small objects. |
| `recalculate_bbox_after_global` | true | After global effects the box is recomputed from the image difference within 20 px around the object, so boxes can include the shadow and the tint. |

## Execution

| Parameter | Value | Justification |
|---|---|---|
| `seed` | 21 | Reproducible in sequential mode only: in parallel mode the random draws of concurrent compositions interleave. |
| `parallel`, `concurrent_limit`, `vram_threshold` | true, 2, 0.7 | Up to two concurrent compositions; GPU memory is cleaned up when its use exceeds 70 %. |
| `save_pipeline_debug` | false | No intermediate images are stored. |
| `validate_quality`, `validate_physics`, `reject_invalid` | true | Composed images that fail validation are rejected. |

## What the validation does and does not measure

In the batch path the validator receives the composed image and its boxes, without reference images:

1. **Perceptual quality.** Without references the score is a self-consistency heuristic on edge density and on the
   LAB standard deviation, which returns 0.7, 0.8 or 1.0. It is not LPIPS.
2. **Anomaly score.** The Isolation Forest is not fitted; the score is a heuristic on channel statistics and on the
   variance of the Laplacian.
3. **Composition.** Area ratio, border, aspect ratio and object count of the boxes; no image regions are compared.
4. **Thresholds.** 0.7 (perceptual), 0.6 (anomaly) and 0.5 (composition) are fixed; the batch request does not
   expose them. The distribution match (FID) is not computed and is reported as 1.0.
5. **Physics.** Only violations of severity `high` reject an image; occlusion checks do not run because no depth
   map is passed.

Validation therefore filters degenerate compositions; it is not a measure of realism. A reference-based check can
be run after generation with `synthetic/posthoc_validate.py`: it validates a random sample of the batch against
real train images through `POST /validate`, which accepts `reference_images`, and reports the LPIPS-based scores
without rejecting anything.

## Library statistics

The crop library has 39 crops of `Metal_Debris` and 246 of `Can` for 5,000 placements of each class, so each of
these crops is reused about 128 and 20 times on average.
