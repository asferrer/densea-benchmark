"""Time split by block and by precision of TensorRT engines profiled with ``jetson/prof_layers.sh``
(``--profilingVerbosity=detailed``, ``--exportLayerInfo`` and ``--exportProfile``).

The per-layer profile adds a synchronization per layer, so its sums are inflated (83.6 ms against 43.8 ms measured
for RT-DETRv2 FP16): use the splits and the differences, not the totals.

Usage: python prof_regions.py DIR engine1,engine2 'block=regex;block2=regex2'
  DIR holds <engine>.layers.json and <engine>.profile.json. A layer goes to the first block whose regex matches
  its name and metadata (lower case); Reformat layers go to 'reformat', the rest to 'other'.
  RT-DETRv2: 'backbone=/backbone/;encoder_conv=/encoder/(input_proj|lateral|fpn|pan|downsample);encoder_aifi=/encoder/encoder;decoder_deformable_attention=cross_attn;decoder_rest=/decoder/'
  RF-DETR:   'backbone_attention=/backbone/.*(attention|attn)/;backbone_mlp=/backbone/.*(mlp|intermediate|output/dense);backbone_norm=/backbone/.*(layernorm|norm);backbone_rest=/backbone/;decoder_deformable_attention=cross_attn;decoder=/transformer/'
"""
import collections
import json
import re
import sys


def parse_rules(spec):
    """'a=regex;b=regex' -> [(a, regex), (b, regex)]."""
    return [tuple(r.split("=", 1)) for r in spec.split(";")]


def region(layer, rules):
    """Block of a profiled layer."""
    text = (layer["Name"] + " " + str(layer.get("Metadata", ""))).lower()
    if layer["LayerType"] == "Reformat" or "reformat" in layer["Name"].lower():
        return "reformat"
    for key, pattern in rules:
        if re.search(pattern, text):
            return key
    return "other"


def precision(layer):
    """Output precision of a profiled layer: INT8, FP16, FP32 or other."""
    outs = layer.get("Outputs") or []
    fmt = outs[0].get("Format/Datatype", "") if outs else ""
    if "Int8" in fmt:
        return "INT8"
    if "Half" in fmt:
        return "FP16"
    if "Float" in fmt or "Thirty-two" in fmt:
        return "FP32"
    return "FP16" if "Channel" in fmt else "other"


def split_times(layers, profile, rules):
    """Total profiled time and its split by block and by output precision."""
    by_block, by_prec = collections.defaultdict(float), collections.defaultdict(float)
    for layer in layers:
        t = profile.get(layer["Name"], 0)
        by_block[region(layer, rules)] += t
        by_prec[precision(layer)] += t
    return sum(profile.values()), by_block, by_prec


def fmt(d, total):
    return ", ".join(f"{k} {v:.1f} ({100 * v / total:.0f} %)" for k, v in sorted(d.items(), key=lambda x: -x[1]))


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    folder, names, rules = sys.argv[1], sys.argv[2].split(","), parse_rules(sys.argv[3])
    for name in names:
        with open(f"{folder}/{name}.layers.json") as f:
            layers = json.load(f)["Layers"]
        with open(f"{folder}/{name}.profile.json") as f:
            profile = {p["name"]: p["averageMs"] for p in json.load(f) if "name" in p}
        total, by_block, by_prec = split_times(layers, profile, rules)
        print(f"== {name}: {len(layers)} layers, profile sum {total:.1f} ms")
        print("  by block:", fmt(by_block, total))
        print("  by output precision:", fmt(by_prec, total))


if __name__ == "__main__":
    main()
