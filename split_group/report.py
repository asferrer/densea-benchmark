"""Report of the group partition: images and boxes per partition, class and source."""
import collections
import json

from common import paths

SOURCES = ["CleanSea", "Neural_Ocean", "Ocean_garbage"]


def tabulate(rows, classes, ids):
    """Counts per partition: images, boxes, per class and per source.

    rows/classes are aligned (see ``pooling.pool_annotations``); ids maps partition -> image ids.
    """
    src = {r["id"]: r["source"] for r in rows}
    anns = {r["id"]: c for r, c in zip(rows, classes)}
    res = {}
    for s, lst in ids.items():
        cls_ann = collections.Counter(c for i in lst for c in anns[i])
        cls_img = collections.Counter(c for i in lst for c in set(anns[i]))
        src_img = collections.Counter(src[i] for i in lst)
        src_ann = collections.Counter({k: 0 for k in SOURCES})
        for i in lst:
            src_ann[src[i]] += len(anns[i])
        res[s] = {"images": len(lst), "annotations": sum(cls_ann.values()),
                  "empty_images": sum(1 for i in lst if not anns[i]),
                  "ann_per_class": {paths.CLASSES[c]: cls_ann[c] for c in range(len(paths.CLASSES))},
                  "img_per_class": {paths.CLASSES[c]: cls_img[c] for c in range(len(paths.CLASSES))},
                  "img_per_source": {k: src_img[k] for k in SOURCES},
                  "ann_per_source": {k: src_ann[k] for k in SOURCES}}
    return res


def _md_table(res, key, cols):
    tot = {c: sum(res[s][key][c] for s in res) for c in cols}
    lines = ["| Partition | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for s in res:
        cells = [f"{res[s][key][c]} ({100 * res[s][key][c] / max(tot[c], 1):.1f} %)" for c in cols]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines.append("| total | " + " | ".join(str(tot[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_report(out, rows, classes, ids, gstats):
    """Write ``report.json`` and ``REPORT.md`` into ``out`` and return the counts."""
    res = tabulate(rows, classes, ids)
    json.dump({"counts": res, "groups": gstats}, open(out / "report.json", "w"), indent=1)
    n_img = sum(r["images"] for r in res.values())
    n_ann = sum(r["annotations"] for r in res.values())
    md = ["# Group partition report", "",
          f"Pooled annotations: {n_img} images, {n_ann} boxes. "
          f"Seed {gstats['seed']}, {gstats['restarts']} restarts (restart {gstats['best_restart']} chosen), "
          f"cost {gstats['cost']}.", "",
          f"Groups: {gstats['groups_final']} (components by name only {gstats['components_name']}, "
          f"by name and content {gstats['components_name_content']}; content edges "
          f"{gstats['content_edges']}; largest group {gstats['largest_group']} images; CleanSea "
          f"grouped by `{gstats['cleansea_mode']}`).", "",
          "## Totals", "", "| Partition | Images | Boxes | Empty images |", "|---|---|---|---|"]
    for s, r in res.items():
        md.append(f"| {s} | {r['images']} ({100 * r['images'] / n_img:.1f} %) | "
                  f"{r['annotations']} ({100 * r['annotations'] / n_ann:.1f} %) | {r['empty_images']} |")
    md += ["", "## Boxes per class", "", _md_table(res, "ann_per_class", paths.CLASSES),
           "", "## Images per class", "", _md_table(res, "img_per_class", paths.CLASSES),
           "", "## Images per source", "", _md_table(res, "img_per_source", SOURCES),
           "", "## Boxes per source", "", _md_table(res, "ann_per_source", SOURCES)]
    (out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf8")
    return res
