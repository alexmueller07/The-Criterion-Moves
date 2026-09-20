"""Generate a folder of NOISE images for a content-free probe set.

Design red-team M2 (design_notes/review_method_design.md): a probe built on COCO images
with the POPE template is the endpoint distribution minus the answer key. Under the paper's
own additive-bias claim the criterion offset is class-independent, so a probe on images that
carry NO object evidence at all (noise) should still expose the drift — either outcome is a
finding: works -> the criterion is an image-independent answer-policy shift (blind/noise
probes are valid, image-overlap objections dissolve); fails -> drift is evidence-conditional.

Writes N RGB images (default 600): half Gaussian noise, half uniform noise, at a fixed size,
seeded. Then build the probe with
  python method/build_probe_set.py --root <root> --images_dir <noise_dir> --out <root>/grounding/probe_noise.jsonl
"""
import argparse
import os
import random

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--size", type=int, default=336)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = random.Random(a.seed)
    for i in range(a.n):
        img = Image.new("RGB", (a.size, a.size))
        px = img.load()
        gaussian = i % 2 == 0
        for y in range(a.size):
            for x in range(a.size):
                if gaussian:
                    v = tuple(min(255, max(0, int(rng.gauss(128, 40)))) for _ in range(3))
                else:
                    v = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
                px[x, y] = v
        img.save(os.path.join(a.out_dir, f"noise_{i:04d}.jpg"), quality=92)
    print(f"[noise] wrote {a.n} {a.size}x{a.size} images -> {a.out_dir}")


if __name__ == "__main__":
    main()
