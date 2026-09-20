"""CHAIR metrics over generated detailed captions of COCO val2014 images.

Vendored synonym list (adapted from the standard CHAIR implementation,
Rohrbach et al. 2018). The list is fixed in this repo so every checkpoint is
scored identically; internal validity does not depend on matching the
canonical list word-for-word (documented deviation).

GT object set per image = instance annotations UNION objects extracted from
the 5 GT captions with the same extractor (standard practice).

Outputs: per-image JSONL (for bootstrap) + summary json/csv row.
CHAIR_s: fraction of captions with >=1 hallucinated object.
CHAIR_i: hallucinated unique-object mentions / all unique-object mentions.
"""
import argparse
import csv
import json
import re

SYN = {
    "person": ["person", "girl", "boy", "man", "woman", "kid", "child", "chef", "baker",
               "people", "adult", "rider", "children", "baby", "worker", "passenger",
               "biker", "policeman", "cop", "officer", "lady", "cowboy", "bride", "groom",
               "male", "female", "guy", "traveler", "mother", "father", "gentleman",
               "player", "skier", "snowboarder", "skater", "skateboarder", "surfer",
               "individual", "human", "men", "women", "couple", "toddler", "pedestrian"],
    "bicycle": ["bicycle", "bike", "unicycle", "minibike", "trike"],
    "car": ["car", "automobile", "van", "minivan", "sedan", "suv", "hatchback", "cab",
            "jeep", "coupe", "taxicab", "limo", "taxi"],
    "motorcycle": ["motorcycle", "motorbike", "scooter", "moped"],
    "airplane": ["airplane", "jetliner", "plane", "monoplane", "aircraft", "jet",
                 "airbus", "biplane", "seaplane"],
    "bus": ["bus", "minibus", "trolley"],
    "train": ["train", "locomotive", "tramway", "caboose"],
    "truck": ["truck", "pickup", "lorry", "hauler", "firetruck", "semi"],
    "boat": ["boat", "ship", "liner", "sailboat", "motorboat", "dinghy", "powerboat",
             "speedboat", "canoe", "kayak", "yacht", "catamaran", "gondola", "ferry",
             "houseboat", "vessel", "rowboat", "raft"],
    "traffic light": ["traffic light", "street light", "streetlight", "traffic signal",
                      "stop light", "stoplight", "streetlamp"],
    "fire hydrant": ["fire hydrant", "hydrant"],
    "stop sign": ["stop sign"],
    "parking meter": ["parking meter"],
    "bench": ["bench"],
    "bird": ["bird", "ostrich", "owl", "seagull", "goose", "geese", "duck", "parakeet",
             "falcon", "robin", "pelican", "waterfowl", "heron", "hummingbird", "mallard",
             "finch", "pigeon", "sparrow", "seabird", "osprey", "blackbird", "fowl",
             "shorebird", "woodpecker", "egret", "chickadee", "quail", "bluebird",
             "kingfisher", "buzzard", "willet", "gull", "swan", "bluejay", "flamingo",
             "cormorant", "parrot", "loon", "gosling", "waterbird", "pheasant", "rooster",
             "sandpiper", "crow", "raven", "turkey", "oriole", "cowbird", "warbler",
             "magpie", "peacock", "cockatiel", "lorikeet", "puffin", "vulture", "condor",
             "macaw", "peafowl", "eagle", "lark", "hen"],
    "cat": ["cat", "kitten", "feline", "tabby"],
    "dog": ["dog", "puppy", "beagle", "pup", "chihuahua", "schnauzer", "dachshund",
            "rottweiler", "canine", "pitbull", "collie", "pug", "terrier", "poodle",
            "labrador", "doggie", "doberman", "mutt", "doggy", "spaniel", "bulldog",
            "sheepdog", "weimaraner", "corgi", "greyhound", "retriever", "hound",
            "whippet", "husky"],
    "horse": ["horse", "stallion", "pony", "mare", "foal", "palomino", "mustang",
              "clydesdale", "bronc", "bronco"],
    "sheep": ["sheep", "lamb", "ram", "goat", "ewe"],
    "cow": ["cow", "cattle", "oxen", "ox", "calf", "holstein", "heifer", "buffalo",
            "bull", "zebu", "bison"],
    "elephant": ["elephant"],
    "bear": ["bear", "panda"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "backpack": ["backpack", "knapsack", "rucksack"],
    "umbrella": ["umbrella", "parasol"],
    "handbag": ["handbag", "wallet", "purse", "briefcase"],
    "tie": ["tie", "bow tie", "necktie"],
    "suitcase": ["suitcase", "suit case", "luggage"],
    "frisbee": ["frisbee"],
    "skis": ["skis", "ski"],
    "snowboard": ["snowboard"],
    "sports ball": ["sports ball", "ball", "soccer ball", "basketball", "volleyball"],
    "kite": ["kite"],
    "baseball bat": ["baseball bat", "bat"],
    "baseball glove": ["baseball glove", "mitt"],
    "skateboard": ["skateboard"],
    "surfboard": ["surfboard", "longboard", "skimboard", "shortboard", "wakeboard"],
    "tennis racket": ["tennis racket", "racket", "racquet"],
    "bottle": ["bottle"],
    "wine glass": ["wine glass", "wineglass"],
    "cup": ["cup", "mug"],
    "fork": ["fork"],
    "knife": ["knife", "pocketknife", "knives"],
    "spoon": ["spoon"],
    "bowl": ["bowl"],
    "banana": ["banana"],
    "apple": ["apple"],
    "sandwich": ["sandwich", "burger", "cheeseburger", "hamburger"],
    "orange": ["orange"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot"],
    "hot dog": ["hot dog", "hotdog"],
    "pizza": ["pizza"],
    "donut": ["donut", "doughnut", "bagel"],
    "cake": ["cake", "cheesecake", "cupcake", "shortcake", "pancake"],
    "chair": ["chair", "stool", "armchair"],
    "couch": ["couch", "sofa", "recliner", "futon", "loveseat", "settee"],
    "potted plant": ["potted plant", "houseplant"],
    "bed": ["bed", "mattress"],
    "dining table": ["dining table", "table", "desk"],
    "toilet": ["toilet", "urinal", "commode", "lavatory", "potty"],
    "tv": ["tv", "television", "monitor"],
    "laptop": ["laptop", "computer", "notebook", "netbook", "chromebook", "macbook"],
    "mouse": ["mouse", "mice"],
    "remote": ["remote", "remote control"],
    "keyboard": ["keyboard"],
    "cell phone": ["cell phone", "cellphone", "phone", "smartphone", "iphone",
                   "mobile phone", "telephone"],
    "microwave": ["microwave"],
    "oven": ["oven", "stove", "stovetop"],
    "toaster": ["toaster"],
    "sink": ["sink"],
    "refrigerator": ["refrigerator", "fridge", "freezer"],
    "book": ["book"],
    "clock": ["clock"],
    "vase": ["vase"],
    "scissors": ["scissors", "shears"],
    "teddy bear": ["teddy bear", "teddybear"],
    "hair drier": ["hair drier", "hairdryer", "hair dryer", "blow dryer"],
    "toothbrush": ["toothbrush"],
}

WORD2CAT = {}
for cat, words in SYN.items():
    for w in words:
        WORD2CAT[w] = cat

IRREGULAR = {"people": "people", "men": "men", "women": "women", "children": "children",
             "geese": "geese", "mice": "mice", "knives": "knives"}


def _norm_token(w):
    if w in WORD2CAT or w in IRREGULAR:
        return w
    if w.endswith("es") and w[:-2] in WORD2CAT:
        return w[:-2]
    if w.endswith("s") and w[:-1] in WORD2CAT:
        return w[:-1]
    return w


def extract_object_mentions(text):
    """Every mention (official CHAIR_i is per-mention, not per-unique-object;
    audit finding M1). Bigrams are also tried with the last word singularized
    so plural compounds ('traffic lights', 'stop signs') match (M2)."""
    toks = [_norm_token(w) for w in re.findall(r"[a-z]+", text.lower())]
    mentions = []
    i = 0
    while i < len(toks):
        if i + 1 < len(toks):
            for second in (toks[i + 1], _norm_token(toks[i + 1])):
                bigram = toks[i] + " " + second
                if bigram in WORD2CAT:
                    mentions.append(WORD2CAT[bigram])
                    i += 2
                    break
            else:
                if toks[i] in WORD2CAT:
                    mentions.append(WORD2CAT[toks[i]])
                i += 1
            continue
        if toks[i] in WORD2CAT:
            mentions.append(WORD2CAT[toks[i]])
        i += 1
    return mentions


def extract_objects(text):
    return set(extract_object_mentions(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--coco_gt", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    gt = json.load(open(args.coco_gt))

    per_img = []
    n_caps = n_hal_caps = n_mentions = n_hal_mentions = 0
    tot_len = tot_trunc = 0
    for line in open(args.gen):
        r = json.loads(line)
        cid = str(r["coco_id"])
        gt_objects = set(gt[cid]["objects"])
        for c in gt[cid]["captions"]:
            gt_objects |= extract_objects(c)
        mentions = extract_object_mentions(r["output"])
        mentioned = set(mentions)
        hal = sorted(mentioned - gt_objects)
        hal_mentions = [m for m in mentions if m not in gt_objects]
        n_caps += 1
        n_mentions += len(mentions)
        n_hal_mentions += len(hal_mentions)
        if hal:
            n_hal_caps += 1
        tot_len += r["n_new_tokens"]
        tot_trunc += int(r["truncated"])
        per_img.append({"id": r["id"], "coco_id": cid,
                        "mentioned": sorted(mentioned), "hallucinated": hal,
                        "n_mentions": len(mentions),
                        "n_hal_mentions": len(hal_mentions),
                        "n_new_tokens": r["n_new_tokens"], "truncated": r["truncated"]})

    summary = {
        "ckpt": args.ckpt, "n_captions": n_caps,
        "chair_s": round(n_hal_caps / max(1, n_caps), 4),
        "chair_i": round(n_hal_mentions / max(1, n_mentions), 4),
        "mentions_per_caption": round(n_mentions / max(1, n_caps), 3),
        "mean_new_tokens": round(tot_len / max(1, n_caps), 1),
        "truncation_rate": round(tot_trunc / max(1, n_caps), 4),
    }
    with open(args.out_prefix + "_per_image.jsonl", "w") as f:
        for row in per_img:
            f.write(json.dumps(row) + "\n")
    with open(args.out_prefix + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(args.out_prefix + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
