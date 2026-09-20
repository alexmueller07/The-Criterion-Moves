"""AMBER generative metrics (CHAIR, Cover, Hal, Cog) — vendored, judge-free scorer.

Faithful reimplementation of the *generative* branch of the official AMBER
evaluator, with the two heavyweight dependencies made optional/vendored so the
scorer runs identically on the cluster with a bare Python.

Provenance (all read at source 2026-09-02, pinned commit
534babf6bbfcce2e735c26289dedfb21cef3c939 of github.com/junyangwang0410/AMBER):
  https://raw.githubusercontent.com/junyangwang0410/AMBER/534babf6bbfcce2e735c26289dedfb21cef3c939/inference.py
  .../data/annotations.json  .../data/relation.json  .../data/safe_words.txt
  .../data/metrics.txt       .../data/query/query_generative.json
Data files are NOT vendored here — download them with pilot/fetch_amber_assets.py
and pass the directory as --amber_data.

Official scoring mechanics reproduced (inference.py, generative branch):
  1. extract_nouns: nltk.word_tokenize -> nltk.pos_tag -> keep POS 'NN*' ->
     WordNetLemmatizer().lemmatize(word) (pos='n', case-sensitive, NOT lowercased).
  2. Keep only nouns inside the AMBER vocabulary (relation.json keys+values;
     418 words). Out-of-vocabulary nouns are ignored entirely (they are never
     counted as hallucinations).
  3. Per image: safe_words = concat(association[w] for w in truth) + truth;
     ha_words likewise from the annotated plausible-but-absent 'hallu' list.
  4. Per kept noun, in order: (a) global safe word (safe_words.txt) -> skip
     (still counted in the CHAIR denominator, and it marks NO cover slot);
     (b) exact match in safe_words -> mark the matched truth object covered;
     (c) exact match in ha_words -> mark that hallu target mentioned (Cog) and
     FALL THROUGH; (d) first ha_word with spaCy en_core_web_lg similarity>0.8
     -> mark its hallu target; (e) first safe_word with similarity>0.8 -> mark
     cover, not hallucinated; else the noun is a hallucinated mention.
  5. Metrics over all captions, with denominators initialised at 0.001 (from
     the official data/metrics.txt — reproduced here; negligible at n=1004,
     visible at tiny n):
       CHAIR = 100 * hallucinated mentions / kept mentions   (per-mention)
       Cover = 100 * covered truth slots / truth slots
       Hal   = 100 * captions with >=1 hallucinated mention / captions
       Cog   = 100 * mentioned hallu-target slots / hallu-target slots
     all rounded to 1 decimal exactly as the official print statements do.

Vendored spaCy similarity ("synonym_checker" in the output JSON):
  Every similarity call the official scorer can make on the official data is
  between two words of the closed 418-word vocabulary (verified: every
  truth/hallu word is a relation.json key, every expansion a relation.json
  value, and mentions are vocabulary-filtered first). The full vocab x vocab
  relation sim(a,b) > 0.8 was therefore precomputed ONCE with the exact
  official call (spacy en_core_web_lg 3.8.0, nlp(w1).similarity(nlp(w2))) and
  is embedded below — bit-identical official semantics, no spaCy at runtime.
  Pairs outside the table's domain (possible only with non-official annotation
  files, e.g. the test fixture) score as not-similar (exact match only).
  The official --similarity_score knob is fixed at its default 0.8 here.
  Table contents: every word is similar to itself (spaCy Doc.similarity
  short-circuits orth-identical docs to 1.0 before touching vectors, so even
  the 13 zero-vector vocabulary typos self-match) plus exactly 15 symmetric
  near-synonym pairs (bathtub-tub, bicycle-bike, blueberry-strawberry,
  boy-girl, bracelet-earrings, bracelet-necklace, broccoli-cabbage,
  cash-money, cat-dog, couch-sofa, earrings-necklace, football-soccer,
  fridge-refrigerator, laptop-notebook, motorbike-motorcycle). NOTE: the
  official README does not pin the en_core_web_lg version; this table was
  computed with 3.8.0, and borderline pairs (cat-dog is 0.8017) could flip
  under other model releases — an ambiguity of the official protocol itself.

Lemmatizer ("lemmatizer" in the output JSON — never silently incomparable):
  - "nltk": used when nltk + its data (punkt, tagger, wordnet) import and pass
    a probe; byte-for-byte the official extract_nouns.
  - "fallback": deterministic stdlib path (--force_fallback forces it):
    regex word tokens (hyphenated compounds kept whole), lowercase, a vendored
    exceptions dict, then WordNet-style noun plural suffix rules validated
    against the loaded vocabulary. FALLBACK_EXCEPTIONS was tuned so that on
    every standard singular/plural inflection of all 418 vocabulary words the
    two paths produce vocabulary-identical results (verified empirically
    against nltk 3.10.3 / WordNet; 881 inflections, zero mismatches).
  Residual divergence risk between the two paths (document, do not mix paths
  within one comparison): (a) capitalized mentions — the official/nltk path is
  case-sensitive and DROPS 'Dog'/'Sofa' etc. (WordNet returns capitalized
  forms unchanged, so they miss the lowercase vocabulary), while the fallback
  lowercases and keeps them; (b) POS — the official path keeps only NN*-tagged
  tokens, the fallback keeps any token whose lemma is in the vocabulary, so
  vocabulary words used as verbs ('watch', 'lights') count only under the
  fallback; (c) tokenizer edge cases (contractions, punctuation-glued words).

Official-scorer quirks found while porting (reproduced faithfully, flagged so
nobody mistakes them for our bugs):
  * Case sensitivity as above; also the vocabulary entry 'TV' only matches the
    exact token 'TV' (lowercase 'tv' is dropped by the official path).
  * Global safe words skip cover marking too: a truth object that is also a
    global safe word can never be covered by its own name, and a hallu target
    that is one (e.g. 'sign', a hallu word for 150/1004 images) can never be
    Cog-credited by its own name — only via association words or similarity.
  * Exact matching scans the concatenated word list and credits the FIRST
    occurrence: a mention equal to truth object B that also appears in the
    association list of an earlier truth object A credits A, not B; duplicate
    truth entries (14/1004 images) leave their second copy forever uncovered.
  * A mention exactly matching a hallu word is Cog-credited AND still
    CHAIR-flagged as hallucinated (unless similarity>0.8 to a safe word) —
    Cog and CHAIR overlap by design.
  * WordNet lemmatization maps some vocabulary words out of the vocabulary:
    plural-form entries 'chopsticks','earrings','slippers','sunglasses'
    lemmatize to singulars absent from the vocabulary and are unmatchable by
    the official pipeline; 'vases'->'vas' and 'leaves'->'leaf' (the vocabulary
    itself uses 'leave') are dropped likewise; plurals of non-WordNet words
    ('paragliders','cooktops','streetlamps',...) stay inflated and are
    dropped. FALLBACK_EXCEPTIONS reproduces all of this in the fallback path.
  * Empty truth/hallu lists would make the official list[-0:] slicing grab the
    whole bookkeeping list (garbage); this never occurs in the official
    annotations (min truth 2, min hallu 3) — we raise instead of reproducing.
  * Zero-mention captions (e.g. empty/degenerate outputs) count as
    non-hallucinated for Hal and add nothing to the CHAIR denominator, so
    degenerate checkpoints look GOOD — n_zero_mention_captions and
    n_empty_outputs are emitted as audit columns; gate on them.

Input --gen JSONL rows: {id, amber_id, output, n_new_tokens, truncated}
(amber_id = official AMBER generative query id, 1..1004). Outputs, house style:
  <out_prefix>.json / .csv   one summary row: the four metrics + audit columns
                             (n, mean_new_tokens, truncation_rate, lemmatizer,
                             synonym_checker, data-file sha256s, raw counts).
  <out_prefix>_per_image.jsonl   per-caption rows for bootstrap.

Usage:
  python metrics_amber.py --gen gen.jsonl --amber_data amber_data/ \
      --out_prefix results/S2_amber --ckpt S2 [--force_fallback]
Fixture check (both paths must agree):
  python metrics_amber.py --gen pilot/tests/amber_fixture/gen.jsonl \
      --amber_data pilot/tests/amber_fixture --out_prefix /tmp/fx --ckpt FX
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Vendored spaCy en_core_web_lg similarity relation over the AMBER vocabulary.
# Generated once from the official call pattern (see module docstring):
#   sim(a,b) = spacy.load("en_core_web_lg")(a).similarity(nlp(b)) > 0.8
# A word maps to every vocabulary word (itself included, when its vector is
# nonzero) with similarity strictly greater than 0.8.
_SIM_JSON = r"""{"model":"en_core_web_lg","table":{"TV":["TV"],"air-conditioning":["air-conditioning"],"airport":["airport"],"alarm":["alarm"],"alcohol":["alcohol"],"antelope":["antelope"],"apple":["apple"],"armrest":["armrest"],"backpack":["backpack"],"baconic":["baconic"],"bag":["bag"],"baggage":["baggage"],"bail":["bail"],"ball":["ball"],"balloon":["balloon"],"bamboo":["bamboo"],"banana":["banana"],"bar":["bar"],"barbecue":["barbecue"],"barrier":["barrier"],"baseball":["baseball"],"basin":["basin"],"basket":["basket"],"basketball":["basketball"],"bat":["bat"],"bath":["bath"],"bathtub":["bathtub","tub"],"beach":["beach"],"bear":["bear"],"bed":["bed"],"bedsheet":["bedsheet"],"beer":["beer"],"bench":["bench"],"bicycle":["bicycle","bike"],"bike":["bicycle","bike"],"bill":["bill"],"bin":["bin"],"bird":["bird"],"blanket":["blanket"],"blueberry":["blueberry","strawberry"],"board":["board"],"boat":["boat"],"book":["book"],"bookshelf":["bookshelf"],"bottle":["bottle"],"bowl":["bowl"],"box":["box"],"boy":["boy","girl"],"bracelet":["bracelet","earrings","necklace"],"bread":["bread"],"bridge":["bridge"],"broccoli":["broccoli","cabbage"],"brush":["brush"],"bucket":["bucket"],"building":["building"],"buoy":["buoy"],"bus":["bus"],"bush":["bush"],"butterfly":["butterfly"],"cabbage":["broccoli","cabbage"],"cabinet":["cabinet"],"cable":["cable"],"cafe":["cafe"],"cage":["cage"],"cake":["cake"],"camel":["camel"],"camera":["camera"],"can":["can"],"candle":["candle"],"cann":["cann"],"car":["car"],"card":["card"],"carpet":["carpet"],"carrot":["carrot"],"cash":["cash","money"],"cat":["cat","dog"],"cathole":["cathole"],"ceiling":["ceiling"],"chair":["chair"],"charcoal":["charcoal"],"charger":["charger"],"cheese":["cheese"],"cherry":["cherry"],"chicken":["chicken"],"child":["child"],"chopsticks":["chopsticks"],"clock":["clock"],"closestool":["closestool"],"cloth":["cloth"],"cloud":["cloud"],"coconut":["coconut"],"coffee":["coffee"],"cola":["cola"],"collar":["collar"],"computer":["computer"],"cone":["cone"],"container":["container"],"controller":["controller"],"cooker":["cooker"],"cooktop":["cooktop"],"cord":["cord"],"core":["core"],"couch":["couch","sofa"],"court":["court"],"cow":["cow"],"cowpea":["cowpea"],"cream":["cream"],"crocodile":["crocodile"],"crutch":["crutch"],"cup":["cup"],"curtain":["curtain"],"cushion":["cushion"],"deer":["deer"],"desert":["desert"],"desk":["desk"],"dog":["cat","dog"],"doghole":["doghole"],"doll":["doll"],"door":["door"],"doughnut":["doughnut"],"dove":["dove"],"drain":["drain"],"drawing":["drawing"],"dresser":["dresser"],"drink":["drink"],"dryer":["dryer"],"duck":["duck"],"dustbin":["dustbin"],"e-book":["e-book"],"eagle":["eagle"],"earing":["earing"],"earphone":["earphone"],"earrings":["bracelet","earrings","necklace"],"ebook":["ebook"],"egg":["egg"],"electrombile":["electrombile"],"elephant":["elephant"],"extinguisher":["extinguisher"],"faucet":["faucet"],"fence":["fence"],"file":["file"],"fish":["fish"],"fishnet":["fishnet"],"flag":["flag"],"floor":["floor"],"flower":["flower"],"flowerpot":["flowerpot"],"football":["football","soccer"],"forest":["forest"],"fork":["fork"],"fox":["fox"],"fridge":["fridge","refrigerator"],"frisbee":["frisbee"],"ginger":["ginger"],"giraffe":["giraffe"],"girl":["boy","girl"],"glacier":["glacier"],"glass":["glass"],"glove":["glove"],"goal":["goal"],"goblet":["goblet"],"goose":["goose"],"grape":["grape"],"grapefruit":["grapefruit"],"grass":["grass"],"grill":["grill"],"ground":["ground"],"grove":["grove"],"guardrail":["guardrail"],"guitar":["guitar"],"handrail":["handrail"],"hanger":["hanger"],"hat":["hat"],"hearth":["hearth"],"hill":["hill"],"holder":["holder"],"horse":["horse"],"hose":["hose"],"hot":["hot"],"house":["house"],"hydrant":["hydrant"],"ice":["ice"],"individual":["individual"],"insect":["insect"],"island":["island"],"juice":["juice"],"juicer":["juicer"],"kennel":["kennel"],"kernel":["kernel"],"kettle":["kettle"],"keyboard":["keyboard"],"kid":["kid"],"kite":["kite"],"kiwi":["kiwi"],"kiwifruit":["kiwifruit"],"knife":["knife"],"lake":["lake"],"lamp":["lamp"],"laptop":["laptop","notebook"],"leave":["leave"],"lemon":["lemon"],"leopard":["leopard"],"light":["light"],"line":["line"],"lion":["lion"],"lock":["lock"],"lounge":["lounge"],"luggage":["luggage"],"magazine":["magazine"],"man":["man"],"manhole":["manhole"],"mat":["mat"],"mattress":["mattress"],"meat":["meat"],"melon":["melon"],"microphone":["microphone"],"microwave":["microwave"],"milk":["milk"],"mirror":["mirror"],"money":["cash","money"],"monitor":["monitor"],"monkey":["monkey"],"moon":["moon"],"motorbike":["motorbike","motorcycle"],"motorcycle":["motorbike","motorcycle"],"mountain":["mountain"],"mouse":["mouse"],"mousepad":["mousepad"],"muffler":["muffler"],"mushroom":["mushroom"],"napery":["napery"],"napkin":["napkin"],"necklace":["bracelet","earrings","necklace"],"necklet":["necklet"],"necktie":["necktie"],"net":["net"],"newspaper":["newspaper"],"note":["note"],"notebook":["laptop","notebook"],"oar":["oar"],"opener":["opener"],"orange":["orange"],"oven":["oven"],"pack":["pack"],"pad":["pad"],"paddle":["paddle"],"painting":["painting"],"pan":["pan"],"panda":["panda"],"paper":["paper"],"paraglider":["paraglider"],"parasail":["parasail"],"path":["path"],"peach":["peach"],"pear":["pear"],"pen":["pen"],"people":["people"],"person":["person"],"phone":["phone"],"piano":["piano"],"pier":["pier"],"pig":["pig"],"pigeon":["pigeon"],"pillow":["pillow"],"pineapple":["pineapple"],"pinwheel":["pinwheel"],"pipe":["pipe"],"pissoir":["pissoir"],"pizza":["pizza"],"plane":["plane"],"plate":["plate"],"plug":["plug"],"pole":["pole"],"pot":["pot"],"potato":["potato"],"pulp":["pulp"],"quilt":["quilt"],"rabbit":["rabbit"],"rack":["rack"],"racket":["racket"],"rail":["rail"],"railing":["railing"],"rainbow":["rainbow"],"range":["range"],"reef":["reef"],"refrigerator":["fridge","refrigerator"],"ribbon":["ribbon"],"rice":["rice"],"riff":["riff"],"ring":["ring"],"river":["river"],"road":["road"],"rock":["rock"],"rod":["rod"],"rope":["rope"],"rpoe":["rpoe"],"rug":["rug"],"rugby":["rugby"],"saddle":["saddle"],"sailing":["sailing"],"sand":["sand"],"sanitizer":["sanitizer"],"sauce":["sauce"],"sausage":["sausage"],"scarf":["scarf"],"scoon":["scoon"],"scoop":["scoop"],"screen":["screen"],"sea":["sea"],"seegull":["seegull"],"shampoo":["shampoo"],"sheep":["sheep"],"sheet":["sheet"],"shelf":["shelf"],"shell":["shell"],"ship":["ship"],"shoe":["shoe"],"shore":["shore"],"shovel":["shovel"],"showerhead":["showerhead"],"showerpuff":["showerpuff"],"shrimp":["shrimp"],"sign":["sign"],"signal":["signal"],"sink":["sink"],"skate":["skate"],"ski":["ski"],"sky":["sky"],"sled":["sled"],"slippers":["slippers"],"sliver":["sliver"],"snack":["snack"],"snow":["snow"],"snowboard":["snowboard"],"snowman":["snowman"],"soap":["soap"],"soccer":["football","soccer"],"socket":["socket"],"sofa":["couch","sofa"],"sound":["sound"],"spade":["spade"],"sponge":["sponge"],"spray":["spray"],"squirrel":["squirrel"],"stage":["stage"],"stair":["stair"],"staircase":["staircase"],"star":["star"],"steak":["steak"],"sticker":["sticker"],"stone":["stone"],"stool":["stool"],"stopcock":["stopcock"],"strawberry":["blueberry","strawberry"],"street":["street"],"streetlamp":["streetlamp"],"string":["string"],"sun":["sun"],"sunflower":["sunflower"],"sunglasses":["sunglasses"],"support":["support"],"surfboard":["surfboard"],"switch":["switch"],"table":["table"],"tablecloth":["tablecloth"],"tangerine":["tangerine"],"tank":["tank"],"tap":["tap"],"tape":["tape"],"telephone":["telephone"],"telescope":["telescope"],"television":["television"],"tennis":["tennis"],"tent":["tent"],"tie":["tie"],"tiger":["tiger"],"tinfoil":["tinfoil"],"tire":["tire"],"tissue":["tissue"],"toaster":["toaster"],"toilet":["toilet"],"tomato":["tomato"],"toothbrush":["toothbrush"],"toothpaste":["toothpaste"],"toothpick":["toothpick"],"tortoise":["tortoise"],"towel":["towel"],"toy":["toy"],"track":["track"],"train":["train"],"tree":["tree"],"trough":["trough"],"truck":["truck"],"tub":["bathtub","tub"],"turtle":["turtle"],"tussock":["tussock"],"umbrella":["umbrella"],"urinal":["urinal"],"vase":["vase"],"vehicle":["vehicle"],"vessel":["vessel"],"volleyball":["volleyball"],"wall":["wall"],"warp":["warp"],"watch":["watch"],"water":["water"],"waterfont":["waterfont"],"waterfront":["waterfront"],"watermelon":["watermelon"],"windmill":["windmill"],"window":["window"],"wine":["wine"],"wineglass":["wineglass"],"wire":["wire"],"woman":["woman"],"worm":["worm"],"yacht":["yacht"],"zebra":["zebra"]},"threshold":0.8,"version":"3.8.0","zero_vector_words":["baconic","cathole","closestool","cowpea","doghole","electrombile","necklet","pissoir","rpoe","scoon","seegull","showerpuff","waterfont"]}"""

_SIM_META = json.loads(_SIM_JSON)
SIM_TABLE = {w: frozenset(ws) for w, ws in _SIM_META["table"].items()}
SYNONYM_CHECKER = "vendored_table:{}-{}:thr{}".format(
    _SIM_META["model"], _SIM_META["version"], _SIM_META["threshold"])
_EMPTY = frozenset()


def check_synonyms_word(word1, word2):
    """Official: nlp(word1).similarity(nlp(word2)) > 0.8. Vendored lookup."""
    return word2 in SIM_TABLE.get(word1, _EMPTY)


# ---------------------------------------------------------------------------
# Fallback lemmatizer (see module docstring). Exceptions map lowercase token ->
# the lemma nltk's WordNetLemmatizer produces, for every vocabulary-relevant
# token where the plain suffix rules would disagree with WordNet.
FALLBACK_EXCEPTIONS = {
    "air-conditionings": "air-conditionings", "baconics": "baconics",
    "bedsheets": "bedsheets", "canns": "canns", "catholes": "catholes",
    "children": "child", "chopsticks": "chopstick",
    "chopstickses": "chopstickses", "chopstickss": "chopstickss",
    "closestools": "closestools", "cooktops": "cooktops",
    "dogholes": "dogholes", "e-books": "e-books", "earings": "earings",
    "earrings": "earring", "earringses": "earringses",
    "earringss": "earringss", "ebooks": "ebooks",
    "electrombiles": "electrombiles", "geese": "goose", "hots": "hots",
    "kiwifruits": "kiwifruits", "knives": "knife", "leaves": "leaf",
    "mice": "mouse", "paragliders": "paragliders", "pissoirs": "pissoirs",
    "rpoes": "rpoes", "sanitizers": "sanitizers", "scoons": "scoons",
    "seegulls": "seegulls", "showerpuffs": "showerpuffs",
    "showerpufves": "showerpufves", "slippers": "slipper",
    "slipperses": "slipperses", "slipperss": "slipperss",
    "streetlamps": "streetlamps", "sunglasses": "sunglass", "vases": "vas",
    "waterfonts": "waterfonts", "women": "woman",
}

# WordNet's noun suffix substitution rules, in the order tried by the fallback.
NOUN_RULES = [("ses", "s"), ("ves", "f"), ("xes", "x"), ("zes", "z"),
              ("ches", "ch"), ("shes", "sh"), ("ies", "y"), ("s", "")]

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")


def make_fallback_extractor(vocab):
    def extract(text):
        out = []
        for tok in _TOKEN_RE.findall(text):
            low = tok.lower()
            if low in FALLBACK_EXCEPTIONS:
                out.append(FALLBACK_EXCEPTIONS[low])
                continue
            if tok in vocab:      # exact vocab hit ('TV')
                out.append(tok)
                continue
            lemma = low
            if low not in vocab:
                for suf, rep in NOUN_RULES:
                    if low.endswith(suf) and len(low) > len(suf):
                        cand = low[:-len(suf)] + rep
                        if cand in vocab:
                            lemma = cand
                            break
            out.append(lemma)
        return out
    return extract


def make_nltk_extractor():
    """Byte-for-byte the official extract_nouns; raises if nltk/data missing."""
    import nltk
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    # probe: fails fast (instead of mid-run) when a data package is absent
    probe = nltk.pos_tag(nltk.word_tokenize("The dogs sat."))
    assert lemmatizer.lemmatize("dogs") == "dog" and probe

    def extract(text):
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        return [lemmatizer.lemmatize(word) for word, pos in tagged
                if pos.startswith("NN")]
    return extract


def resolve_extractor(vocab, force_fallback):
    if not force_fallback:
        try:
            return "nltk", make_nltk_extractor()
        except Exception as e:
            print(f"nltk unavailable ({type(e).__name__}: {e}); "
                  "using fallback lemmatizer", flush=True)
    return "fallback", make_fallback_extractor(vocab)


# ---------------------------------------------------------------------------
def score_caption(nouns, ann, association, global_safe, vocab):
    """Port of the official generative branch for one caption.

    Returns (kept_nouns, hallucinated_nouns, safe_tail, ha_tail) where the
    tails are the official safe_list[-safe_len:] / ha_list[-ha_len:] marks.
    """
    truth, hallu = ann["truth"], ann["hallu"]
    if not truth or not hallu:
        raise ValueError(
            f"annotation id {ann['id']} has empty truth/hallu; the official "
            "scorer's list[-0:] slicing is undefined here (never occurs in "
            "official data)")

    after_process_nouns = [n for n in nouns if n in vocab]

    safe_words, safe_list = [], []
    for idx, word in enumerate(truth):
        safe_words += association[word]
        safe_list += [idx] * len(association[word])
    ha_words, ha_list = [], []
    for idx, word in enumerate(hallu):
        ha_words += association[word]
        ha_list += [idx] * len(association[word])

    safe_words += truth
    safe_len = len(truth)
    safe_list += [0] * safe_len
    ha_words += hallu
    ha_len = len(hallu)
    ha_list += [0] * ha_len

    safe_flag_list = [0] * len(after_process_nouns)
    safe_off = len(safe_list) - safe_len
    ha_off = len(ha_list) - ha_len

    for idx, noun in enumerate(after_process_nouns):
        if noun in global_safe:
            continue

        if noun in safe_words:
            j = safe_words.index(noun)          # official: first occurrence
            if j < safe_off:
                safe_list[safe_list[j] + safe_off] = 1
            else:
                safe_list[j] = 1
            continue

        if noun in ha_words:
            j = ha_words.index(noun)
            if j < ha_off:
                ha_list[ha_list[j] + ha_off] = 1
            else:
                ha_list[j] = 1
            # official: NO continue — falls through to the similarity checks

        for j, check_word in enumerate(ha_words):
            if check_synonyms_word(noun, check_word):
                if j < ha_off:
                    ha_list[ha_list[j] + ha_off] = 1
                else:
                    ha_list[j] = 1
                break

        flag = False
        for j, check_word in enumerate(safe_words):
            if check_synonyms_word(noun, check_word):
                flag = True
                if j < safe_off:
                    safe_list[safe_list[j] + safe_off] = 1
                else:
                    safe_list[j] = 1
                break
        if flag:
            continue

        safe_flag_list[idx] = 1

    hallucinated = [n for n, f in zip(after_process_nouns, safe_flag_list) if f]
    return after_process_nouns, hallucinated, safe_list[-safe_len:], ha_list[-ha_len:]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True,
                    help="JSONL: {id, amber_id, output, n_new_tokens, truncated}")
    ap.add_argument("--amber_data", required=True,
                    help="dir with annotations.json, relation.json, safe_words.txt "
                         "(see fetch_amber_assets.py)")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--ckpt", required=True, help="checkpoint label, e.g. S2")
    ap.add_argument("--force_fallback", action="store_true",
                    help="use the vendored fallback lemmatizer even if nltk works")
    args = ap.parse_args()

    ann_path = os.path.join(args.amber_data, "annotations.json")
    rel_path = os.path.join(args.amber_data, "relation.json")
    safe_path = os.path.join(args.amber_data, "safe_words.txt")

    association = json.load(open(rel_path, encoding="utf-8"))
    ann_by_id = {}
    for a in json.load(open(ann_path, encoding="utf-8")):
        ann_by_id[a["id"]] = a
    # official reader semantics: line.split('\n')[0], no other stripping
    with open(safe_path, encoding="utf-8") as f:
        global_safe = [line.split("\n")[0] for line in f]

    vocab = set(association)
    for words in association.values():
        vocab.update(words)

    lemmatizer_name, extract = resolve_extractor(vocab, args.force_fallback)

    # official init (data/metrics.txt): denominators start at 0.001
    m = {"chair_score": 0, "chair_num": 0.001,
         "safe_cover_score": 0, "safe_cover_num": 0.001,
         "hallu_cover_score": 0, "hallu_cover_num": 0.001,
         "non_hallu_score": 0, "non_hallu_num": 0.001}
    counts = {"mentions": 0, "hal_mentions": 0, "truth_slots": 0,
              "truth_covered": 0, "hallu_slots": 0, "hallu_covered": 0,
              "hal_captions": 0, "zero_mention": 0, "empty_output": 0}

    per_img = []
    n = 0
    tot_len = tot_trunc = 0
    for lineno, line in enumerate(open(args.gen), 1):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        missing = [k for k in ("id", "amber_id", "output", "n_new_tokens",
                               "truncated") if k not in r]
        if missing:
            sys.exit(f"{args.gen}:{lineno}: missing keys {missing}")
        ann = ann_by_id.get(r["amber_id"])
        if ann is None or ann.get("type") != "generative":
            sys.exit(f"{args.gen}:{lineno}: amber_id {r['amber_id']} is not a "
                     "generative AMBER annotation")

        nouns = extract(r["output"])
        mentions, hallucinated, safe_tail, ha_tail = score_caption(
            nouns, ann, association, global_safe, vocab)

        m["chair_score"] += len(hallucinated)
        m["chair_num"] += len(mentions)
        m["safe_cover_score"] += sum(safe_tail)
        m["safe_cover_num"] += len(safe_tail)
        m["hallu_cover_score"] += sum(ha_tail)
        m["hallu_cover_num"] += len(ha_tail)
        if not hallucinated:
            m["non_hallu_score"] += 1
        m["non_hallu_num"] += 1

        counts["mentions"] += len(mentions)
        counts["hal_mentions"] += len(hallucinated)
        counts["truth_slots"] += len(safe_tail)
        counts["truth_covered"] += sum(safe_tail)
        counts["hallu_slots"] += len(ha_tail)
        counts["hallu_covered"] += sum(ha_tail)
        counts["hal_captions"] += int(bool(hallucinated))
        counts["zero_mention"] += int(not mentions)
        counts["empty_output"] += int(not r["output"].strip())

        n += 1
        tot_len += r["n_new_tokens"]
        tot_trunc += int(r["truncated"])
        per_img.append({
            "id": r["id"], "amber_id": r["amber_id"],
            "mentions": mentions, "hallucinated": hallucinated,
            "n_mentions": len(mentions), "n_hal_mentions": len(hallucinated),
            "n_truth": len(safe_tail), "n_truth_covered": sum(safe_tail),
            "n_hallu": len(ha_tail), "n_hallu_covered": sum(ha_tail),
            "hal": int(bool(hallucinated)),
            "n_new_tokens": r["n_new_tokens"], "truncated": r["truncated"],
        })

    if n == 0:
        sys.exit(f"{args.gen}: no rows")

    # official formulas + rounding (inference.py print block)
    summary = {
        "ckpt": args.ckpt, "n": n,
        "chair": round(m["chair_score"] / m["chair_num"] * 100, 1),
        "cover": round(m["safe_cover_score"] / m["safe_cover_num"] * 100, 1),
        "hal": round(100 - m["non_hallu_score"] / m["non_hallu_num"] * 100, 1),
        "cog": round(m["hallu_cover_score"] / m["hallu_cover_num"] * 100, 1),
        "n_mentions": counts["mentions"],
        "n_hal_mentions": counts["hal_mentions"],
        "n_truth_slots": counts["truth_slots"],
        "n_truth_covered": counts["truth_covered"],
        "n_hallu_slots": counts["hallu_slots"],
        "n_hallu_covered": counts["hallu_covered"],
        "n_hal_captions": counts["hal_captions"],
        "n_zero_mention_captions": counts["zero_mention"],
        "n_empty_outputs": counts["empty_output"],
        "mentions_per_caption": round(counts["mentions"] / n, 3),
        "mean_new_tokens": round(tot_len / n, 1),
        "truncation_rate": round(tot_trunc / n, 4),
        "lemmatizer": lemmatizer_name,
        "synonym_checker": SYNONYM_CHECKER,
        "annotations_sha256": sha256_file(ann_path),
        "relation_sha256": sha256_file(rel_path),
        "safe_words_sha256": sha256_file(safe_path),
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
