"""Full-study dependency probe: UCIT schema, Qwen2.5-VL feasibility, AMBER data."""
import json, os, sys, traceback

report = {}

def sec(name):
    print(f"\n===== {name} =====", flush=True)

sec("UCIT")
try:
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
    cfgs = get_dataset_config_names("MLLM-CL/UCIT")
    report["ucit_configs"] = cfgs
    print("configs:", cfgs, flush=True)
    for cfg in cfgs[:10]:
        try:
            splits = get_dataset_split_names("MLLM-CL/UCIT", cfg)
            info = {}
            for sp in splits:
                d = load_dataset("MLLM-CL/UCIT", cfg, split=f"{sp}[:2]")
                info[sp] = {"columns": d.column_names}
                ex = d[0]
                info[sp]["example"] = {k: (str(type(v).__name__) if not isinstance(v, (str, int, float)) else str(v)[:120]) for k, v in ex.items()}
            # full sizes without downloading all: builder info
            from datasets import load_dataset_builder
            b = load_dataset_builder("MLLM-CL/UCIT", cfg)
            sizes = {k: v.num_examples for k, v in (b.info.splits or {}).items()}
            print(cfg, "splits:", splits, "sizes:", sizes, flush=True)
            print("   cols:", {sp: v["columns"] for sp, v in info.items()}, flush=True)
            print("   ex:", json.dumps(info[splits[0]]["example"])[:500], flush=True)
        except Exception as e:
            print(cfg, "ERR:", repr(e)[:200], flush=True)
except Exception:
    traceback.print_exc()

sec("QWEN2.5-VL")
try:
    import torch
    from transformers import AutoProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    print("processor ok; image_processor:", type(proc.image_processor).__name__, flush=True)
    print("min/max pixels:", getattr(proc.image_processor, "min_pixels", None), getattr(proc.image_processor, "max_pixels", None), flush=True)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Is there a dog in the image?"}]}]
    tmpl = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    print("CHAT TEMPLATE RENDER:", json.dumps(tmpl), flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16).cuda()
    print("loaded; VRAM GB:", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
    names = [n for n, _ in model.named_modules()]
    lm_layers = [n for n in names if ".layers.0." in n and n.endswith("_proj")][:12]
    vis = [n for n in names if n.startswith("visual.")][:6]
    print("LM layer-0 proj modules:", lm_layers, flush=True)
    print("visual module prefixes:", vis, flush=True)
    from PIL import Image
    import numpy as np
    img = Image.fromarray((np.random.rand(336, 336, 3) * 255).astype("uint8"))
    proc.image_processor.max_pixels = 768 * 28 * 28
    enc = proc(text=[tmpl], images=[img], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**enc)
    print("forward ok; logits", tuple(out.logits.shape), "peak VRAM GB:",
          round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
    gen = model.generate(**enc, max_new_tokens=8, do_sample=False)
    print("gen ok:", proc.tokenizer.decode(gen[0][enc['input_ids'].shape[1]:], skip_special_tokens=True), flush=True)
except Exception:
    traceback.print_exc()

sec("AMBER")
try:
    import urllib.request
    for url in ["https://raw.githubusercontent.com/junyangwang0410/AMBER/master/data/query/query_generative.json",
                "https://raw.githubusercontent.com/junyangwang0410/AMBER/master/data/annotations.json"]:
        try:
            data = urllib.request.urlopen(url, timeout=30).read()
            print("OK", url.split("/")[-1], len(data), "bytes", flush=True)
        except Exception as e:
            print("FAIL", url, repr(e)[:120], flush=True)
    from huggingface_hub import HfApi
    hits = HfApi().list_datasets(search="AMBER", limit=20)
    print("HF dataset candidates:", [d.id for d in hits], flush=True)
except Exception:
    traceback.print_exc()

sec("DISK")
os.system("df -h /home | tail -1")
print("PROBE_DONE", flush=True)
