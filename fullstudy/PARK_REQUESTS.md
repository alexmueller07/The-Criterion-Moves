# Park's requests (2026-09-03) — tracking

Added to the full-study queue. Status updated as each lands.

1. **Famous external hallucination benchmarks** with clear sequential-vs-joint data.
   - Adding judge-free benchmarks to the endpoint battery: AMBER (done), Object HalBench
     (CHAIR-scored), MME-Hallucination (yes/no acc+). Judge-bound (MMHal-Bench, HallusionBench)
     documented as run-if-API-provisioned. [DONE: Object HalBench + MME-Hall scorers built,
     fixture+source-verified, wired into sb_fs_bench.sbatch endpoint job; AMBER-generative if
     images resolve; AMBER-discriminative scorer done but its prompt-builder is a small open gap.]
   - **AMBER UPDATE 2026-09-03:** AMBER images could NOT be sourced (both HF candidates absent from
     the Hub); AMBER (generative + discriminative) is dropped from the metric set. Object HalBench
     + MME-Hallucination + POPE + CHAIR carry the famous-benchmark story. If an AMBER image mirror
     turns up later it slots back in via the guarded path.
   - **HONESTY FLAG for Alex:** our own finding is that the naive "sequential worse than joint"
     endpoint gap is largely an ARTIFACT of criterion drift, not genuine grounding loss (that is
     the paper's headline). So on famous benchmarks we will show the endpoint gap AND its
     decomposition — we must NOT present the raw gap as genuine degradation. Worth a word with
     Park so the framing is aligned; the benchmarks strengthen external validity of the
     *phenomenon + decomposition*, not of a "CL makes hallucination worse" claim we retracted.

2. **SOTA method comparison table** — anchor vs standard baselines on final hallucination.
   - Adding CL baselines EWC / O-LoRA / (LwF if it fits 24GB) as arms, faithful to MCITlib
     configs. [DONE: EWC/O-LoRA/LwF implemented, unit-tested, parity preserved; matrix Wave E.]
   - Table: methods {SEQ, JOINT, ER, EWC, O-LoRA, LwF, anchor-v1, anchor-v2} x endpoint hallucination
     {POPE-F1, CHAIR_i, Object-HalBench CHAIR, MME-Hall} + criterion c range + new-task plasticity.
     [table generator: build after first real results validate the results_fs schema]

3. **Motivation figure (Figure 1)** — one figure a reviewer immediately understands.
   - SDT schematic (criterion drifts, d' fixed) + real-data c/d' trajectory (SEQ vs anchor).
     [DONE: analysis/out/fig_motivation.pdf, committed; delivered to Alex.]

4. **General robustness** — the two-suite full study (mechanism on pilot suite; generalization
   on UCIT + Qwen 2nd backbone), 3 seeds x orderings, is the core of this.
