"""Is the anchor's 9x wider d_a scatter real geometry, or estimation noise?

Written 2026-09-20 in response to a reviewer objection, and it killed the claim
it was written to defend. Result 2 of an earlier draft read "the anchor distorts
the evidence geometry, it does not merely mis-place the threshold", resting on
sd(d_a) = 0.362 against the sequential arm's 0.040. But d_a is a function of the
slope b, which the same draft declares not identified for the anchor, and the
anchor's fits have half the z(FA) lever arm (0.212 vs 0.444) and 1.7x the
residual RMSE (0.032 vs 0.018). Both inflate the sampling variance of b_hat
before any real geometry enters.

The null below: every anchor cell comes from ONE fixed ROC. Simulate six checkpoints at
the anchor's own z(FA) positions, add residual noise at the anchor's own RMSE,
fit OLS, compute d_a. If sd(d_a) under this null reaches the observed 0.362,
Result 2 is an artifact of a short lever arm, not a property of the arm.
"""
import json, math, random, statistics as st
def z(p,n=4500):
    p=min(max(p,1/(2*n)),1-1/(2*n)); lo,hi=-10.,10.
    for _ in range(80):
        m=(lo+hi)/2
        if 0.5*(1+math.erf(m/math.sqrt(2)))<p: lo=m
        else: hi=m
    return (lo+hi)/2
def fit(xs,ys):
    mx,my=st.mean(xs),st.mean(ys); sxx=sum((x-mx)**2 for x in xs)
    b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sxx; a=my-b*mx
    ssr=sum((y-a-b*x)**2 for x,y in zip(xs,ys))
    return a,b,math.sqrt(ssr/len(xs)), math.sqrt(2/(1+b*b))*a

d=json.load(open("analysis/readout/fs_aggregate.json"))["backbones"]["llava15"]["arms"]
stats={}
for arm in ("seq","anchor"):
    xr,rmse,das,xs_all=[],[],[],[]
    for k,v in d.items():
        if not k.startswith(arm+"|"): continue
        xs=[z(v[str(s)]["pope"]["FA"]) for s in range(1,7)]
        ys=[z(v[str(s)]["pope"]["H"]) for s in range(1,7)]
        a,b,r,da=fit(xs,ys)
        xr.append(max(xs)-min(xs)); rmse.append(r); das.append(da); xs_all.append(xs)
    stats[arm]=dict(xrange=st.mean(xr), rmse=st.mean(rmse), sd_da=st.stdev(das),
                    mean_da=st.mean(das), xs=xs_all)
    print("%-7s  mean z(FA) range %.4f   mean residual RMSE %.4f   d_a mean %.3f  sd %.4f"
          % (arm, stats[arm]["xrange"], stats[arm]["rmse"], stats[arm]["mean_da"], stats[arm]["sd_da"]))

print("\nNull: one fixed ROC (a,b from the pooled anchor fit), sampled at each anchor")
print("cell's OWN z(FA) positions, noise at that cell's OWN residual RMSE.\n")
rng=random.Random(20260920)
for arm in ("anchor","seq"):
    # pooled fit for this arm supplies the single true ROC
    allx=[x for xs in stats[arm]["xs"] for x in xs]
    ally=[]
    for k,v in d.items():
        if k.startswith(arm+"|"): ally += [z(v[str(s)]["pope"]["H"]) for s in range(1,7)]
    A,B,_,_=fit(allx,ally)
    sds=[]
    for _ in range(20000):
        das=[]
        for xs in stats[arm]["xs"]:
            ys=[A+B*x+rng.gauss(0,stats[arm]["rmse"]) for x in xs]
            das.append(fit(xs,ys)[3])
        sds.append(st.stdev(das))
    sds.sort()
    obs=stats[arm]["sd_da"]
    p=(sum(1 for v in sds if v>=obs)+1)/(len(sds)+1)
    print("%-7s  true ROC a=%.3f b=%.3f | simulated sd(d_a): median %.4f, 99th %.4f, max %.4f"
          % (arm,A,B,sds[len(sds)//2],sds[int(.99*len(sds))],sds[-1]))
    print("         observed sd(d_a) = %.4f   ->  P(null >= observed) = %.4f\n" % (obs,p))
