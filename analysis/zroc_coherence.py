#!/usr/bin/env python3
"""Do a cell's stages lie on ONE z-ROC? The assumption-free form of the claim.

WHY (2026-09-11). The pre-submission sweep raised the sharpest objection we have:
we invoke unequal variance to reverse the anchor's corrected-ceiling verdict, but
we lean on single-point d' to carry the PRIMARY claim. A reviewer will ask why d'
is unreliable enough to overturn one result and reliable enough to support the
other. An independent report (arXiv 2603.14893) finds z-ROC slopes 0.52-0.84 in
LLMs with instruct models MORE extreme than base, so tuning may move the variance
ratio itself -- in which case d' across stages is not even a constant quantity.

This test sidesteps the whole argument. "Only the criterion moves" means the two
evidence distributions are FIXED and the threshold slides. Geometrically, that
means every stage of a cell is a different operating point ON THE SAME ROC CURVE.
So fit z(H) = a + b*z(FA) across a cell's six stages and look at the residuals:

  high R^2  -> the points slide ALONG one curve: distributions fixed, criterion
               moving. This is the primary claim, with NO equal-variance
               assumption anywhere -- b is estimated, not assumed to be 1.
  low R^2   -> the points scatter OFF any single curve: the underlying
               distributions are changing, so the arm is not merely re-thresholding.

d_a = sqrt(2/(1+b^2)) * a is the unequal-variance sensitivity index, reported
alongside because it is the quantity d' approximates only when b = 1.

Caveat, stated because it limits the reading: six points per fit is modest, and a
high R^2 is CONSISTENT with fixed distributions rather than proof of them. A low
R^2, though, does reject them for that arm.
"""
import json, math, os, statistics as st, sys

def z(p):
    a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00]
    b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01]
    c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00]
    d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00]
    pl=0.02425
    if p<pl:
        q=math.sqrt(-2*math.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p>1-pl:
        q=math.sqrt(-2*math.log(1-p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q=p-0.5; r=q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def fit(pts):
    n=len(pts); mx=sum(x for x,_ in pts)/n; my=sum(y for _,y in pts)/n
    sxx=sum((x-mx)**2 for x,_ in pts); sxy=sum((x-mx)*(y-my) for x,y in pts)
    syy=sum((y-my)**2 for _,y in pts)
    if sxx<=0 or syy<=0: return None
    b=sxy/sxx; a=my-b*mx
    r2=1-sum((y-(a+b*x))**2 for x,y in pts)/syy
    return a, b, r2, math.sqrt(2/(1+b*b))*a

def main():
    here=os.path.dirname(os.path.abspath(__file__))
    agg=json.load(open(os.path.join(here,"readout","fs_aggregate.json")))
    arms=agg["backbones"]["llava15"]["arms"]; out={}
    print(f"{'arm':8} {'cell':9} {'R^2':>7} {'slope b':>8} {'d_a':>7}")
    print("-"*44)
    for arm in ("seq","anchor","joint"):
        rows=[]
        for k,cell in sorted(arms.items()):
            if k.split("|")[0]!=arm: continue
            pts=[]
            for s in sorted((x for x in cell if x.isdigit()), key=int):
                p=cell[s].get("pope")
                if p and 0<p["H"]<1 and 0<p["FA"]<1: pts.append((z(p["FA"]), z(p["H"])))
            if len(pts)<4: continue
            f=fit(pts)
            if not f: continue
            a,b,r2,da=f; rows.append((k.split("|",1)[1].replace("|","/"),r2,b,da))
            print(f"{arm:8} {rows[-1][0]:9} {r2:7.4f} {b:8.3f} {da:7.3f}")
        if rows:
            out[arm]={"n":len(rows),
                      "mean_r2":round(st.mean(r[1] for r in rows),4),
                      "min_r2":round(min(r[1] for r in rows),4),
                      "mean_d_a":round(st.mean(r[3] for r in rows),4),
                      "sd_d_a":round(st.pstdev([r[3] for r in rows])*math.sqrt(len(rows)/max(1,len(rows)-1)),4),
                      "per_cell":[{"cell":r[0],"r2":round(r[1],4),"slope":round(r[2],4),"d_a":round(r[3],4)} for r in rows]}
            print(f"{arm:8} {'MEAN':9} {out[arm]['mean_r2']:7.4f} {'':8} {out[arm]['mean_d_a']:7.3f}\n")
    json.dump(out, open(os.path.join(here,"readout","zroc_coherence.json"),"w"), indent=2)
    print("wrote readout/zroc_coherence.json")
    return 0

if __name__=="__main__": sys.exit(main())
