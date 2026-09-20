"""Independent re-derivation of the z-ROC model comparison.
M0: one binormal ROC per cell, z(H_k) = a + b*u_k, F_k = Phi(u_k)  -> 8 params (a, b, u_1..u_6)
M1: saturated, H_k and F_k free                                      -> 12 params
G2 = 2(LL_M1 - LL_M0), df 4; AIC prefers M0 iff G2 < 2*(12-8) = 8.
Binomial likelihood on counts reconstructed from H, FA and n_gt_yes/n_gt_no."""
import json, math
Phi=lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
def xlogy(k,p): return 0.0 if k==0 else k*math.log(p)
def ll(k,n,p):
    p=min(max(p,1e-12),1-1e-12); return xlogy(k,p)+xlogy(n-k,1-p)
def golden(f,lo,hi,it=80):
    g=(math.sqrt(5)-1)/2; a,b=lo,hi; c=b-g*(b-a); d=a+g*(b-a); fc,fd=f(c),f(d)
    for _ in range(it):
        if fc>fd: b,d,fd=d,c,fc; c=b-g*(b-a); fc=f(c)
        else: a,c,fc=c,d,fd; d=a+g*(b-a); fd=f(d)
    return (a+b)/2
def profile(ab,stages):
    a,b=ab; tot=0.0
    for (h,ny,fa,nn) in stages:
        obj=lambda u: ll(fa,nn,Phi(u))+ll(h,ny,Phi(a+b*u))
        u=golden(obj,-4.0,1.5); tot+=obj(u)
    return tot
def nelder(f,x0,step=(0.1,0.05),it=400):
    pts=[list(x0),[x0[0]+step[0],x0[1]],[x0[0],x0[1]+step[1]]]; vals=[f(p) for p in pts]
    for _ in range(it):
        o=sorted(range(3),key=lambda i:vals[i]); pts=[pts[i] for i in o]; vals=[vals[i] for i in o]
        c=[(pts[0][j]+pts[1][j])/2 for j in range(2)]
        r=[c[j]+(c[j]-pts[2][j]) for j in range(2)]; fr=f(r)
        if fr<vals[0]:
            e=[c[j]+2*(c[j]-pts[2][j]) for j in range(2)]; fe=f(e)
            pts[2],vals[2]=(e,fe) if fe<fr else (r,fr)
        elif fr<vals[1]: pts[2],vals[2]=r,fr
        else:
            k=[c[j]+0.5*(pts[2][j]-c[j]) for j in range(2)]; fk=f(k)
            if fk<vals[2]: pts[2],vals[2]=k,fk
            else:
                for i in (1,2): pts[i]=[pts[0][j]+0.5*(pts[i][j]-pts[0][j]) for j in range(2)]; vals[i]=f(pts[i])
    i=min(range(3),key=lambda i:vals[i]); return pts[i],vals[i]
def zinv(p):
    lo,hi=-8,8
    for _ in range(90):
        m=(lo+hi)/2
        if Phi(m)<p: lo=m
        else: hi=m
    return (lo+hi)/2
arms=json.load(open("analysis/readout/fs_aggregate.json"))["backbones"]["llava15"]["arms"]
res={}
for key,v in sorted(arms.items()):
    if not all(str(s) in v and "pope" in v[str(s)] for s in range(1,7)): continue
    st=[]; sat=0.0; xs=[]; ys=[]
    for s in range(1,7):
        p=v[str(s)]["pope"]; ny,nn=p["n_gt_yes"],p["n_gt_no"]
        h=round(p["H"]*ny); fa=round(p["FA"]*nn); st.append((h,ny,fa,nn))
        sat+=ll(h,ny,h/ny)+ll(fa,nn,fa/nn); xs.append(zinv(fa/nn)); ys.append(zinv(h/ny))
    n=6; mx=sum(xs)/n; my=sum(ys)/n
    b0=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs); a0=my-b0*mx
    (a,b),neg=nelder(lambda ab:-profile(ab,st),(a0,b0))
    G2=2*(sat-(-neg)); res.setdefault(key.split("|")[0],[]).append((key,G2,a,b))
for arm,rows in res.items():
    g=[r[1] for r in rows]
    print("%-7s n=%d  mean G2 %.3f   AIC prefers one ROC (G2<8): %d/%d   slopes b %.2f..%.2f" %
          (arm,len(rows),sum(g)/len(g),sum(1 for x in g if x<8),len(g),min(r[3] for r in rows),max(r[3] for r in rows)))
print("\npaper: seq mean G2 2.750, AIC 9/9 | anchor 5.827, 6/9 | joint AIC 2/2")
