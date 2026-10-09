"""Test 1 follow-up: response predicted by the LIM of each run (lag 1 yr, its own segment, boxes of fdt_boxes.py) to
the forcing f calibrated on 1240 (first 30 yr of the small steps). Prints e, U1, M1 per W m^-2 at 30/100/300/1000 yr
and the ratio e/M1 of the predicted response at 300 and 1000 yr, against the forced transients."""
import sys, numpy as np
sys.argv = ['x', '30', '1']
src = open('prototyping/plasim_reduction/fdt_boxes.py').read().split("L = lim(W, TAU, SCALE)")[0]; exec(src)
f = fit_f(lim(W, TAU, SCALE), obs)
print('f:', ' '.join(f'{n}={x:+.3g}' for n, x in zip(NAMES, f)))
for lab in ['1245', '1242p5', '1240', '1237p5', '1235', '1233p75', '1232p5']:
    yrs, X = boxes(lab); Wk = window(lab, yrs, X); Lk = lim(Wk, 1, SCALE)
    ev = sorted(np.linalg.eigvals(Lk), key=lambda z: -z.real)
    r = {T: G(Lk, T) @ f for T in (30, 100, 300, 1000)}
    print(f'{lab:8s} slowest {ev[0].real:+.4f} {ev[1].real:+.4f} | ' + ' '.join(f't={T}: e {r[T][0]:+.3f} U1 {r[T][3]:+.3f} M1 {r[T][5]:+.3f}' for T in r)
          + f' | e/M1 at 300: {r[300][0]/r[300][5]:+.1f}, 1000: {r[1000][0]/r[1000][5]:+.1f}')
