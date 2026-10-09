import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/states/'
LABELS = ['1245', '1242p5', '1240', '1237p5', '1235', '1235_new_IC', '1233p75', '1232p5']
amp = json.load(open('/Users/niccolo/.claude/jobs/96e03936/tmp/forced_amp.json'))
spacing = dict(zip(amp['labels'], amp['a']))
fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=False)
print(f"{'run':12s} {'yrs':>5s} | {'Ts mean':>8s} {'std':>5s} {'trend/kyr':>9s} {'half diff':>9s} | {'θ0-700 trend/kyr':>16s} {'half diff':>9s} {'std':>6s} | {'deep θ trend/kyr':>16s} {'half diff':>9s} | SA θ rows max |trend|/kyr")
for lab in LABELS:
    d = np.load(TMP + lab + '.npz'); y = d['years']; Tg = d['Ts'] @ d['weight']
    rel = y - y[0]
    def tr(v):
        ok = np.isfinite(v); p = np.polyfit(y[ok], v[ok], 1)[0] * 1000; h = len(v) // 2
        return p, np.nanmean(v[h:]) - np.nanmean(v[:h])
    t1 = tr(Tg); t2 = tr(d['theta_up_global']); t3 = tr(d['theta_deep_global'])
    sa = d['sa_ocean']; sa_tr = np.array([np.polyfit(y, sa[:, j], 1)[0] * 1000 for j in range(sa.shape[1])])
    print(f"{lab:12s} {len(y):5d} | {Tg.mean():8.2f} {Tg.std():5.2f} {t1[0]:+9.3f} {t1[1]:+9.3f} | {t2[0]:+16.4f} {t2[1]:+9.4f} {d['theta_up_global'].std():6.4f} | {t3[0]:+16.4f} {t3[1]:+9.4f} | {np.abs(sa_tr).max():.3f} K")
    ax[0].plot(rel, Tg, lw=0.4, label=lab.replace('p', '.'))
    ax[1].plot(rel, d['theta_up_global'] - d['theta_up_global'].mean(), lw=0.5)
    ax[2].plot(rel, d['theta_deep_global'] - d['theta_deep_global'].mean(), lw=0.8)
    ax[3].plot(rel, d['sa_ocean'] @ d['sa_ocean_w'] - (d['sa_ocean'] @ d['sa_ocean_w']).mean(), lw=0.4)
ax[0].set_ylabel('global Ts (K)'); ax[0].legend(fontsize=7, ncol=8)
ax[1].set_ylabel('global θ 0–700 m\nanomaly (K)'); ax[2].set_ylabel('global θ >1000 m\nanomaly (K)'); ax[3].set_ylabel('S Atlantic θ 0–700 m\nanomaly (K)')
ax[3].set_xlabel('years since window start'); fig.suptitle('Warm-branch analysis windows: annual values, no smoothing'); fig.tight_layout()
fig.savefig('figures/plasim_mechanism/window_stationarity.png', dpi=110)
