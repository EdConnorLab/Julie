"""
Self-contained mapping verification. Run in the folder with common.py.
Set LOADER_SOURCE to 'mine' to test YOUR replicate_analysis.load_data_kw,
or 'reference' to test the reference implementation below.
"""
import pandas as pd, numpy as np
from common import MONKEY_NAME, SUBJECT, K_TO_FULL, KW_PKL

PKL = KW_PKL
LOADER_SOURCE = 'mine'   # <-- change to 'mine' in your env to test your edited file

order9 = [m for i, m in enumerate(MONKEY_NAME) if i != SUBJECT]

# reference loader (same logic I gave you)
def load_data_kw_reference(pkl_path):
    d = pd.read_pickle(pkl_path)
    d = d[d.MonkeyGroup == 'Zombies']
    rows, meta = [], []
    for (nid, ws, we), g in d.groupby(['NeuronID','WindowStart_ms','WindowEnd_ms']):
        m = g.set_index('MonkeyName')['MeanSpikeRate']
        if all(k in m.index for k in order9):
            rows.append([m[k] for k in order9])
            meta.append({'Cell': nid, 'Time Window': f'({ws}, {we})'})
    return np.array(rows, float), pd.DataFrame(meta)

if LOADER_SOURCE == 'mine':
    from common import load_data_kw
else:
    load_data_kw = load_data_kw_reference

X_mean, df = load_data_kw(PKL)
print(f"Loaded X_mean shape: {X_mean.shape}  (expect 38 x 9)\n")
print("Expected column order:", order9)

raw = pd.read_pickle(PKL); raw = raw[raw.MonkeyGroup=='Zombies']
cell0 = df.iloc[0]['Cell']; win0 = df.iloc[0]['Time Window']
ws, we = [int(float(x)) for x in win0.strip('()').split(',')]
g = raw[(raw.NeuronID==cell0)&(raw.WindowStart_ms==ws)&(raw.WindowEnd_ms==we)]
raw_rates = g.set_index('MonkeyName')['MeanSpikeRate']

print(f"\nTEST 1 — cell {cell0}, window {win0}")
print(f"{'monkey':>7s} {'X_mean[0,k]':>12s} {'raw pkl':>10s} {'':>4s}")
all_ok=True
for k,mk in enumerate(order9):
    ok=np.isclose(X_mean[0,k], raw_rates[mk]); all_ok&=ok
    print(f"{mk:>7s} {X_mean[0,k]:12.4f} {raw_rates[mk]:10.4f} {'OK' if ok else 'X'}")
print(f"  TEST 1 {'PASSED' if all_ok else 'FAILED'}")

full_via_k=[MONKEY_NAME[K_TO_FULL[k]] for k in range(9)]
t2=(full_via_k==order9)
print(f"\nTEST 2 — behavior indexing order: {full_via_k}\n  {'PASSED' if t2 else 'FAILED'}")

Xs=X_mean.copy(); Xs[:,[0,1]]=Xs[:,[1,0]]
broke=not np.isclose(Xs[0,0], raw_rates[order9[0]])
print(f"\nTEST 3 — deliberate swap detected: {'YES' if broke else 'NO'}")
print(f"\nOVERALL: {'ALL PASSED' if (all_ok and t2 and broke) else 'FAILED'}")
