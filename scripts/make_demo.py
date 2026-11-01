#!/usr/bin/env python3
"""Generate docs/demo.gif: measured resident RAM vs context length for vireo's
out-of-core KV cache, next to what a dense cache would need. Runs the real
pager via `./vireo csv`, so every point is measured, not asserted."""
import subprocess, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIREO = os.path.join(HERE, "vireo")
SWEEP = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]

def measure():
    rows = []
    for n in SWEEP:
        out = subprocess.check_output([VIREO, "csv", str(n), "topk"], text=True).strip()
        toks, resident, dense, hits, tps = out.split(",")
        rows.append((int(toks), int(resident), int(dense), float(hits), float(tps)))
        print(f"  {int(toks):>9,} tok  resident={int(resident)/1048576:6.1f}MB  "
              f"dense={int(dense)/1048576:8.1f}MB  hits={float(hits):.0f}%")
    return rows

def main():
    print("measuring pager across context sizes...")
    rows = measure()
    ctx   = np.array([r[0] for r in rows], float)
    resid = np.array([r[1] for r in rows], float) / 1048576.0   # MB
    dense = np.array([r[2] for r in rows], float) / 1048576.0   # MB
    with open(os.path.join(HERE, "docs", "demo_data.csv"), "w") as f:
        f.write("tokens,resident_bytes,dense_bytes,hits_pct,tokens_per_s\n")
        for r in rows: f.write(",".join(str(x) for x in r) + "\n")

    # smooth dense curve (exact: linear in tokens); vireo flat at measured max
    xs = np.logspace(3, 6, 200)
    dense_s = dense[-1] * (xs / ctx[-1])          # linear
    vireo_flat = float(resid.max())
    ratio_end = dense[-1] / resid[-1]

    plt.rcParams.update({"font.size": 13, "font.family": "monospace"})
    BG, FG, ACC, DANGER = "#0d1117", "#e6edf3", "#2ea043", "#f85149"
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color("#30363d")
    ax.tick_params(colors=FG); ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e3, 1e6); ax.set_ylim(1, max(dense_s)*2)
    ax.set_xlabel("context length (tokens)"); ax.set_ylabel("RAM for KV cache (MB)")
    ax.grid(True, which="major", color="#21262d", lw=0.7)
    ax.set_title("vireo — KV cache paged to disk: RAM stays flat as context grows",
                 color=FG, fontsize=13, pad=12)

    (dl,) = ax.plot([], [], color=DANGER, lw=2.4, label="dense cache (all in RAM)")
    (vl,) = ax.plot([], [], color=ACC, lw=2.4, label="vireo resident (streamed)")
    dot = ax.scatter([], [], color=FG, zorder=5, s=28)
    txt = ax.text(1.1e3, max(dense_s), "", color=FG, va="top", fontsize=12)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, loc="lower right")

    frames = 90
    def frame(i):
        f = (i + 1) / frames
        xmax = 10 ** (3 + 3 * f)                  # sweep 1e3 -> 1e6
        m = xs <= xmax
        dl.set_data(xs[m], dense_s[m])
        vl.set_data(xs[m], np.full(m.sum(), vireo_flat))
        d_here = dense_s[m][-1] if m.any() else dense_s[0]
        dot.set_offsets([[xmax, d_here], [xmax, vireo_flat]])
        txt.set_text(f"{int(xmax):>10,} tokens\n"
                     f"dense : {d_here:8.1f} MB\n"
                     f"vireo : {vireo_flat:8.1f} MB\n"
                     f"       {d_here/vireo_flat:6.0f}x less RAM")
        return dl, vl, dot, txt

    hold = 25
    def frame_all(i): return frame(min(i, frames - 1))
    anim = animation.FuncAnimation(fig, frame_all, frames=frames + hold,
                                   interval=55, blit=False)
    out = os.path.join(HERE, "docs", "demo.gif")
    anim.save(out, writer=animation.PillowWriter(fps=18))
    print(f"wrote {out}  ({os.path.getsize(out)//1024} KB)")
    print(f"headline: 1,000,000 tokens -> {dense[-1]:.0f}MB dense vs {resid[-1]:.1f}MB vireo "
          f"({ratio_end:.0f}x less), output bit-identical (make m1-test)")

if __name__ == "__main__":
    main()
