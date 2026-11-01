#!/usr/bin/env python3
"""Generate a long 'needle in a haystack' document: a secret fact planted near
the start, buried in filler, with a question at the end. Used to show that
vireo's streamed KV cache still retrieves long-range facts."""
import random, sys
random.seed(7)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
NEEDLE = "The vault access code is MARIGOLD-5521."
subj=["The quarterly report","A field technician","The weather station","Our logistics team",
      "The research group","A maintenance crew","The night shift","The data pipeline",
      "The supply depot","The monitoring system"]
verb=["recorded","observed","noted","reported","measured","flagged","archived","confirmed","reviewed","logged"]
obj=["a steady rise in ambient temperature","no anomalies in the sensor array",
     "routine traffic across the east corridor","nominal readings throughout the day",
     "minor fluctuations in the power grid","a calm and uneventful period",
     "consistent throughput on all channels","standard wear on the equipment",
     "clear skies and light winds","stable pressure in the main line"]
lines=[]
for i in range(N):
    if i == 22: lines.append(NEEDLE)
    lines.append(f"Entry {i:03d}: {random.choice(subj)} {random.choice(verb)} {random.choice(obj)} during the observation window.")
doc="\n".join(lines)
doc+="\n\nBased on the document above, answer the question.\nQuestion: What is the vault access code?\nAnswer: The vault access code is"
sys.stdout.write(doc)
