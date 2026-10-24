---
library_name: uraionspec
license: mit
language:
- en
tags:
- legacy
- unsupported
- research-code
- speculative-decoding
- dspark
---

# Legacy research code — no model or speedup result

**Status as of 2026-07-27:** this repository is retained as historical research code. It is not a
model checkpoint, is not an active Uraion Labs product, and is disconnected from FinStruct.

The repository's own reproduction report records 80/80 unit tests, but it also states that target
model smoke training was blocked, GPU smoke evaluation was not run, and no meaningful speculative
latency benchmark was produced. It uses a simplified `nn.TransformerEncoder` rather than the full
paper backbone, a synthetic throughput profile, no multi-GPU path, and no vLLM integration.

Accordingly, prior wording such as “faithful,” “verified,” or “runnable” must not be read as a paper
reproduction, trained checkpoint, latency improvement, or production claim. The full pre-audit card
is preserved in [`LEGACY_CARD.md`](LEGACY_CARD.md).

Current Uraion Labs work is [FinStruct](https://github.com/arnavprabhu/uraion-finstruct): auditable,
local-first structured intelligence from financial documents. See
[uraionlabs.com](https://uraionlabs.com).


--x-- 
Block size : how many more points ahead will I speculate for , here config defines for 7 tokens ahead
N : Anchor points : tokens after which I can speculate , means if Ith token is choosen I can choose I+1th token there next 

Draft model : backbone + sequential head (markov-head)
* Backbone : this is transformer decoder layer only 
* Markov Head : these are added over backbone to get sequential values  
