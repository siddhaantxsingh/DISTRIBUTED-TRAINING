#!/bin/sh
# Training throughput (weak scaling: 8 sequences of 64 tokens per rank per step)
# and a convergence comparison. Writes JSONL logs to $OUT.
set -e
OUT=${OUT:-benchmarks/results/$(date +%F)}
STEPS=${STEPS:-30}
mkdir -p $OUT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
M="--dim 128 --heads 4 --layers 4 --ctx 64 --batch 8 --lr 1e-3 --warmup 10"
L() { rm -f "$OUT/$1.jsonl"; echo "$OUT/$1.jsonl"; }
python3 -m dtrain.train --mode single --steps $STEPS $M --log $(L tp_single_w1)
for W in 2 4; do
  python3 -m dtrain.launch --nproc $W -- python3 -m dtrain.train --mode ddp --steps $STEPS $M --log $(L tp_ddp_w$W)
  python3 -m dtrain.launch --nproc $W -- python3 -m dtrain.train --mode ddp --no-overlap --steps $STEPS $M --log $(L tp_ddp_nooverlap_w$W)
  python3 -m dtrain.launch --nproc $W -- python3 -m dtrain.train --mode zero --steps $STEPS $M --log $(L tp_zero_w$W)
done
# convergence: 4-rank DDP vs the identical global batch in one process
CSTEPS=${CSTEPS:-300}
python3 -m dtrain.launch --nproc 4 -- python3 -m dtrain.train --mode ddp --steps $CSTEPS $M --log $(L conv_ddp_w4)
python3 -m dtrain.train --mode single --emulate-world 4 --steps $CSTEPS $M --log $(L conv_single_emul4)
