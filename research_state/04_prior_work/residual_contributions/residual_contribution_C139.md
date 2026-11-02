# C139 residual and decision

Only pursue quantization-conditioned computation-graph selection if a fixed
budget oracle first demonstrates substantial transferable headroom. Generic
joint quantization/sparsity, attention-mass mixed precision, layer-step
scheduling, learned sparse routing and QAT are not new contributions.

C139A would be a mechanism-plus-method paper only if compression changes the
best support in a reproducible direction and a teacher-free predictor exploits
it across prompts and at least two model/step regimes with real latency gains.
C139B would need a strong uncertainty mechanism and system improvement to
avoid being an implementation tweak. Both remain untested; no experiments
launched as part of the literature discussion.
