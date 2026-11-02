# Final optimization delivery — current90%NVFP4/10%INT8 path

The requested optimization pass is implemented and complete. This is not a
claim of global kernel optimality or a large NVFP4 mainloop speedup. Accepted
changes improve first capture and the cached-reuse full API, while regressions
and unproven mainloop variants were restored. Current package-local binary:
1f52b6dffc744d3bc175beb53a90d49124e66ee264814aaf39fab766e92949a1.

## Delivered requirements and evidence

- Analyze preparation/routing/materialization/main/prefix/assembly costs:
  E163–E169 stage/kernel/isolation results. Materializer's in-chain span is not
  its own kernel time; isolated costs are tiny. Static full route caching did
  not produce a meaningful gain and was not integrated.
- Optimize first capture:scale-after-max and warp-local partial maxima with
  separate reduction remove repeated scalar products and CTA barriers. QKV
  preparation avoids unused FP16 Q/K writes and specializes/unrolls hot modes.
  E188 actual first-step captures across3layers show~12.9%lower full API time.
- Optimize cached reuse:the same preparation changes apply automatically to
  no-pool Q128/D128 NVFP4+INT8. E188 same-layer step0->step10 replay across
  layers0/25/49 shows2.4–2.6%lower full API time, with unchanged proxy control.
- Investigate fixed-route opportunities:full physical route caching E171,
  query-precision activity E175, and cached route remapping E181; these do not
  justify additional memory/cost for the current90/10policy. Dense prefix needs
  INT8 K/V, and almost all queryrows require both formats.
- Investigate main NVFP4 performance:Nsight E179/E190 plus concrete E174,
  E177,E180–E185,E191–E193 variants. Registers255/thread and16.7%occupancy,
  fixed-latency/FP32dependencies and shared conflicts are measured bottlenecks.
  No variant produced a robust net API gain, so all are excluded from delivery.
- Necessary correctness:original/current output+route hashes agree for target
 480p/768p inputs;E188 real layers0/25/49 first/later-step hashes allagree.
 User asked not to make this overly strict:full-generation E189 was never
 uploaded/launched, and is not a remaining gate. No additional generation test
 is required to claim the measured operator improvements.
- Reviewable artifact:kernel_optimization.patch contains only accepted delta
 against pre-optimization source snapshots,3CUDAfiles/173lines. A reverse
 git-apply check passed against current sources. Build logE186 and artifact
 hashes identify the compiled implementation. No git commit/push performed.

## Representative measured result

E188 uses real480p124f15795-token inputs from3H3layers,step0route capture and
step10reuse. GPU1;10warmups,30calls/path/order,forward/reverse pathorders;
originalA/current/originalB. Original values averageA/B.

|Layer|Capture original/current ms|Reuse original/current ms|
|---:|---:|---:|
|0|38.113 /33.212|29.754 /28.982|
|25|38.192 /33.248|29.764 /29.014|
|49|37.823 /32.963|29.598 /28.884|

Unchanged proxy control changed only+0.2–0.4% in current round.
Earlier final first-block ABA E186:480p capture37.937->33.074ms/reuse29.614
->28.986ms;768p capture140.055->123.748ms/reuse101.795->101.117ms. The768p
reuse gain0.7% is close to clock variation and not presented as a robust result.
These are operator measurements, not end-to-end generation speedup claims.

## Last mainloop experiments

E191 reordered shared-copy threads with inverse oldswizzle; bitwise equal but
shared excess stayed699,906,045 and no stablegain. Reverted.
E192 changed only32-byte Vrow swizzle from legacy to rowgroup4 xor16-byte
column, shared excess699,906,045->417,286,141 (~40%less), bitwiseequal. E193
50call ABA compared against acceptedE186:480p reuse28.712baseline vs29.017
candidate;768p100.245 vs100.311. No netgain, so restored source+binary.

No optimization tmux task remains. Original and accepted binary/source
snapshots and all rejected variants remain under results/local/E### for
reproducibility. Reports/logs are under results/research/E### andresults/logs.
