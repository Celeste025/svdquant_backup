# E195 — final optimized build at skip75 / NVFP4 15 / INT8 10

Interpretation:75%of video tile interactions skipped,15%NVFP4,10%INT8.
API retained precision fractions are NVFP4=.6 andINT8=.4;prefix policy isINT8.
Q128,124frames,realrace step0/block0 QKV.480p=480x832/15795tokens;
768p=768x1344/38661tokens. This is fullattentionAPI timing,not fullDiT/video.

GPU1,10warmups40timedcalls/path;originalA->optimized->originalB. Original
meansbelow averageA/B. Optimizedbinary1f52b6dffc744d3bc175beb53a90d49124e66ee264814aaf39fab766e92949a1.

|Input/path|Original mean ms|Optimized ms|Reduction|
|---|---:|---:|---:|
|480p capture|37.8027|32.9202|12.92%|
|480p cachedreuse|14.9580|14.1969|5.09%|
|768p capture|139.6657|122.7665|12.10%|
|768p cachedreuse|41.9572|40.3474|3.84%|

Same optimizedround proxy(DraftMap eachstep) was15.5391ms480p and43.5588ms
768p. Optimizedcachedreuse is8.64%and7.37%faster than thatproxy, respectively.
Unchangedproxycontrol differed only+0.10%/+0.45%vsoriginalA/Bmean.

Existingvalidator completed finiteoutput,firstcapture-vs-denseINT8 NMSE0,
and reuse with DraftMap/selection patched to raise. Userrequested lightweight
correctness;no extra generation or trajectorytest was added for thisquestion.

Rawreports:480p/768p_{original_a,optimized,original_b}.json. Launchsource:
scripts/research/launch_e195_sparse_aba.sh. AutomaticEXITrestoration completed;
currentbinaryhashmatchedbackup,notmuxexperimentleft. No sourcechanges made.
