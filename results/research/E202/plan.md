# E202 probability-map route reuse: matched four-video study

E150-aligned p025/p048 (124 frames) and p044/p007 (243 frames), 768×1344, 20 steps, BF16 linears. Same prompt, seed, initial audio/video noise and BF16 reference as E159–E162. First-only captures INT8 softmax-mean map at step 1; periodic refreshes at steps 1,6,11,16. In both, layer 0 and refiner stay BF16; remaining layers use 90% NVFP4 + 10% INT8 without skipped blocks between capture steps. Baseline rebuilds original DraftMap on all steps/all 50 main layers, as E162. Each policy is regenerated to collect matched per-step DiT timings and videos. Report all-frame RGB PSNR/LPIPS and mean per-step DiT time by case and duration; original user table timing was p025 only, not four-case average.

The probability implementation is an experimental two-pass Triton calculation of actual INT8 QK probabilities, not fused into the native dense attention kernel. Therefore timings reflect this prototype. GPU assignment p044→5, p007→6, p025→7, p048→3. Native source snapshots are in E200/source.

## Scope correction

The user requested only one additional first-step softmax-map route. E202 first-only is the only new comparison arm. Old E159/E160/E161/E162 results remain untouched; mistakenly launched E202 periodic jobs were stopped and E202 baseline jobs were not run. Compare the new first-only results to the pre-existing three rows, not to unfinished E202 periodic artifacts.
