# Template-matching revision and recording check (3.6.17)

## Previous workflow (3.6.16)

The GUI loaded the saved `detected_ss_templates.npz` and `detected_cs_templates.npz` average templates **and** individual waveforms into banks. It smoothed each trace with the selected frame setting, estimated the chosen baseline (30 ms median in the supplied settings), subtracted it, and inverted negative-going fluorescence spikes. CS and SS then entered separate filter paths. In the supplied settings CS had a 150 Hz high cut; SS had no high cut. Template waveforms and data were resampled to at least 5 kHz. A Gaussian signal-versus-noise log-likelihood ratio (LLR) score was computed at each position from the bank mean and variability. The old cutoff was `setting × MAD(LLR score)` measured from zero even though the noise score is often strongly negative. CS candidates passed a >4 ms FWHM check, then SS scoring excluded the initial blank, a 9 ms pre/post-CS mask, and windows touching that mask. SS peaks were separated by at least 4 ms and a measured FWHM >3 ms was rejected. SS widths were measured on the baseline-corrected trace.

This full 7.6 ms SS template can mismatch a spike whose recovery overlaps the next one. The fixed-amplitude Gaussian model can also penalize otherwise recognizable spikes with different amplitudes. A zero-based score cutoff is not calibrated to the background LLR center. In the supplied B6 channels, a narrow artifact near 1,112 Hz dominates unfiltered SS traces and changes both scores and width measurements.

## Current workflow (3.6.17)

CSV/Excel time units are inferred from a labelled time header or an unambiguous sampling rate; the inferred unit appears in Info and exports. After the same user-selected frame processing and baseline subtraction, template SS detection applies a visible, adjustable 700 Hz low-pass by default. A user-selected SS filter still applies as configured. The original baseline-corrected trace is retained for waveform and SNR review.

Two template scores are available:

1. **LLR Probability Vector** retains the Gaussian bank model. Its cutoff is now the background-score median plus the configured number of score MADs, bounded below by zero so that accepted positions provide positive likelihood evidence. A waveform response floor removes weak score-only maxima.
2. **Normalized Similarity** calculates local Pearson correlation to the event-centred bank template. CS uses its full template; SS uses a 3 ms spike core so that a neighboring spike need not reproduce the complete recovery tail. Its defaults are CS `r ≥ 0.90`, SS `r ≥ 0.80`, SS response `≥2.2 × MAD` of the SS trace, and CS response `≥3 × MAD` of the CS trace. Correlation and response are both required; a shape-only noise match cannot pass.

Both scores honor CS exclusions, template-window support, initial blank, the user-set SS minimum distance, and CS morphology. Template SS widths are measured on the **filtered detection trace**, rejecting finite widths below the configurable 0.8 ms minimum or above the existing 3 ms maximum. Unknown widths remain marked uncertain rather than silently passing. `SS_MIN_DIST_MS=4` still prevents reporting two events closer than 4 ms; reduce it in Advanced Settings only after checking real close-burst examples.

## Check on the supplied recording

The 10.000 s, 17-cell `BestSS_MC_pooled_3333hz.csv` has a `t(ms)` column sampled at approximately 3333.33 Hz. Its template files contain 3,071 individual SS waveforms and 163 individual CS waveforms plus their average templates. I compared the saved 3.6.16 result (LLR, SS cutoff 0.75, 4 ms minimum distance, 30 ms median baseline, two-frame smoothing) with the new similarity defaults, using the same banks and the existing maximum-width filter. Counts below are **post-width candidate counts**, not validated true spikes. The amplitude measure is the median event-centred, baseline-corrected raw response divided by the raw MAD, so it is comparable between runs but is not the application's full waveform SNR statistic.

| Cell | Saved SS | New SS | Saved median response/MAD | New median response/MAD |
|---|---:|---:|---:|---:|
| 0903_B10_1 | 185 | 206 | 2.49 | 2.47 |
| 0903_B10_2 | 214 | 223 | 2.41 | 2.39 |
| 0827_B9_1 | 240 | 274 | 2.34 | 2.54 |
| 0827_B9_3 | 228 | 272 | 2.50 | 2.50 |
| 0828_B7_3 | 0 | 213 | — | 2.58 |
| 0827_B6_2 | 0 | 144 | — | 1.73 |
| **All 17** | **1,578** | **3,422** | — | — |

The end-to-end GUI run on a temporary copy reproduces these counts and exports the inferred `ms` unit and 700 Hz effective low-pass. Across all 17 cells, its mean SS waveform SNR is **2.57**, compared with **2.89** in the saved result. Thus the default increases candidate rate while including weaker events; it does not improve every metric at once. Raising the minimum response from 2.2 to 2.5 on the same data yields 2,410 candidates and mean SNR 2.76, at the cost of losing some events in already active cells. The B6 channels have much stronger 1,112 Hz interference than the other shown channels, so raw waveform SNR can also understate the benefit of filtering; their newly recovered events still need manual confirmation. More generally, the saved result is not ground truth: the rate increase does **not** establish higher precision or recall. The supplied annotated images do not include absolute event times and cell IDs, so they cannot be scored quantitatively against this CSV. Representative fixed-window comparisons and the interference spectrum are in [the event plot](template_matching_qc_3.6.17.png) and [the spectrum plot](template_matching_interference_3.6.17.png).

The next rigorous check is a blinded event list for these recordings, including ambiguous close bursts and negative controls. Compare precision/recall at fixed false-positive rate per cell and stratify by inter-spike interval and waveform SNR. A per-cell robust template or template family, followed by overlap-aware subtraction/deconvolution, may improve closely spaced events further, but should be selected using those labels rather than event count alone.
