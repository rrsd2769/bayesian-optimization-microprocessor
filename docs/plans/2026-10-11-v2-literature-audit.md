# v2 Plan Literature Audit (2026-10-11)

Checks every design choice in `2026-10-11-v2-study.md` against journal articles
with a **print year of 2020 or later**. Search: Consensus (peer-reviewed, year ≥ 2020).
Every DOI below was checked on `api.crossref.org` (`type = journal-article`, print year ≥ 2020).
Conference papers, preprints and anything printed before 2020 are excluded, even when they
are the original source.

Status: **B** = backed by journal precedent · **P** = partly backed (the method is backed, a specific constant or detail is not) · **G** = gap (no post-2020 journal precedent found; the plan was changed).

## 1. Audit table

| # | Plan item | Status | Post-2020 journal backing (DOI) | Plan action |
|---|---|---|---|---|
| 1 | HW-aware NAS on MCUs with flash/SRAM limits as hard constraints | B | King 2025 Sci Rep `10.1038/s41598-025-90764-z` (latency + peak-memory limits on MCUs); Chitty-Venkata 2023 ACM CSUR `10.1145/3524500`; Garavagno 2024 IEEE Sens Lett `10.1109/lsens.2024.3387056` | none |
| 2 | Peak SRAM = max over layers (in + out + scratch) | B | King 2025 (peak memory limit) | none |
| 3 | b ∈ {32, 16, 8} on Cortex-M4 (fp32, int16, int8) | B | Novac 2021 Sensors `10.3390/s21092984` (fp32 / int16 / int8 on Cortex-M4F, compared with STM32Cube.AI); Karuppusamy 2022 Microprocess Microsyst `10.1016/j.micpro.2022.104634` | none |
| 4 | b = 4: int4 per-channel weights + int8 activations | B | Capotondi 2020 IEEE TCAS-II `10.1109/tcsii.2020.2983648` (CMix-NN, independent 8/4/2-bit weight and activation precision, per-channel, on STM32H7) | none |
| 5 | 4-bit via QAT fine-tune from the float checkpoint | P | Sher 2023 Mathematics `10.3390/math11092112` (2–8-bit QAT on CIFAR-10) | **"5 epochs, lr 1e-4" has no source.** Changed: QAT epoch count chosen by a validation-plateau pilot on 2 architectures, then fixed. |
| 6 | Width × precision trade-off evaluated with an energy model | B | de Putter 2025 IEEE Access `10.1109/access.2025.3567046` (channel multiplier × bit-width, processor energy model, Pareto) | none |
| 7 | BatchNorm folding before quantization | — | detail of how the model is built; no claim made | not cited |
| 8 | Cycle model: per-layer MACs × cycles-per-MAC, calibrated on silicon | B | King 2025 (per-layer latency lookup tables); Yassin 2021 J Signal Process Syst `10.1007/s11265-020-01544-z` (model calibrated on Cortex-M4 measurements) | none |
| 9 | cpm[16], cpm[4] derived from Cortex-M4 TRM instruction counts | **G** | none found | **Changed:** take the int16:int8 and int4:int8 cycle ratios from measured journal data (Novac 2021 for int16, Capotondi 2020 for int4), label them literature-derived estimates, keep the TRM count only as a cross-check. |
| 10 | `k_flash` flash-read term | **G** | none found | **Changed:** only added if leave-one-out error > 10 % without it. |
| 11 | E = V·I_run·t_active + V·I_sleep·t_idle with datasheet currents | B | Wu 2021 Appl Sci `10.3390/app11062581` (STM32F407, same F4 family: energy = power × time, constant peripheral power, sleep-mode break-even); Zidar 2024 Electronics `10.3390/electronics13050826`; Sofianidis 2025 Technologies `10.3390/technologies13020082` | none |
| 12 | Discrete datasheet operating points (DVFS) | B | Zidar 2024; Sofianidis 2025 | none |
| 13 | Measured-energy / latency / memory Pareto curves on MCU | B | Lattanzi 2022 Sensors `10.3390/s22072637` | none |
| 14 | Fixed-priority preemptive scheduling on FreeRTOS, Cortex-M4 | B | Oliveira 2023 Des Autom Embed Syst `10.1007/s10617-023-09267-7` (RM vs EDF in FreeRTOS on ARM-M4, overheads + energy savings) | none |
| 15 | Context switch 200 cycles, tick 100 cycles | P | Oliveira 2023; Ungurean 2020 Symmetry `10.3390/sym12040592` (measured RTOS timings, Cortex-M4); Heider 2025 ACM TECS `10.1145/3762651` (tick overhead) | **Constants not from a source.** Changed: replace with values reported in Oliveira 2023 / Ungurean 2020 (read full text in Phase 3); add ×0.5 / ×2 overhead sensitivity. |
| 16 | Harmonic periods (10 ms, 50 ms) | B | Pavić 2020 IEEE Access `10.1109/access.2020.3025159` | none |
| 17 | Sensor 90k / comms 360k cycle WCETs | **G** | — (invented) | **Changed:** keep as the nominal task set; sensitivity adds 2 task sets generated with a published method (Kim 2021 J Comput Sci Eng `10.5626/jcse.2021.15.2.72`, HarmonicT) at a low and a high utilization. |
| 18 | Miss ratio ≤ 1 % as a soft-deadline constraint | B | Pazzaglia 2021 ACM TECS `10.1145/3404888` (weakly hard: occasional misses tolerated) | none |
| 19 | Poisson inference arrivals | B | He 2020 IEEE IoT J `10.1109/jiot.2020.2981338` | none |
| 20 | Round robin among equal priorities with slice τ | P | Oliveira 2023 (FreeRTOS scheduling); no journal studies τ as a design variable for DNN inference | kept; this is part of the gap the study fills |
| 21 | Joint search of network + system/scheduling parameters beats a decoupled search | B | Chen 2023 ACM TECS `10.1145/3533251` (CHaNAS: network + scheduling policy co-search; already in `refs.bib`); Tuli 2023 ACM TECS `10.1145/3575798` | none |
| 22 | Accuracy surrogate: PCHIP in log w over 5 widths | **G** | Surrogate-assisted NAS is backed (Liu 2022 J Autom Intell `10.1016/j.jai.2022.100002`; Sun 2021 IEEE TEVC `10.1109/tevc.2021.3055076`), but no journal uses monotone spline interpolation over width | **Changed:** width becomes a discrete ordered choice of the 5 trained widths and accuracy is a measured table lookup, NATS-Bench style: Dong 2021 IEEE TPAMI `10.1109/tpami.2021.3054824` (size search space, every candidate trained). Holdout architectures are no longer needed. |
| 23 | Train/val split; test set used once | B | Lones 2024 Patterns `10.1016/j.patter.2024.101046` | none |
| 24 | Seed-noise repeats | B | Lones 2024 | none |
| 25 | qNEHVI with outcome constraints | B | Low 2024 npj Comput Mater `10.1038/s41524-024-01274-x` (constrained qNEHVI); Mamun 2025 MLST `10.1088/2632-2153/ada47d` | original qNEHVI paper is a conference paper → not cited |
| 26 | qNParEGO baseline | B | Mamun 2025 (qEHVI vs qNEHVI vs qParEGO vs qNParEGO, discrete space) | none |
| 27 | MOBO for TinyML NAS | B | Deutel 2025 ACM TELO `10.1145/3715012`; Iqbal 2023 JAIR `10.1613/jair.1.14139` | none |
| 28 | NSGA-II baseline (pymoo) | B | Blank 2020 IEEE Access `10.1109/access.2020.2990567` (pymoo); Guella 2024 IEEE TCAS-I `10.1109/tcsi.2024.3365952` (NSGA-II for MCU NN configuration); Wang 2021 Sensors `10.3390/s21206927` | Deb 2002 removed (pre-2020) |
| 29 | BO vs NSGA-II at equal evaluations | B | Manoj 2023 Mater Manuf Process `10.1080/10426914.2023.2195915` | none |
| 30 | MOTPE | B | Ozaki 2022 JAIR `10.1613/jair.1.13188` (incl. a CNN design problem) | none |
| 31 | MOTPE constraint handling (`constraints_func`) | P | no journal source (constrained TPE is conference-only) | kept, described as an Optuna option in the text; no claim cited |
| 32 | Random search baseline | B | Karl 2023 ACM TELO `10.1145/3610536`; Hernández 2023 Artif Intell Rev `10.1007/s10462-022-10359-2` | none |
| 33 | 12 Sobol initial points out of 100 | B | Le Riche 2021 Struct Multidiscip Optim `10.1007/s00158-021-02977-1` (small initial design); Greenhill 2020 IEEE Access `10.1109/access.2020.2966228` | Sobol 1967 / Owen 1998 removed |
| 34 | HV against an exhaustive reference front; IGD | B | Guerreiro 2022 ACM CSUR (in `refs.bib`); Cai 2021 IEEE TEVC `10.1109/tevc.2020.2991040` (IGD needs the true front); Nuh 2021 Appl Sci `10.3390/app11073117` | Zitzler 2003 / Coello 2005 removed |
| 35 | Single fixed HV reference point | P | Pang 2024 IEEE TEVC `10.1109/tevc.2023.3287399`: results depend on the reference point | **Changed:** also report HV at a second reference corner. |
| 36 | Friedman + Wilcoxon (Holm) + Cliff's δ | B | Carrasco 2020 Swarm Evol Comput `10.1016/j.swevo.2020.100665`; Ismail 2026 Algorithms `10.3390/a19030191` | Wilcoxon 1945 / Cliff 1993 removed |
| 37 | 20 seeds | P | Carrasco 2020 discusses sample size; recent journal comparisons commonly use 30 runs | **Changed:** 30 seeds if the Phase 6 pilot is fast enough, otherwise 20 with the minimum achievable p reported. |
| 38 | STM32Cube.AI Developer Cloud for on-silicon cycles/memory | B | Berta 2024 IEEE OJIES `10.1109/ojies.2024.3451959` (uses the Developer Cloud to benchmark on real boards); Crocioni 2020 IEEE Access `10.1109/access.2020.3007046`; Sakr 2020 Sensors `10.3390/s20092638` | none |
| 39 | Datasheet / reference-manual constants (currents, wait states, ART) | — | primary vendor documents, named in text (existing rule) | none |
| 40 | Bayesian optimization background | B | Wang 2023 ACM CSUR `10.1145/3582078` | Shahriari 2016, Jones 1998, Letham 2019 removed |
| 41 | Edge DL inference background | B | Shuvo 2023 Proc IEEE `10.1109/jproc.2022.3226481` | Sze 2017 removed |
| 42 | NAS background | B | Ren 2022 ACM CSUR `10.1145/3447582` | Elsken 2019 removed |
| 43 | EHVI background | B | Pang 2023 IEEE TEVC `10.1109/tevc.2022.3228516` | Emmerich 2006 removed |

Excluded during the audit: Garbay ICECS 2021 (closest cycle/power model on an STM32 Cortex-M4, but a conference paper); Miranda WPerformance 2021 (measured FreeRTOS tick/switch on M4, workshop); μNAS (workshop); APQ (CVPR); Singh IEEE TEVC (print year 2019); Pickles npj Comput Mater and Chen Comput Electron Agric (Crossref resolves only to the preprint).

## 2. Existing `refs.bib` entries printed before 2020

The ≥ 2020 rule affects 17 of the 34 current entries. Proposed replacements (all verified above):

| Current key | Year | Replacement |
|---|---|---|
| `elsken2019survey` | 2019 | Ren 2022 ACM CSUR |
| `liu1973scheduling` | 1973 | Oliveira 2023 DAES |
| `bambagini2016energy` | 2016 | Oliveira 2023 DAES / Zidar 2024 |
| `sheikh2018multicore` | 2018 | Sofianidis 2025 Technologies, or drop |
| `chandrakasan1992lowpower` | 1992 | Wu 2021 Appl Sci (the v2 energy model is current × time, not CV²f) |
| `sze2017efficient` | 2017 | Shuvo 2023 Proc IEEE |
| `shahriari2016review` | 2016 | Wang 2023 ACM CSUR |
| `jones1998efficient` | 1998 | drop (single-objective EI not used in v2) |
| `letham2019constrained` | 2019 | Low 2024 npj Comput Mater |
| `emmerich2006single` | 2006 | Pang 2023 IEEE TEVC |
| `sobol1967points` | 1967 | Le Riche 2021 SMO |
| `owen1998scrambling` | 1998 | drop (Sobol named in text) |
| `deb2002nsga2` | 2002 | Blank 2020 IEEE Access |
| `zitzler2003performance` | 2003 | Hernández 2023 AIR / Nuh 2021 |
| `coello2005igd` | 2005 | Cai 2021 IEEE TEVC |
| `wilcoxon1945individual` | 1945 | Carrasco 2020 SWEVO |
| `cliff1993dominance` | 1993 | Ismail 2026 Algorithms |
