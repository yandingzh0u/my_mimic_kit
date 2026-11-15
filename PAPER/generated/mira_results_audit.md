# MIRA measured seed-0 result audit

MIRA Climb uses 3000 PPO updates (786,432,000 environment samples); all other results use 2000 updates (524,288,000 samples). Each run uses 8192 training environments and seed 0. The data are logged reference-start tests. They are not newly collected independent evaluations or multi-seed averages.

| Metric | AMP | DeepMimic | ADD | MIRA |
| --- | ---: | ---: | ---: | ---: |
| Root pos. [m] | 0.470355 | 0.066546 | 0.068289 | 0.041147 |
| Root rot. [rad] | 0.769662 | 0.129669 | 0.064010 | 0.031542 |
| Body pos. [m] | 0.209041 | 0.048791 | 0.033512 | 0.027212 |
| Body rot. [rad] | 0.792876 | 0.194958 | 0.125420 | 0.112390 |
| DoF vel. [rad/s] | 1.422410 | 0.868369 | 0.676244 | 0.661760 |
| Root vel. [m/s] | 0.417818 | 0.171232 | 0.195753 | 0.146422 |
| Root ang. vel. [rad/s] | 1.008975 | 0.416184 | 0.382162 | 0.229277 |

## Per-motion measured values

The four methods are reported using raw values with their actual budgets. Lower tracking errors are better; incomplete-trajectory errors cover only the executed portion.

| Motion | Method | Updates | Samples | Root pos. | Root rot. | Body pos. | Body rot. | DoF vel. | Root vel. | Root ang. vel. | Complete |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Roll | AMP | 2000 | 524288000 | 0.564474 | 1.669796 | 0.446277 | 1.531319 | 2.343228 | 0.733096 | 2.144311 | ✓ |
| Roll | DeepMimic | 2000 | 524288000 | 0.175238 | 0.404291 | 0.145254 | 0.512615 | 1.314119 | 0.341780 | 0.732933 | ✓ |
| Roll | ADD | 2000 | 524288000 | 0.077870 | 0.092656 | 0.039887 | 0.138884 | 0.833770 | 0.327842 | 0.553197 | ✓ |
| Roll | MIRA | 2000 | 524288000 | 0.074464 | 0.062108 | 0.046714 | 0.171270 | 0.950914 | 0.300730 | 0.455821 | ✓ |
| Climb | AMP | 2000 | 524288000 | 1.327212 | 1.430078 | 0.264049 | 1.459584 | 1.361819 | 0.585905 | 1.041243 | ✗ |
| Climb | DeepMimic | 2000 | 524288000 | 0.088287 | 0.152397 | 0.054555 | 0.229941 | 0.781111 | 0.183715 | 0.473947 | ✗ |
| Climb | ADD | 2000 | 524288000 | 0.162081 | 0.129522 | 0.063032 | 0.202526 | 0.746647 | 0.231891 | 0.515636 | ✗ |
| Climb | MIRA | 3000 | 786432000 | 0.051898 | 0.036550 | 0.024994 | 0.095557 | 0.547290 | 0.098013 | 0.234035 | ✓ |
| Backflip | AMP | 2000 | 524288000 | 0.403170 | 0.726356 | 0.228362 | 0.809751 | 2.240314 | 0.679867 | 1.551677 | ✗ |
| Backflip | DeepMimic | 2000 | 524288000 | 0.053722 | 0.098739 | 0.039628 | 0.179241 | 1.297614 | 0.269886 | 0.766224 | ✓ |
| Backflip | ADD | 2000 | 524288000 | 0.090497 | 0.089178 | 0.043732 | 0.165228 | 1.042275 | 0.352860 | 0.606055 | ✓ |
| Backflip | MIRA | 2000 | 524288000 | 0.054400 | 0.030334 | 0.037336 | 0.170105 | 0.979345 | 0.275952 | 0.231570 | ✓ |
| Crawl | AMP | 2000 | 524288000 | 0.038983 | 0.060481 | 0.028877 | 0.108076 | 0.328660 | 0.076579 | 0.201184 | ✓ |
| Crawl | DeepMimic | 2000 | 524288000 | 0.009997 | 0.013678 | 0.006199 | 0.037069 | 0.238457 | 0.032965 | 0.078033 | ✓ |
| Crawl | ADD | 2000 | 524288000 | 0.007575 | 0.007385 | 0.007876 | 0.041114 | 0.156761 | 0.030477 | 0.077828 | ✓ |
| Crawl | MIRA | 2000 | 524288000 | 0.007437 | 0.007500 | 0.006491 | 0.035559 | 0.173517 | 0.026732 | 0.067157 | ✓ |
| Get-Up | AMP | 2000 | 524288000 | 0.359801 | 0.630222 | 0.244519 | 0.678185 | 0.920618 | 0.236438 | 0.520589 | ✓ |
| Get-Up | DeepMimic | 2000 | 524288000 | 0.027914 | 0.054284 | 0.022976 | 0.102185 | 0.405883 | 0.063497 | 0.135313 | ✓ |
| Get-Up | ADD | 2000 | 524288000 | 0.036962 | 0.024350 | 0.023662 | 0.105237 | 0.302374 | 0.080665 | 0.117002 | ✓ |
| Get-Up | MIRA | 2000 | 524288000 | 0.025459 | 0.019518 | 0.024210 | 0.094135 | 0.321720 | 0.052877 | 0.095783 | ✓ |
| Spin Kick | AMP | 2000 | 524288000 | 0.128489 | 0.101041 | 0.042160 | 0.170344 | 1.339819 | 0.195023 | 0.594847 | ✗ |
| Spin Kick | DeepMimic | 2000 | 524288000 | 0.044120 | 0.054626 | 0.024132 | 0.108695 | 1.173029 | 0.135548 | 0.310654 | ✗ |
| Spin Kick | ADD | 2000 | 524288000 | 0.034748 | 0.040970 | 0.022883 | 0.099533 | 0.975634 | 0.150785 | 0.423252 | ✓ |
| Spin Kick | MIRA | 2000 | 524288000 | 0.033228 | 0.033241 | 0.023525 | 0.107714 | 0.997773 | 0.124228 | 0.291295 | ✓ |

## Logged test lengths

A completion mark in the paper only checks whether the mean logged test length reaches the nominal motion horizon (tolerance 0.001 step). It is not a task-specific success rate.

| Motion | Horizon | AMP | DeepMimic | ADD | MIRA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Roll | 60 | 60.000 | 60.000 | 60.000 | 60.000 |
| Climb | 299 | 289.857 | 295.468 | 75.002 | 299.000 |
| Backflip | 60 | 56.782 | 60.000 | 60.000 | 60.000 |
| Crawl | 77 | 77.000 | 77.000 | 77.000 | 77.000 |
| Get-Up | 91 | 91.000 | 91.000 | 91.000 | 91.000 |
| Spin Kick | 60 | 59.986 | 59.806 | 60.000 | 60.000 |

## Backflip ablations

| Configuration | Body pos. | Body rot. | DoF vel. | Test length | Sampled gradient max | Numerically zero reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MIRA | 0.03734 | 0.17010 | 0.97935 | 60.000 | 0.995 | 0.00% |
| w/o Monotone Modeling | 0.06927 | 0.30609 | 1.78123 | 60.000 | 0.624 | 0.00% |
| w/o Diagonal Norm. | 0.05959 | 0.24487 | 1.44644 | 26.001 | 10.214 | 39.72% |
| w/o Both Components | 0.06511 | 0.27675 | 1.43574 | 28.000 | 9.174 | 34.08% |

## Interpretation and limitations

- One measured seed per configuration; no measured seed uncertainty or significance claim.
- Metrics use logged in-training reference-start tests, not new independent evaluation results.
- Existing baseline eval/final/summary.json files use a separate protocol and are excluded.
- Early-terminated episode errors cover only executed timesteps; full-horizon completion is a survival check, not a task-specific success classifier.
- MIRA Climb uses the measured 3000-update record; all other terminal results use 2000 updates. These are not equal-budget terminal comparisons.
- Learning curves align each run by its own training-budget fraction; this is not equal-sample efficiency.
- Removing monotone modeling jointly changes representation, normalization, and weight signs; its effects cannot be attributed to just one change.
- Sampled score-gradient norms in different input representations do not establish comparable physical-space sensitivity.
- No synthetic seed placeholders enter active tables, curves, or aggregates.
- Additional independent training seeds and a unified independent checkpoint evaluation remain TODO.

Source paths and SHA-256 hashes are stored in mira_measured_seed0.json. All per-motion raw values are available in mira_per_motion_seed0.csv.
