<!-- bdbv-submission:seabbs-rwnowcast:2026-11-28 -->
Submission of `seabbs-rwnowcast` for 2026-11-28, from data as of 2026-11-22, for 546 locations.

Signed off by Sam Abbott (@seabbs).

Model: random walk on a nowcast (`randomwalknowcast`).

![Nowcast and forecast for 2026-11-28](https://raw.githubusercontent.com/seabbs-bot/BDBV-Modeling-Hub/fa96d475eb1738193db109ca34916fcbcbae63e4/plots/seabbs-rwnowcast/2026-11-28/45492920880.png)

|  | reported, mean of last 2 complete weeks | nowcast, week ending 2026-11-21 | forecast, week ending 2026-11-28 | forecast / reported |
| --- | --- | --- | --- | --- |
| DRC | 478 | 418 (280 to 696) | 437 (263 to 756) | 0.91 |
| Bas Uele | 0 | 1 (0 to 2.05) | 1 (0 to 3) | n/a |
| Haut Uele | 25.5 | 11 (4 to 26) | 11 (3.95 to 28) | 0.43 |
| Ituri | 230 | 272 (209 to 420) | 281 (192 to 416) | 1.22 |
| Nord Kivu | 210 | 122 (45 to 290) | 128 (36.9 to 334) | 0.61 |
| Nord Ubangi | 1 | 0 (0 to 2) | 0 (0 to 2) | 0.00 |
| Sud Kivu | 0 | 0 (0 to 0) | 0 (0 to 0) | n/a |
| Sud Ubangi | 0 | 0 (0 to 1) | 0 (0 to 1) | n/a |
| Tshopo | 12.5 | 3 (1 to 7.05) | 3 (0 to 10) | 0.24 |

| check | status |
| --- | --- |
| Convergence | N/A |
| Posterior predictive checks | N/A |
| Forecast checks | WARN |
| Hub validation | PASS |
| Data coherence | PASS |

This round in our docs: https://samabbott.co.uk/bdbv-hub-modelling/submission/rounds/2026-11-28/

<details><summary>Details</summary>

**National (`CD`) by week**

| week ending | median | 90% interval |
| --- | --- | --- |
| 2026-11-07 | 470 | 441 to 547 |
| 2026-11-14 | 495 | 411 to 679 |
| 2026-11-21 | 418 | 280 to 696 |
| 2026-11-28 | 437 | 263 to 756 |
| 2026-12-05 | 438 | 260 to 690 |
| 2026-12-12 | 418 | 239 to 808 |
| 2026-12-19 | 441 | 228 to 810 |

**Forecast checks**

- WARN zones: 519 zones; 5875 province sample-weeks differ from the sum of their zones
- WARN bounds: CD-IT-26: 95th centile 1.35e+03 is 17.1 times the bound base 79
- WARN plausible: CD-IT-26 2026-12-05: 97.5% quantile 1.55e+03 is over 15 times 92.9, the upper limit of the recent maximum 74 (and 2 more)
- WARN previous: CD-BU: median 7 against last week's submission 0 over 2026-10-17 to 2026-12-12 (ratio 2.4) (and 9 more)
- Passed: schema, values, samples, coverage, national_sum, growth, observed.

</details>

This was opened by a bot. Please ping @seabbs for any questions.
