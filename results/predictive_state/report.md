# EXP-0012 results (TEST split, locked protocol + amendments)

All numbers MEASURED in simulation (OccPick2D, 32x32, one laptop GPU/CPU). Success = fraction of episodes completing the task; cells NOM/OCC/DIST/HARD, macro = their mean; LONG = fraction of 10-pick episodes fully completed; picks = mean fraction of picks completed.

## Full-perception success by arm (mean over seeds)

| arm | NOM | OCC | DIST | HARD | macro | per-seed macro | LONG complete | LONG picks frac | seeds |
|---|---|---|---|---|---|---|---|---|---|
| G | 0.518 | 0.477 | 0.459 | 0.356 | 0.452 | 0.502, 0.511, 0.344 | 0.010 | 0.115 | 3 |
| G-I4a | 0.487 | 0.410 | 0.469 | 0.322 | 0.422 | 0.438, 0.363, 0.464 | 0.003 | 0.106 | 3 |
| G-I4u | 0.537 | 0.475 | 0.465 | 0.353 | 0.458 | 0.588, 0.328 | 0.025 | 0.150 | 2 |
| G-I8a | 0.463 | 0.487 | 0.400 | 0.360 | 0.427 | 0.427 | 0.000 | 0.089 | 1 |
| G-Ta | 0.596 | 0.599 | 0.529 | 0.424 | 0.537 | 0.531, 0.595, 0.485 | 0.013 | 0.166 | 3 |
| G-Ta-s8 | 0.570 | 0.497 | 0.530 | 0.363 | 0.490 | 0.490 | 0.010 | 0.131 | 1 |
| G-Tu | 0.566 | 0.522 | 0.521 | 0.420 | 0.507 | 0.487, 0.474, 0.561 | 0.017 | 0.152 | 3 |
| G0 | 0.664 | 0.631 | 0.596 | 0.493 | 0.596 | 0.613, 0.569, 0.606 | 0.013 | 0.182 | 3 |
| R | 0.707 | 0.461 | 0.622 | 0.336 | 0.531 | 0.502, 0.639, 0.453 | 0.047 | 0.207 | 3 |
| S4 | 0.700 | 0.489 | 0.636 | 0.339 | 0.541 | 0.481, 0.494, 0.648 | 0.077 | 0.244 | 3 |
| T16 | 0.731 | 0.688 | 0.671 | 0.512 | 0.651 | 0.628, 0.578, 0.746 | 0.080 | 0.268 | 3 |

## Question A: does persistent state help? (macro over OCC/HARD/DIST vs controls)

| arm | OCC | DIST | HARD | mean(OCC,DIST,HARD) |
|---|---|---|---|---|
| R | 0.461 | 0.622 | 0.336 | 0.473 |
| S4 | 0.489 | 0.636 | 0.339 | 0.488 |
| T16 | 0.688 | 0.671 | 0.512 | 0.624 |
| G0 | 0.631 | 0.596 | 0.493 | 0.573 |
| G | 0.477 | 0.459 | 0.356 | 0.430 |

## Question B: JEPA objective (G vs G0, paired by seed)

macro G - G0 = -0.144; bootstrap 95% [-0.164, -0.124]


## Question D: adaptive perception (TEST; encoder fraction = encoder runs / control steps)

| model | trigger@budget | NOM | OCC | DIST | HARD | macro | enc fraction | macro / own full | DIST / own full |
|---|---|---|---|---|---|---|---|---|---|
| G | full | 0.518 | 0.477 | 0.459 | 0.356 | 0.452 | 0.649 | 1.000 | 1.000 |
| G | fixed@0.2 | 0.512 | 0.457 | 0.461 | 0.322 | 0.438 | 0.177 | 0.969 | 1.005 |
| G | fixed@0.33 | 0.516 | 0.473 | 0.486 | 0.338 | 0.453 | 0.279 | 1.002 | 1.058 |
| G | fixed@0.5 | 0.539 | 0.486 | 0.500 | 0.359 | 0.471 | 0.415 | 1.041 | 1.090 |
| G | learned@0.2 | 0.457 | 0.396 | 0.409 | 0.291 | 0.388 | 0.157 | 0.858 | 0.891 |
| G | learned@0.33 | 0.468 | 0.446 | 0.456 | 0.322 | 0.423 | 0.258 | 0.935 | 0.993 |
| G | learned@0.5 | 0.529 | 0.468 | 0.492 | 0.348 | 0.459 | 0.396 | 1.015 | 1.073 |
| G | pix@0.2 | 0.476 | 0.429 | 0.433 | 0.297 | 0.409 | 0.177 | 0.904 | 0.944 |
| G | pix@0.33 | 0.513 | 0.457 | 0.458 | 0.337 | 0.441 | 0.287 | 0.975 | 0.998 |
| G | pix@0.5 | 0.534 | 0.472 | 0.498 | 0.352 | 0.464 | 0.412 | 1.026 | 1.085 |
| G-I4a | full | 0.487 | 0.410 | 0.469 | 0.322 | 0.422 | 0.647 | 1.000 | 1.000 |
| G-I4a | fixed@0.33 | 0.518 | 0.453 | 0.507 | 0.344 | 0.456 | 0.278 | 1.080 | 1.081 |
| G-I4a | learned@0.33 | 0.467 | 0.400 | 0.463 | 0.300 | 0.407 | 0.276 | 0.966 | 0.988 |
| G-I4a | pix@0.33 | 0.518 | 0.417 | 0.481 | 0.319 | 0.434 | 0.290 | 1.028 | 1.026 |
| G-I4u | full | 0.537 | 0.475 | 0.465 | 0.353 | 0.458 | 0.648 | 1.000 | 1.000 |
| G-I4u | fixed@0.33 | 0.495 | 0.425 | 0.445 | 0.345 | 0.427 | 0.284 | 0.934 | 0.957 |
| G-I4u | learned@0.33 | 0.460 | 0.392 | 0.445 | 0.293 | 0.398 | 0.254 | 0.869 | 0.957 |
| G-I4u | pix@0.33 | 0.495 | 0.428 | 0.450 | 0.340 | 0.428 | 0.274 | 0.936 | 0.968 |
| G-I8a | full | 0.463 | 0.487 | 0.400 | 0.360 | 0.427 | 0.644 | 1.000 | 1.000 |
| G-Ta | full | 0.596 | 0.599 | 0.529 | 0.424 | 0.537 | 0.653 | 1.000 | 1.000 |
| G-Ta | fixed@0.2 | 0.552 | 0.472 | 0.480 | 0.326 | 0.458 | 0.175 | 0.852 | 0.908 |
| G-Ta | fixed@0.33 | 0.591 | 0.527 | 0.546 | 0.392 | 0.514 | 0.279 | 0.957 | 1.032 |
| G-Ta | fixed@0.5 | 0.606 | 0.579 | 0.551 | 0.427 | 0.541 | 0.418 | 1.007 | 1.042 |
| G-Ta | learned@0.2 | 0.516 | 0.431 | 0.440 | 0.293 | 0.420 | 0.161 | 0.782 | 0.832 |
| G-Ta | learned@0.33 | 0.578 | 0.517 | 0.528 | 0.338 | 0.490 | 0.269 | 0.913 | 0.998 |
| G-Ta | learned@0.5 | 0.603 | 0.549 | 0.516 | 0.368 | 0.509 | 0.400 | 0.948 | 0.975 |
| G-Ta | pix@0.2 | 0.547 | 0.460 | 0.421 | 0.314 | 0.436 | 0.169 | 0.811 | 0.796 |
| G-Ta | pix@0.33 | 0.596 | 0.542 | 0.560 | 0.387 | 0.521 | 0.281 | 0.971 | 1.059 |
| G-Ta | pix@0.5 | 0.634 | 0.571 | 0.582 | 0.420 | 0.552 | 0.424 | 1.028 | 1.101 |
| G-Ta-s8 | full | 0.570 | 0.497 | 0.530 | 0.363 | 0.490 | 0.650 | 1.000 | 1.000 |
| G-Tu | full | 0.566 | 0.522 | 0.521 | 0.420 | 0.507 | 0.649 | 1.000 | 1.000 |
| G-Tu | fixed@0.33 | 0.636 | 0.556 | 0.552 | 0.438 | 0.545 | 0.282 | 1.075 | 1.060 |
| G-Tu | learned@0.33 | 0.589 | 0.519 | 0.522 | 0.376 | 0.501 | 0.259 | 0.988 | 1.002 |
| G-Tu | pix@0.33 | 0.654 | 0.561 | 0.570 | 0.431 | 0.554 | 0.284 | 1.093 | 1.094 |
| G0 | full | 0.664 | 0.631 | 0.596 | 0.493 | 0.596 | 0.662 | 1.000 | 1.000 |

## Exploratory (post-hoc, not gated): long blackouts (30-50 steps)

| arm | BLK40 | BLK40D (+displacement) | seeds |
|---|---|---|---|
| G | 0.492 | 0.390 | 3 |
| G-I4a | 0.451 | 0.388 | 3 |
| G-I4u | 0.487 | 0.373 | 2 |
| G-I8a | 0.443 | 0.363 | 1 |
| G-Ta | 0.543 | 0.418 | 3 |
| G-Ta-s8 | 0.503 | 0.423 | 1 |
| G-Tu | 0.540 | 0.426 | 3 |
| G0 | 0.627 | 0.536 | 3 |
| R | 0.484 | 0.299 | 3 |
| S4 | 0.500 | 0.327 | 3 |
| T16 | 0.611 | 0.499 | 3 |

## Analytical per-step cost (ANALYTICAL; weight+state bytes and MACs; not measured)

| arm | enc fraction | MACs/step (k) | bytes/step (kB) |
|---|---|---|---|
| G | 1.00 | 1265 | 588.1 |
| G-I8a | 1.00 | 1265 | 298.5 |
| G-I4u | 1.00 | 1265 | 383.3 |
| G-I4a | 1.00 | 1265 | 152.6 |
| G-Tu | 1.00 | 1265 | 347.1 |
| G-Ta | 1.00 | 1265 | 77.4 |
| R | 1.00 | 1145 | 348.5 |
| T16 | 1.00 | 1433 | 908.1 |

Controls retaining >=95% of G macro success: ['G-I4u'].


### G4 pairs: candidate vs strongest eligible control under the SAME learned trigger (measured test encoder fraction), analytical ratio control/candidate

(all INT4/INT8 controls listed; 'eligible' = retains >=95% of G macro success, the locked eligibility rule. Scope matters: G-Ta quantizes encoder+updater, G-Tu only the updater; the like-for-like INT4 control is the same scope.)

| candidate | control | eligible | enc frac cand/control | bytes ratio | MACs ratio |
|---|---|---|---|---|---|
| G-Ta | G-I4a | no | 0.27/0.28 | 2.01 | 1.02 |
| G-Ta | G-I4u | yes | 0.27/0.25 | 3.09 | 0.96 |
| G-Tu | G-I4a | no | 0.26/0.28 | 0.86 | 1.04 |
| G-Tu | G-I4u | yes | 0.26/0.25 | 1.33 | 0.99 |

## State usefulness: ridge probes from the recurrent state / encoder embedding (R^2; standardized features, CV lambda; fit on DEV LONG steps<=100, scored on TEST LONG)

Primary probe = h -> position of the occluded (blackout-frame) target. h->target on valid frames and z->target are positive controls; if the primary probe is near 0 while controls are high, h does not carry the remembered position.

| arm | seed | probe | n fit | R2 early (<=100) | R2 late (>160) | n late |
|---|---|---|---|---|---|---|
| G-I4a | 0 | h->target|blackout | 2597 | 0.597 | 0.647 | 322 |
| G-I4a | 0 | h->target|valid | 8531 | 0.754 | 0.776 | 1002 |
| G-I4a | 0 | z->target|valid | 8531 | 0.676 | 0.724 | 1002 |
| G-I4a | 0 | h->goal|blackout | 3112 | 0.788 | 0.850 | 377 |
| G-I4a | 1 | h->target|blackout | 2425 | 0.417 | 0.284 | 191 |
| G-I4a | 1 | h->target|valid | 8417 | 0.613 | 0.616 | 890 |
| G-I4a | 1 | z->target|valid | 8417 | 0.550 | 0.600 | 890 |
| G-I4a | 1 | h->goal|blackout | 2813 | 0.750 | 0.749 | 259 |
| G-I4a | 2 | h->target|blackout | 2573 | 0.518 | 0.522 | 521 |
| G-I4a | 2 | h->target|valid | 8500 | 0.673 | 0.626 | 1371 |
| G-I4a | 2 | z->target|valid | 8500 | 0.575 | 0.491 | 1371 |
| G-I4a | 2 | h->goal|blackout | 3097 | 0.871 | 0.906 | 605 |
| G-I4u | 0 | h->target|blackout | 2299 | 0.685 | 0.606 | 505 |
| G-I4u | 0 | h->target|valid | 8497 | 0.823 | 0.834 | 1932 |
| G-I4u | 0 | z->target|valid | 8497 | 0.785 | 0.823 | 1932 |
| G-I4u | 0 | h->goal|blackout | 2995 | 0.846 | 0.878 | 651 |
| G-I4u | 1 | h->target|blackout | 2530 | 0.220 | 0.477 | 53 |
| G-I4u | 1 | h->target|valid | 7932 | 0.354 | 0.291 | 260 |
| G-I4u | 1 | z->target|valid | 7932 | 0.451 | 0.079 | 260 |
| G-I4u | 1 | h->goal|blackout | 2738 | 0.763 | 0.796 | 53 |
| G-I8a | 0 | h->target|blackout | 2729 | 0.569 | 0.657 | 89 |
| G-I8a | 0 | h->target|valid | 8330 | 0.725 | 0.841 | 577 |
| G-I8a | 0 | z->target|valid | 8330 | 0.653 | 0.756 | 577 |
| G-I8a | 0 | h->goal|blackout | 3204 | 0.904 | 0.855 | 97 |
| G-Ta-s8 | 0 | h->target|blackout | 2463 | 0.581 | 0.623 | 406 |
| G-Ta-s8 | 0 | h->target|valid | 7930 | 0.774 | 0.797 | 1455 |
| G-Ta-s8 | 0 | z->target|valid | 7930 | 0.745 | 0.769 | 1455 |
| G-Ta-s8 | 0 | h->goal|blackout | 2997 | 0.797 | 0.713 | 505 |
| G-Ta | 0 | h->target|blackout | 2435 | 0.589 | 0.657 | 364 |
| G-Ta | 0 | h->target|valid | 8245 | 0.767 | 0.814 | 1178 |
| G-Ta | 0 | z->target|valid | 8245 | 0.705 | 0.735 | 1178 |
| G-Ta | 0 | h->goal|blackout | 3059 | 0.880 | 0.895 | 425 |
| G-Ta | 1 | h->target|blackout | 2300 | 0.611 | 0.751 | 795 |
| G-Ta | 1 | h->target|valid | 8540 | 0.827 | 0.862 | 2102 |
| G-Ta | 1 | z->target|valid | 8540 | 0.779 | 0.834 | 2102 |
| G-Ta | 1 | h->goal|blackout | 3122 | 0.839 | 0.886 | 974 |
| G-Ta | 2 | h->target|blackout | 2439 | 0.552 | 0.329 | 327 |
| G-Ta | 2 | h->target|valid | 8499 | 0.728 | 0.631 | 958 |
| G-Ta | 2 | z->target|valid | 8499 | 0.696 | 0.658 | 958 |
| G-Ta | 2 | h->goal|blackout | 2914 | 0.789 | 0.785 | 389 |
| G-Tu | 0 | h->target|blackout | 2572 | 0.661 | 0.607 | 401 |
| G-Tu | 0 | h->target|valid | 7822 | 0.842 | 0.873 | 1145 |
| G-Tu | 0 | z->target|valid | 7822 | 0.797 | 0.829 | 1145 |
| G-Tu | 0 | h->goal|blackout | 3309 | 0.791 | 0.746 | 518 |
| G-Tu | 1 | h->target|blackout | 2137 | 0.673 | 0.505 | 339 |
| G-Tu | 1 | h->target|valid | 8635 | 0.861 | 0.895 | 916 |
| G-Tu | 1 | z->target|valid | 8635 | 0.815 | 0.857 | 916 |
| G-Tu | 1 | h->goal|blackout | 2713 | 0.706 | 0.761 | 446 |
| G-Tu | 2 | h->target|blackout | 2414 | 0.638 | 0.512 | 684 |
| G-Tu | 2 | h->target|valid | 8528 | 0.823 | 0.802 | 1772 |
| G-Tu | 2 | z->target|valid | 8528 | 0.788 | 0.790 | 1772 |
| G-Tu | 2 | h->goal|blackout | 3247 | 0.804 | 0.771 | 819 |
| G0 | 0 | h->target|blackout | 2186 | 0.638 | 0.642 | 623 |
| G0 | 0 | h->target|valid | 8589 | 0.845 | 0.850 | 2154 |
| G0 | 0 | z->target|valid | 8589 | 0.785 | 0.821 | 2154 |
| G0 | 0 | h->goal|blackout | 2836 | 0.908 | 0.926 | 755 |
| G0 | 1 | h->target|blackout | 2226 | 0.677 | 0.686 | 560 |
| G0 | 1 | h->target|valid | 8555 | 0.842 | 0.869 | 1682 |
| G0 | 1 | z->target|valid | 8555 | 0.830 | 0.842 | 1682 |
| G0 | 1 | h->goal|blackout | 2853 | 0.870 | 0.922 | 680 |
| G0 | 2 | h->target|blackout | 2422 | 0.727 | 0.590 | 711 |
| G0 | 2 | h->target|valid | 8423 | 0.893 | 0.915 | 2180 |
| G0 | 2 | z->target|valid | 8423 | 0.839 | 0.862 | 2180 |
| G0 | 2 | h->goal|blackout | 3175 | 0.874 | 0.892 | 1024 |
| G | 0 | h->target|blackout | 2452 | 0.623 | 0.640 | 433 |
| G | 0 | h->target|valid | 8365 | 0.804 | 0.853 | 1164 |
| G | 0 | z->target|valid | 8365 | 0.776 | 0.821 | 1164 |
| G | 0 | h->goal|blackout | 2989 | 0.840 | 0.898 | 545 |
| G | 1 | h->target|blackout | 2422 | 0.567 | 0.681 | 607 |
| G | 1 | h->target|valid | 8179 | 0.740 | 0.834 | 1794 |
| G | 1 | z->target|valid | 8179 | 0.677 | 0.721 | 1794 |
| G | 1 | h->goal|blackout | 2869 | 0.868 | 0.920 | 733 |
| G | 2 | h->target|blackout | 2614 | 0.468 | 0.529 | 298 |
| G | 2 | h->target|valid | 8705 | 0.681 | 0.802 | 623 |
| G | 2 | z->target|valid | 8705 | 0.489 | 0.687 | 623 |
| G | 2 | h->goal|blackout | 3104 | 0.827 | 0.886 | 334 |

| arm | seed | mean abs h early/late | max abs h | NaN |
|---|---|---|---|---|
| G-I4a | 0 | 0.364/0.360 | 1.000 | False |
| G-I4a | 1 | 0.371/0.381 | 1.000 | False |
| G-I4a | 2 | 0.367/0.368 | 1.000 | False |
| G-I4u | 0 | 0.402/0.403 | 1.000 | False |
| G-I4u | 1 | 0.361/0.387 | 1.000 | False |
| G-I8a | 0 | 0.421/0.435 | 1.000 | False |
| G-Ta-s8 | 0 | 0.387/0.394 | 1.000 | False |
| G-Ta | 0 | 0.401/0.410 | 1.000 | False |
| G-Ta | 1 | 0.410/0.401 | 1.000 | False |
| G-Ta | 2 | 0.390/0.391 | 1.000 | False |
| G-Tu | 0 | 0.393/0.383 | 1.000 | False |
| G-Tu | 1 | 0.397/0.394 | 1.000 | False |
| G-Tu | 2 | 0.410/0.403 | 1.000 | False |
| G0 | 0 | 0.435/0.441 | 1.000 | False |
| G0 | 1 | 0.416/0.422 | 1.000 | False |
| G0 | 2 | 0.424/0.421 | 1.000 | False |
| G | 0 | 0.405/0.407 | 1.000 | False |
| G | 1 | 0.393/0.406 | 1.000 | False |
| G | 2 | 0.383/0.386 | 1.000 | False |

## LONG: per-pick conditional success (P(complete pick j | completed j-1)); episodes end at first failure

| arm | picks 1-2 | picks 3-5 | picks 6-10 | episodes reaching pick 6 |
|---|---|---|---|---|
| G | 0.532 | 0.522 | 0.697 | 13 of 300 |
| G-I4a | 0.503 | 0.529 | 0.613 | 13 of 300 |
| G-I4u | 0.544 | 0.675 | 0.766 | 20 of 200 |
| G-I8a | 0.483 | 0.444 | 0.000 | 2 of 100 |
| G-Ta | 0.614 | 0.653 | 0.629 | 30 of 300 |
| G-Ta-s8 | 0.572 | 0.526 | 0.714 | 5 of 100 |
| G-Tu | 0.575 | 0.636 | 0.722 | 27 of 300 |
| G0 | 0.651 | 0.632 | 0.682 | 31 of 300 |
| R | 0.622 | 0.748 | 0.782 | 46 of 300 |
| S4 | 0.698 | 0.706 | 0.830 | 55 of 300 |
| T16 | 0.721 | 0.720 | 0.836 | 60 of 300 |

## Embedding collapse monitor: final per-dimension std of z across the training batch (JEPA arms)

| run | z_std |
|---|---|
| G-I4a_s0 | 0.392 |
| G-I4a_s1 | 0.342 |
| G-I4a_s2 | 0.443 |
| G-I4u_s0 | 0.415 |
| G-I4u_s1 | 0.368 |
| G-I8a_s0 | 0.438 |
| G-Ta-s8_s0 | 0.335 |
| G-Ta_s0 | 0.338 |
| G-Ta_s1 | 0.332 |
| G-Ta_s2 | 0.323 |
| G-Tu_s0 | 0.378 |
| G-Tu_s1 | 0.424 |
| G-Tu_s2 | 0.399 |
| G_s0 | 0.445 |
| G_s1 | 0.430 |
| G_s2 | 0.414 |

## Multi-step latent prediction (cosine to EMA target embedding; k steps open-loop; DEV NOM+DIST)

| arm | seed | k=1 pred/shuffled/persist | k=4 | k=8 |
|---|---|---|---|---|
| G-I4a | 0 | 0.938/0.933/0.970 | 0.866/0.845/0.874 | 0.798/0.749/0.752 |
| G-I4a | 1 | 0.930/0.926/0.953 | 0.852/0.836/0.828 | 0.806/0.775/0.730 |
| G-I4a | 2 | 0.940/0.935/0.961 | 0.869/0.856/0.852 | 0.804/0.765/0.668 |
| G-I4u | 0 | 0.936/0.931/0.955 | 0.854/0.829/0.798 | 0.774/0.735/0.651 |
| G-I4u | 1 | 0.924/0.922/0.956 | 0.830/0.807/0.835 | 0.773/0.743/0.706 |
| G-Ta | 0 | 0.913/0.912/0.944 | 0.846/0.823/0.725 | 0.770/0.720/0.520 |
| G-Ta | 1 | 0.922/0.917/0.959 | 0.845/0.824/0.779 | 0.826/0.777/0.651 |
| G-Ta | 2 | 0.874/0.871/0.908 | 0.783/0.757/0.701 | 0.730/0.667/0.512 |
| G-Tu | 0 | 0.915/0.912/0.959 | 0.818/0.806/0.822 | 0.693/0.681/0.628 |
| G-Tu | 1 | 0.911/0.908/0.953 | 0.804/0.793/0.777 | 0.724/0.690/0.612 |
| G-Tu | 2 | 0.896/0.892/0.931 | 0.818/0.802/0.750 | 0.744/0.710/0.566 |
| G | 0 | 0.951/0.948/0.956 | 0.897/0.881/0.835 | 0.844/0.812/0.713 |
| G | 1 | 0.958/0.955/0.976 | 0.894/0.875/0.860 | 0.874/0.836/0.779 |
| G | 2 | 0.959/0.957/0.974 | 0.923/0.919/0.933 | 0.890/0.870/0.847 |

## Measured CPU latency (PyTorch FP32 reference, batch 1; quantized arms NOT timed)

| run | p50 ms | p95 ms | max ms | enc fraction |
|---|---|---|---|---|
| threads1|full | 0.803 | 1.645 | 41.440 | 0.88 |
| threads1|learned@0.5 | 0.677 | 1.456 | 3.412 | 0.41 |
| threads1|learned@0.33 | 0.581 | 1.452 | 5.652 | 0.28 |
| threads1|pix@0.5 | 0.556 | 1.120 | 2.191 | 0.56 |
| threads1|pix@0.33 | 0.423 | 0.921 | 2.085 | 0.34 |
| threads4|full | 0.761 | 1.297 | 8.181 | 0.88 |
| threads4|learned@0.5 | 0.586 | 1.257 | 3.455 | 0.41 |
| threads4|learned@0.33 | 0.579 | 1.203 | 4.151 | 0.28 |
| threads4|pix@0.5 | 0.586 | 1.358 | 23.105 | 0.56 |
| threads4|pix@0.33 | 0.565 | 1.303 | 10.807 | 0.34 |

## Gate summary (computed)
```json
{
 "G1_floor": {
  "macro": 0.4522222222222222,
  "NOM": 0.5177777777777778,
  "pass": false
 },
 "G2_G-Tu": {
  "ratio": 1.1216216216216217,
  "ratio_lower95_onesided": 1.080610045252794,
  "abs_macro": 0.5072222222222222,
  "relative_pass": true,
  "absolute_pass": false,
  "pass": false
 },
 "G5_long_G-Tu": {
  "picks_ratio": 1.3179190751445085,
  "pass_picks": true
 },
 "G2_G-Ta": {
  "ratio": 1.1873464373464375,
  "ratio_lower95_onesided": 1.1425163269168284,
  "abs_macro": 0.5369444444444444,
  "relative_pass": true,
  "absolute_pass": false,
  "pass": false
 },
 "G5_long_G-Ta": {
  "picks_ratio": 1.439306358381503,
  "pass_picks": true
 },
 "B_jepa_delta": {
  "delta": -0.14388888888888884,
  "ci": [
   -0.16444444444444445,
   -0.12387500000000007
  ]
 },
 "G3_G-Ta": {
  "enc_reduction": 0.5874906491788905,
  "macro_ratio": 0.9125711329539575,
  "dist_ratio": 0.9978991596638657,
  "learned_minus_best_control_macro": -0.03111111111111109,
  "pass": false
 },
 "G3_G-Tu": {
  "enc_reduction": 0.6011662518837383,
  "macro_ratio": 0.988499452354874,
  "dist_ratio": 1.0021321961620469,
  "learned_minus_best_control_macro": -0.0527777777777777,
  "pass": false
 },
 "G3_G": {
  "enc_reduction": 0.6022336266891903,
  "macro_ratio": 0.9348894348894349,
  "dist_ratio": 0.9927360774818401,
  "learned_minus_best_control_macro": -0.030277777777777792,
  "pass": false
 },
 "G4_G-Ta_vs_G-I4a": {
  "eligible_control": false,
  "bytes_ratio": 2.0050176491332254,
  "macs_ratio": 1.0150697591950903,
  "pass_analytical": false
 },
 "G4_G-Ta_vs_G-I4u": {
  "eligible_control": true,
  "bytes_ratio": 3.094691546697296,
  "macs_ratio": 0.9632567457565631,
  "pass_analytical": false
 },
 "G4_G-Ta_vs_G_full": {
  "bytes_ratio": 11.255436993910752,
  "macs_ratio": 2.7168476010948415
 },
 "G4_G-Tu_vs_G-I4a": {
  "eligible_control": false,
  "bytes_ratio": 0.8612410243535628,
  "macs_ratio": 1.0415772112685935,
  "pass_analytical": false
 },
 "G4_G-Tu_vs_G-I4u": {
  "eligible_control": true,
  "bytes_ratio": 1.3293026716687988,
  "macs_ratio": 0.9884111568611432,
  "pass_analytical": false
 },
 "G4_G-Tu_vs_G_full": {
  "bytes_ratio": 4.834692647405508,
  "macs_ratio": 2.78779514625089
 },
 "G5_probe": {
  "G": {
   "early": 0.5524203233573093,
   "late": 0.6167071155176974
  },
  "G-Tu": {
   "early": 0.657497034823809,
   "late": 0.5415224329701794,
   "late_drop_vs_G": 0.07518468254751798,
   "primary_probe_informative": true
  },
  "G-Ta": {
   "early": 0.5840456523916377,
   "late": 0.5788659730164653,
   "late_drop_vs_G": 0.03784114250123216,
   "primary_probe_informative": true
  }
 }
}
```
