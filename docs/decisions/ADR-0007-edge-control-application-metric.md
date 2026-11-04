# ADR-0007 — Application-level quality metric for the edge-video control: zone occupancy

Status: Proposed (agent proposal; the founder decides acceptance)
Date: 2026-11-04

## Context

EXP-0010 Stage A scored a COCO person detector with AP50 at IoU 0.5 against *visible-extent* boxes and found the score entangled with box semantics and class confusion. EXP-0010B (Control Adequacy Gate) needed an application-level metric for fixed industrial cameras before any temporal-execution comparison, and found that humanoid robots in the labels are a first-order issue.

## Decision (proposed)

1. The primary quality metric for the edge-video control is **zone-occupancy balanced accuracy** on a fixed image-zone grid, with the person's ground-contact point as position, POS/NEG/DONTCARE states with a boundary margin, and a block-bootstrap confidence interval (details: `analysis/control_gate/protocol_v2.json`, `zones.py`). AP50, presence and count are secondary.
2. Any quality bar is labelled **TARGET** until a customer requirement exists. The EXP-0010B bar (TPR ≥ 0.90, FPR ≤ 0.10, lower CI of balanced accuracy ≥ 0.85) is a stated engineering threshold, not a safety standard.
3. **Humanoid-robot classes (FourierGR1T2, AgilityDigit) are ignore regions** (DONTCARE zones) in the metric, as in Stage A's AP scoring, unless the founder states that the application must treat them as persons or report them separately.
4. NEG zones require no ground-truth foot (scored or ignored) within a margin larger than the prediction tolerance, and ignored persons (too small, foot outside the frame) make their zones DONTCARE (the preregistered EXP-0010B metric did not, see KI-015).
5. A fresh layout-disjoint holdout is required for any new tuned configuration, and the layout rule must screen for the class mix (humanoids, vehicles) it will be scored on.

## Alternatives Considered

- **AP50 on boxes:** sensitive to visible-vs-whole extents; Stage A showed this dominates the score.
- **Person presence:** degenerate (nearly every frame contains a person).
- **Count:** sensitive to crowding and occlusion; reported but not primary.
- **Ground-plane zones:** more physical, requires a mapping from detections to the ground plane; deferred.

## Why

Zone occupancy is what a fixed-camera monitor consumes, tolerates box-top disagreement, has a non-degenerate base rate, and cannot be won by a base-rate shortcut (a constant predictor scores exactly 0.5). EXP-0010B showed the metric is workable but also exposed its dependence on label semantics (point 3) and on a calibration-derived ground truth that passed validation on only some scenes.

## Consequences

Later quality claims on this data use this metric; the geometry-consistency rule gates which scenes can supply zone labels. Humanoid handling is a customer-semantics question the founder should answer. The preregistered EXP-0010B bar was **not** met by any tested configuration; this ADR does not claim a control exists.

## Evidence

`docs/experiments/EXP-0010B-control-adequacy-gate.md` (R1, R5, R6). Agent proposal; founder direction authorized choosing a primary task and bar but not the semantics of humanoid robots.

## Supersedes

None.

## Superseded By

None.
