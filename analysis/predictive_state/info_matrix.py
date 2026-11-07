"""EXP-0013 Phase 1.1: controller information-access matrix (single source of truth; tests/test_controllers_obs.py checks it against the code).

Codes: Y = receives it; N = does not; "img" = only as recovered from the image by the controller itself; "ray" = a local ray-cast
through the true wall map (privileged sensor, range <= 0.25); "hand" = hand-coded assumption, not read from the simulator;
"derived" = computed by the controller from what it holds. A controller is OBSERVATION-MATCHED only if every privileged column is N
(rows marked OBS). PRIVILEGED rows are diagnostics and upper bounds, never evidence that a deployable agent can do the same.
Usage: python -m analysis.predictive_state.info_matrix [OUT.json]  (prints the markdown table)
"""
import json
import sys

COLS = ["rgb_obs", "proprio", "task_spec", "obs_history", "persistent_memory", "full_wall_geometry", "true_dynamics",
        "priv_object_positions", "layout_id", "geodesic_value_map"]
PRIVILEGED_COLS = ["full_wall_geometry", "true_dynamics", "priv_object_positions", "layout_id", "geodesic_value_map"]

M = {
    "greedy":                    {"class": "PRIV", "rgb_obs": "N", "proprio": "Y (reads env.ee/held == proprio)", "task_spec": "Y", "obs_history": "N", "persistent_memory": "N", "full_wall_geometry": "N", "true_dynamics": "N", "priv_object_positions": "Y", "layout_id": "N", "geodesic_value_map": "N"},
    "reactive (wall-follower)":  {"class": "PRIV", "rgb_obs": "N", "proprio": "Y (reads env.ee/held == proprio)", "task_spec": "Y", "obs_history": "N", "persistent_memory": "N (mode/side counters only)", "full_wall_geometry": "ray", "true_dynamics": "N", "priv_object_positions": "Y", "layout_id": "N", "geodesic_value_map": "N"},
    "reactive_memory":           {"class": "PRIV", "rgb_obs": "N", "proprio": "Y (reads env.ee/held == proprio)", "task_spec": "Y", "obs_history": "N", "persistent_memory": "Y (flip counts, retreat)", "full_wall_geometry": "ray", "true_dynamics": "N", "priv_object_positions": "Y", "layout_id": "N", "geodesic_value_map": "N"},
    "oracle_descent":            {"class": "PRIV", "rgb_obs": "N", "proprio": "Y (reads env.ee/held == proprio)", "task_spec": "Y", "obs_history": "N", "persistent_memory": "N", "full_wall_geometry": "Y", "true_dynamics": "Y", "priv_object_positions": "Y", "layout_id": "Y", "geodesic_value_map": "Y"},
    "oracle_shoot (geodesic cost)": {"class": "PRIV", "rgb_obs": "N", "proprio": "Y (reads env.ee/held == proprio)", "task_spec": "Y", "obs_history": "N", "persistent_memory": "N", "full_wall_geometry": "Y", "true_dynamics": "Y", "priv_object_positions": "Y", "layout_id": "Y", "geodesic_value_map": "Y"},
    "oracle_shoot (euclid cost)":   {"class": "PRIV", "rgb_obs": "N", "proprio": "Y (reads env.ee/held == proprio)", "task_spec": "Y", "obs_history": "N", "persistent_memory": "N", "full_wall_geometry": "Y (through la_move only)", "true_dynamics": "Y", "priv_object_positions": "Y", "layout_id": "N (field built but unused)", "geodesic_value_map": "N"},
    "planning_diag planners":    {"class": "PRIV", "rgb_obs": "N", "proprio": "Y (reads env.ee/held == proprio)", "task_spec": "Y", "obs_history": "N", "persistent_memory": "N", "full_wall_geometry": "Y (through la_move)", "true_dynamics": "Y", "priv_object_positions": "Y", "layout_id": "Y if geodesic cost else N", "geodesic_value_map": "Y if geodesic cost else N"},
    "obs_A_single_frame":        {"class": "OBS", "rgb_obs": "Y (current frame only)", "proprio": "Y", "task_spec": "Y", "obs_history": "N", "persistent_memory": "N", "full_wall_geometry": "img", "true_dynamics": "hand (kinematic step rule, parsed-map collisions)", "priv_object_positions": "N (img)", "layout_id": "N", "geodesic_value_map": "derived (from parsed map, every step)"},
    "obs_B_map_memory":          {"class": "OBS", "rgb_obs": "Y", "proprio": "Y", "task_spec": "Y", "obs_history": "Y (via memory)", "persistent_memory": "Y (hand-coded: wall OR-map, last target, goal bbox)", "full_wall_geometry": "img", "true_dynamics": "hand (kinematic step rule, parsed-map collisions)", "priv_object_positions": "N (img)", "layout_id": "N", "geodesic_value_map": "derived (from remembered map)"},
    "FUTURE learned agent (Phase 2, not built)": {"class": "OBS", "rgb_obs": "Y", "proprio": "Y", "task_spec": "Y", "obs_history": "Y (via state)", "persistent_memory": "Y (learned belief state)", "full_wall_geometry": "img (learned)", "true_dynamics": "learned", "priv_object_positions": "N", "layout_id": "N", "geodesic_value_map": "N unless a value head is trained (then label it)"},
}


def is_privileged_free(row):
    return all(str(row[c]).startswith("N") or str(row[c]).startswith(("img", "hand", "derived", "learned")) for c in PRIVILEGED_COLS)


def markdown():
    head = "| controller | class | " + " | ".join(COLS) + " |"
    sep = "|" + "---|" * (len(COLS) + 2)
    rows = [f"| {k} | {v['class']} | " + " | ".join(str(v[c]) for c in COLS) + " |" for k, v in M.items()]
    return "\n".join([head, sep] + rows)


if __name__ == "__main__":
    print(markdown())
    if len(sys.argv) > 1:
        json.dump({"columns": COLS, "privileged_columns": PRIVILEGED_COLS, "matrix": M}, open(sys.argv[1], "w"), indent=1)
