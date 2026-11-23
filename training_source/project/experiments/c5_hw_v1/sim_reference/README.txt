{
 "generator": "experiments/c5_hw_v1/scripts/sim_actuator_reference.py",
 "task": "Mjlab-Leap-RetV1-A",
 "grasp": 49,
 "seed": 8301,
 "num_envs": 256,
 "steps": 600,
 "policies": {
  "A3": "experiments/hwlimit_ac/checkpoints/A_seed3_selected_model_2600.pt",
  "C5": "experiments/grasp_rotation_v2/checkpoints/C5_1_selected_model_2400.pt",
  "R1": "experiments/retention_v1/checkpoints/R1_selected_model_2600.pt",
  "P0": "experiments/grasp_rotation_v2/checkpoints/P0_1_selected_model_1500.pt"
 }
}
