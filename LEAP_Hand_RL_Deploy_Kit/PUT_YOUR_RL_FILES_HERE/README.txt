Copy your RL training output here, for example from
  logs/rsl_rl/<experiment>/<run>/

  model_XXXX.pt          required (if there are several, the highest iteration is used)
  params/env.yaml        strongly recommended (or env.yaml directly in this folder)
  params/agent.yaml      recommended

Then run:
  ./leap.sh inspect
  ./leap.sh check
  ./leap.sh shadow
  ./leap.sh run

To keep several policies side by side, put each in its own folder and pass the
path:  ./leap.sh shadow /path/to/policy_folder
