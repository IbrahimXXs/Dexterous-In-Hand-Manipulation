# Student guide

Start with the visible simulation in `README.md`. The supplied actor receives a ten-frame history of sixteen measured joint positions and sixteen previous commands; it does not receive cube pose or camera images. Compare the commanded finger gait with the measured cube yaw in `outputs/simulation/yaw_trace.csv`.

Choose one route for the activity:

1. Simulation only: safe on Linux or macOS and sufficient for the student submission.
2. Hardware observation: watch the TA complete shadow and short supervised trials.
3. Both: compare the portable simulation trace with a separately recorded physical trial.

For improvement projects, read `training_source/README.md`. The complete original task and training code is supplied, but the accelerated training stack is substantially heavier than the portable viewer.

