# `test_estimation_env.py::test_actor_observation_layout`: intermittent, not caused by C5-HW-V2 (2026-11-26)

The test (unchanged since 726a4be) checks that after 15 steps the newest joint frame of the actor observation equals
the current joint position or the previous frame (observation delay 0..1). It has failed intermittently since
23 Nov (c5_hw_v1 `logs/pytest_training_suite.log`), before C5-HW-V2 existed.

Runs on gpu-04 (all four GPUs at 100% utilisation by other users), conda env `leap-estimator`:

| Run | Code | Result |
|---|---|---|
| full suite, main repo on `task2-c5-hw-v2`, 18:34 | with C5-HW-V2 | 92 passed, 1 failed (this test), `pytest_training_suite.log` |
| full suite, same tree, after training | with C5-HW-V2 | 93 passed, `pytest_training_suite_final.log` |
| full suite, `task7-consistency` (merge of tasks 1-4) | with C5-HW-V2 | 92 passed, 1 failed (this test), `pytest_training_suite_task7_merged.log` |
| single test, 3 interleaved pairs | `origin/main` / merged | main 3 passed; merged 3 failed |
| diagnostic rollout (same env, seed, 15 steps), 3 runs each | `origin/main` | 2 pass, 1 fail (env 14) |
| same | merged, C5-HW-V2 registered | 2 fail (env 14), 1 pass |
| same | merged, C5-HW-V2 registration skipped (cache hidden) | 3 pass |

In every failure the offending environment (index 14) had **not** reset; its newest joint frame differed from the
current and the previous joint position by about 0.035 rad. The number of resets in the same 15 steps varied between
runs of identical code (4 to 6), so the GPU rollout itself is not reproducible run to run.

**Does the C5-HW-V2 package change any other task?** No. The configuration of every project task
(`Mjlab-Leap-C5HW-EFFORT`, `-GRV2-C5`, `-GRV2-P0`, `-HwLim-A`, `-HwLim-C`, `-RetV1-A`, the four
`-Right-HandCube-MBZUAI-Rotate*` tasks and the four `-Scratch-*` tasks) was printed in full with and without the
package; after normalising the repository path and function addresses, the two dumps are byte-identical
(2,373,015 bytes each). Registering `Mjlab-Leap-C5HW2` builds its own configuration and mutates no shared object.

**Conclusion.** The failure is intermittent and pre-existing; it occurs on unmodified `main`. Its rate looked higher
with the C5-HW-V2 package registered in these few runs, which no configuration difference explains. The likely
cause is the test's tolerance against non-reproducible GPU rollouts. The test was not modified, skipped or fixed.
