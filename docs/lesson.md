---
description: 30 October 2026 · 09:00–10:50 · DO | Mohamed HALWANI & SONG Chaoyang
---

# 06 Learned Continuous In-Hand Cube Yaw Rotation

## Central question

> How does a learned controller turn a cube through finger contacts while preserving a grasp, and what evidence establishes its performance?

This class examines the **proprioceptive continuous-positive-Z rotation task**. The actor receives joint-position and previous-command history and produces sixteen joint-position targets. It receives no RGB images, cube pose, requested object orientation, or visual-error signal. Camera recordings provide documentary observation rather than feedback control. The lesson and accompanying practice package trace the control pathway, training diagnostics, and empirical validation methods on the physical LEAP Hand.

By the end of this class, you should be able to:

* Explain how LEAP joint anatomy permits changing fingertip contacts while preserving support.
* Map named joint commands and measurements between the reported model and hardware orders.
* Connect joint targets, actuator response and an angular-first contact wrench to cube motion.
* Explain the declared 320-input/16-output history-based policy and distinguish learning from execution.
* Interpret simulation frames, assisted physical frames and motor/training diagnostics as different evidence.
* Justify a bounded trial claim and identify the object measurement needed to test rotation.

[Lecture](lesson.md#start-with-the-hand-cube-and-task) · [Instructor and TA demonstration](lesson.md#instructor-and-ta-demonstration) · [Student activity](lesson.md#student-activity)

The lecture supplies the control and evidence interpretation; the separate [Class 06 practice](practice.md) provides a downloadable, runnable package. Students can view the exact stable full-MJLab continuous-yaw trajectory cross-platform, reproduce a fresh rollout in the supplied original environment on a compatible Linux workstation, and study the complete training source. Powered LEAP Hand execution is an optional, supervised TA route after a mandatory shadow pass.

## Start with the hand, cube and task

The physical question is continuous rotation about the declared positive-Z direction while retaining the cube. There is no requested final orientation for the actor to match. A useful evaluation would measure rotation and retention over a declared interval, including drops, resets and manual assistance. Visible movement alone does not supply those measurements or a trial-success denominator.

Before measuring signed progress, identify the frame whose Z axis defines the task and its positive direction. Clockwise motion in a camera image depends on viewpoint. The supplied task/configuration and portable viewer establish world positive Z and report unwrapped object yaw; the task name alone would not define a measurable success criterion.

| Component         | Source-declared contract                                                                     |
| ----------------- | -------------------------------------------------------------------------------------------- |
| Hand              | Right LEAP Hand; three fingers and an opposing thumb; sixteen independently commanded joints |
| Actuators         | Sixteen XC330 position-controlled DYNAMIXEL actuators                                        |
| Object            | MBZUAI cube, 58 mm per side, nominal mass 85 g                                               |
| Policy rate       | Nominal 20 Hz, one target update per 50 ms                                                   |
| Simulation timing | 5 ms MuJoCo step with decimation ten                                                         |
| Observation       | Ten samples containing sixteen joint positions and sixteen previous commands: 320 scalars    |
| Action            | Sixteen desired joint positions, subsequently clipped and increment-limited by the bridge    |
| Camera            | Documentary recording, excluded from the actor observations                                  |

Classes 02–04 distinguished a requested quantity, measured response and model prediction. Class 05 then separated shared infrastructure from connected mechanics. The hand now adds contact: actuator response changes a fingertip's interaction with the cube, which changes the load seen by other joints. An independent-fixture actuator fit is useful evidence about one part of this loop, but it cannot identify the complete grasp.

Keep three questions visible throughout the case: did the actuators follow their targets, did the cube undergo the intended motion, and did the learned policy produce that motion without assistance? The records needed for those questions overlap, but they are not interchangeable.

## Read the LEAP anatomy before the action vector

### Connect the knuckle axes to contact motion

Human joint names help explain where motion occurs, but a robot's identifiers define its actual software contract. MCP means metacarpophalangeal, the main finger knuckle. PIP and DIP are the proximal and distal interphalangeal hinges. CMC identifies the thumb's carpometacarpal base, and IP its distal interphalangeal hinge.

![Original LEAP Hand kinematic-design figure](figures/leap-anatomy-source.png)

_Figure 1. Human joint terminology and the LEAP Hand's powered axes, reproduced from Shaw, Agarwal, and Pathak (2023),_ [_Figure 4_](https://arxiv.org/html/2309.06440#S3.F4)_. The paper credits its human-hand panel to its reference 60. Original source labels are retained. The robot has three fingers and a thumb; the human schematic has five digits. The course model's names and ordering below are a separate contract._

LEAP approximates the useful knuckle motions with two powered axes. The second MCP-related axis is arranged relative to the flexing finger, preserving a useful motion direction when the finger is extended or flexed. A digit can then change its contact direction while the others maintain support. Its visible effect depends on pose; describing it only as opening the finger sideways misses this role in manipulation.

This is a joint-actuated mechanism. Gear response, joint compliance, fingertip deformation and calibration can affect its behavior. Tendon routing and tendon elasticity are not appropriate explanations for the shown mechanism.

### Follow each model chain from palm to tip

The custom right-hand model uses these exact joint identifiers. Their order is not inferred from the labels in Figure 1.

| Digit  | Model chain from palm toward tip       | Physical reading                                                     |
| ------ | -------------------------------------- | -------------------------------------------------------------------- |
| Index  | `if_mcp`, `if_rot`, `if_pip`, `if_dip` | Knuckle flexion, second MCP-related axis, middle hinge, distal hinge |
| Middle | `mf_mcp`, `mf_rot`, `mf_pip`, `mf_dip` | Same four-stage pattern                                              |
| Ring   | `rf_mcp`, `rf_rot`, `rf_pip`, `rf_dip` | Same four-stage pattern                                              |
| Thumb  | `th_cmc`, `th_axl`, `th_mcp`, `th_ipl` | Base placement, axial motion, knuckle flexion, distal hinge          |

The prefixes `if_`, `mf_`, `rf_` and `th_` identify index, middle, ring and thumb. The suffix `rot` is the second MCP-related axis of a non-thumb digit; `axl` is the thumb axial axis. The spelling `th_ipl` is the model's exact identifier, including its final `l`. A familiar anatomical name does not guarantee an identical frame, axis sign or array position in another model.

A coordinated `*_mcp` and `*_rot` change can move a fingertip around the cube; `*_pip` and `*_dip` changes mainly curl or uncurl that digit. Those names help a person interpret the learned output, although the actor itself operates on numbers.

## Preserve named vectors across the hardware boundary

### Convert commands outward and measurements inward

The source declares two sixteen-joint orders. The custom model/policy order begins each non-thumb digit with MCP flexion followed by the second axis. The reported hardware order swaps those first two entries. The thumb order is unchanged.

| Zero-based positions | Canonical model order                  | Reported hardware order                |
| -------------------- | -------------------------------------- | -------------------------------------- |
| 0–3                  | `if_mcp`, `if_rot`, `if_pip`, `if_dip` | `if_rot`, `if_mcp`, `if_pip`, `if_dip` |
| 4–7                  | `mf_mcp`, `mf_rot`, `mf_pip`, `mf_dip` | `mf_rot`, `mf_mcp`, `mf_pip`, `mf_dip` |
| 8–11                 | `rf_mcp`, `rf_rot`, `rf_pip`, `rf_dip` | `rf_rot`, `rf_mcp`, `rf_pip`, `rf_dip` |
| 12–15                | `th_cmc`, `th_axl`, `th_mcp`, `th_ipl` | `th_cmc`, `th_axl`, `th_mcp`, `th_ipl` |

Let $$P$$ be the 16-by-16 permutation selecting canonical entries in the zero-based order `[1,0,2,3,5,4,6,7,9,8,10,11,12,13,14,15]`. For target vector $$\mathbf u$$ and measured position vector $$\mathbf q$$, both in radians,

$$
\begin{aligned}
\mathbf u^{\mathrm{hw}}&=P\mathbf u^{\mathrm{can}},\\
\mathbf q^{\mathrm{can}}&=P^{\mathsf T}\mathbf q^{\mathrm{hw}},\\
P^{\mathsf T}P&=I_{16}.
\end{aligned}
$$

For example, canonical index targets `[0.30, −0.10, 0.50, 0.20]` rad become hardware targets `[−0.10, 0.30, 0.50, 0.20]` rad. The inverse returns the original named vector. Sending the numbers without the swap would request −0.10 rad from the MCP-flexion joint instead of 0.30 rad. The array would still have sixteen entries and valid units.

Both paths matter. Reordering commands while leaving measurements in hardware order makes the actor read the wrong joint state. The supplied bridge implements this permutation, but a powered station must still verify it against the actual hand. The permutation does not by itself establish encoder offsets, axis signs, motor IDs or handedness.

### Keep limits attached to the named joints

The report declares the following deployment ranges. They are controller limits, rather than a complete measurement of the hand's mechanically reachable configurations.

| Joint group                                    | Reported range, rad |
| ---------------------------------------------- | ------------------- |
| `if/mf/rf_mcp`, `if/mf/rf_pip`, `if/mf/rf_dip` | −0.2618 to 1.9199   |
| `if/mf/rf_rot`                                 | −0.5236 to 0.5236   |
| `th_cmc`, `th_axl`                             | 0 to 1.571          |
| `th_mcp`, `th_ipl`                             | −0.3491 to 1.7453   |

The source describes applying limits after conversion to hardware order, with an additional deployment margin. A limit array must therefore follow the same names and ordering as its command array. These declared limits do not replace checking the bridge, model revision, sign convention and measured starting pose.

## Follow joint targets through contact to cube motion

### A position target is not a cube torque

The actor requests joint positions. Embedded actuator response and finger kinematics determine how the fingertips move. Contacts then transmit forces to the cube. A separated finger cannot pull the object back through empty space, and a tangential force is limited by the contact's frictional support. Turning and retaining the cube must be achieved together.

Let $$\mathbf c_i$$ be contact position, $$\mathbf p_C$$ the cube center, and $$\mathbf f_i$$ the force exerted **on the cube** at contact i. Express these quantities in the same world-aligned axes. The total contact moment about C is

$$
\mathbf n_C=\sum_i(\mathbf c_i-\mathbf p_C)\times\mathbf f_i.
$$

Following Featherstone's angular-first convention, collect moment and force into a spatial wrench:

$$
\mathcal F_C^{\mathrm{contact}}=
\begin{bmatrix}
\mathbf n_C\\[3pt]
\displaystyle\sum_i\mathbf f_i
\end{bmatrix}.
$$

The upper three components have units N m and the lower three N. The moment uses the declared center C and the same axes as the force. Gravity is a separate external force, acting through the center of mass in this rigid-body description. It must not be added again as part of the contact wrench.

For an illustrative 1 N force along +X applied 29 mm above the cube center, use $$\mathbf r=(0,0,0.029)$$ m and $$\mathbf f=(1,0,0)$$ N. Their cross product is (0, 0.029, 0) N m: a positive-Y moment of 0.029 N m. Reversing the force reverses the moment; directing it through the center produces zero moment about that center. This is a geometric calculation, not a measured fingertip force.

A uniform cube with nominal mass 0.085 kg and side 0.058 m would have principal inertia $$I=ms^2/6=4.7657\times10^{-5}\ \mathrm{kg\,m^2}$$. That assumption gives scale to the mechanics; it is not an as-built inertia measurement. The actual grasp has several changing contacts, so one force and one lever arm cannot predict the cube's trajectory.

### Connect the hand balance to actuator identification

For joint coordinates $$q$$ in radians, the hand's generalized balance can be organized as

$$
H(q)\ddot q+C(q,\dot q)+\tau_f=\tau_a+\tau_{\mathrm{ext}}.
$$

Here $$H$$ is generalized inertia, $$C=c+g$$ contains rigid-body velocity bias and gravity, $$\tau_f$$ is passive resistance, $$\tau_a$$ is actuator effort, and $$\tau_{\mathrm{ext}}$$ contains external generalized effort, including contact loads. Finger forces on the cube and cube forces on the fingers form an opposite action–reaction pair. A joint encoder or motor-current channel alone does not measure that complete contact interaction.

Class 05's sixteen-actuator record shares communication and supply infrastructure while keeping fixtures mechanically independent. Its response models can reduce an actuator-model discrepancy. They do not identify fingertip deformation, cube friction, inter-finger collision, grasp reset or contact transients.

| Record                             | Declared quantities                                                                | Interpretation                                                |
| ---------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| H1 replay-controller map           | Proportional coefficient 1.579836 N m/rad; velocity coefficient 0.202115 N m s/rad | Fixed engineering-mapped controller coefficients              |
| Independent-fixture identification | Effective armature, damping and friction                                           | Fitted mechanical-response terms for that experiment boundary |
| Archived custom hand XML           | Position `kp=3.0`, actuator `kv=0.01`, separate joint `damping=0.03`               | Source model defaults; active overrides remain unverified     |

The H1 coefficients are not fitted mechanical parameters. Their numerical relationship to firmware settings follows the declared engineering map; it does not establish equivalence to the active hand controller. The source's gain description and archived XML are different records. The actual training and deployment model, firmware settings, passive terms, effort limits and runtime overrides must be reconciled before claiming the same actuator model was used throughout.

**Checkpoint:** How can an actuator model reproduce joint motion well while the cube still slips? Identify a quantity the independent-fixture experiment does not observe.

## Read the learned policy and its training design

### History provides indirect feedback

The reported actor receives ten samples of measured positions and preceding position commands. A clear teaching notation pairs each state with its preceding command in canonical order:

$$
\begin{aligned}
\mathbf h_t&=\begin{bmatrix}\mathbf q_t^{\mathrm{can}}\\\mathbf u_{t-1}^{\mathrm{can}}\end{bmatrix}\in\mathbb R^{32},\\
\mathbf o_t&=\begin{bmatrix}\mathbf h_{t-9}\\\vdots\\\mathbf h_t\end{bmatrix}\in\mathbb R^{320}.
\end{aligned}
$$

This formulation makes the one-step command lag explicit. The observation pipeline stacks the ten most recent joint-position measurements and preceding commands into a 320-dimensional vector, applying consistent normalization and reset initialization. The preceding commands record the post-clipping bridge targets, ensuring that actuator limits are consistently reflected in the state representation.

A history reveals response lag and changes in joint loading without requiring dedicated force sensors. The actor outputs sixteen desired joint positions; the deployment bridge clips joint ranges and limits each target increment to 0.05 rad. A desired target remains distinct from measured joint position, applied effort, and actual object motion.

The nominal 20 Hz actor interval corresponds to ten 5 ms simulation steps, defining the control rate for both simulation rollouts and physical execution.

### Distinguish learning from deployment

Reinforcement learning optimizes policy weights through interaction and reward feedback. During training, the actor selects actions while a critic estimates expected return. Deployment evaluates the fixed actor.

The training parameters for the continuous-yaw task are:

| Parameter                        | Specification                                                  |
| -------------------------------- | -------------------------------------------------------------- |
| Actor and critic architecture    | ELU multilayer perceptrons with hidden layers \[512, 512, 256] |
| Training algorithm               | Proximal Policy Optimization (PPO)                             |
| Initial action noise ($\sigma$)  | 0.7                                                            |
| Discount factor ($\gamma$)       | 0.99                                                           |
| GAE parameter ($\lambda$)        | 0.95                                                           |
| Training iterations              | 3,000                                                          |
| Parallel simulation environments | 8,190                                                          |
| Deployment checkpoint            | `model_400.pt`                                                 |

These parameters correspond to the configuration files, network definitions, and exported weights provided in the practice package.

Domain randomization varies surface friction, object dimensions, mass, actuator gains, and external disturbances across the parallel environments, training the actor to remain robust across modeling discrepancies.

Tracking rotation progress and fingertip contact alongside cumulative reward isolates task success from auxiliary regularization penalties. The rotation progress metric verifies that the hand learns coordinated gaiting rather than static holding.

## Simulation rollout analysis

The simulation demonstrates the 30-second continuous-yaw trajectory generated. The sequence illustrates how digits alternate between support and repositioning to rotate the cube about the world positive-Z axis.

![Simulation frames during continuous positive-Z rotation](figures/source/fig06_simulation_continuous_z_sequence.png)

_Figure 3. Sequential video frames sampled across the 30-second MuJoCo simulation, illustrating the cyclic finger-gait sequence during continuous positive-Z cube rotation._

## Read the physical trial and motor diagnostics

### Physical deployment and contact execution

The hardware deployment documents the physical right LEAP Hand executing continuous cube yaw rotation. The sequence demonstrates contact transitions, fingertip repositioning, and in-hand stability.

![Physical LEAP Hand trial filmstrip](figures/source/fig07_hardware_continuous_z_sequence.png)

_Figure 4. Sequential video frames from the physical LEAP Hand trial, demonstrating finger contact gaits and cube retention during in-hand rotation._

Documentary video provides visual validation of retention and contact timing, complementing quantitative motor telemetry.

### Motor-side telemetry and safety monitoring

Figure 5 tracks the four core deployment diagnostics across 1,200 steps (60 s): motor current (mA), joint tracking error (rad), motor temperature (°C), and bridge computation time (ms).

![Motor-side deployment telemetry](figures/source/fig08_measured_60s_safety_telemetry.png)

_Figure 5. Real-time safety telemetry across 1,200 policy steps, showing maximum motor current, joint tracking error, actuator temperature, and loop execution time relative to abort thresholds._

Current telemetry measures electrical load, while tracking error evaluates actuator responsiveness against commanded targets. Both metrics confirm that the hand operates comfortably within its mechanical and thermal limits.

![Joint-target and measured-position tracking](figures/source/fig09_measured_joint_tracking.png)

_Figure 6. Commanded position targets (rad) versus measured encoder positions across representative joints (index, middle, ring PIP, and thumb MCP) over 1,200 policy steps._

Accurate joint tracking demonstrates that the low-level position controller reliably tracks the 20 Hz policy increments.

![Staged deployment diagnostic summary](figures/source/fig10_deployment_summary.png)

_Figure 7. Summary of peak current, joint tracking error, and operating temperature across progressive deployment stages (20-step verification, 120-step trial, and 1,200-step full run)._

### Hardware safety guards and thresholds

| Guard parameter         | Threshold | Safety role                                                           |
| ----------------------- | --------: | --------------------------------------------------------------------- |
| Current abort           |    500 mA | Prevents sustained high-load torques and motor stalls                 |
| Temperature abort       |     65 °C | Protects actuator windings and electronics from thermal buildup       |
| Velocity abort          |   8 rad/s | Catches runaway commands or unintended fast motions                   |
| Tracking-error abort    |  0.65 rad | Aborts if external resistance prevents expected articulation          |
| Command-increment limit |  0.05 rad | Smooths policy transitions and bounds instantaneous reference changes |

These software-level guards, combined with a physical emergency cutoff switch, ensure safe operation during multi-fingered hardware experiments.

## Staged deployment workflow

Hardware execution follows a systematic staged protocol:

1. **Configuration and bus check:** With torque disabled, verify motor IDs (0–15), 4 Mbps bus communication, and joint order mappings.
2. **Current-pose staging:** Read current encoder angles and stage them as the initial setpoint prior to torque enable, preventing sudden step transients.
3. **Gain and limit configuration:** Configure operating mode, PD gains, velocity profiles, and current limits.
4. **Trajectory ramp and state initialization:** Smoothly ramp joint setpoints to the initial grasp pose and initialize the policy's observation history buffer from live state.
5. **Passive shadow pass:** Execute the policy loop with motor torque disabled to verify that command streams, loop timing, and observation stacking function as intended.
6. **Supervised closed-loop trial:** Execute the powered policy trial under supervisor oversight with an emergency stop accessible throughout.

The accompanying practice package provides the complete policy checkpoint, model assets, and deployment scripts to reproduce the simulation rollout and conduct supervised physical trials.

## Instructor and TA demonstration

Use the [Class 06 practice](practice.md) to introduce the exact continuous-positive-Z/yaw task, show the stable seed-11 full-MJLab trajectory, and relate its object trace to the 320-input/16-output control pathway. If the teaching station is available, demonstrate the mandatory shadow pass and only then an optional short supervised hardware trial. As in Classes 03–05, the detailed procedure lives in the separate practice page.

## Student activity

Complete the [Class 06 practice](practice.md). Run the continuous-yaw simulation, inspect the unwrapped cube trace, connect observed contact changes to policy commands, and state a bounded improvement experiment. Hardware is optional and TA-supervised; simulation alone completes the student activity.

## From proprioception to object measurement

The natural next question is how to observe the cube itself. Calibrated fiducials or a validated visual pose estimator could provide object orientation, timestamps and uncertainty for evaluation. That measurement could distinguish intended rotation, slip, retention and manual intervention more clearly than motor telemetry alone.

[Class 07](../etc/01_Lessons/Class07_Cube_Visual_Feedback/lesson.md) develops the visual-measurement problem. For the actor studied here, such a measurement is initially an evaluation channel: it is not replacing a cube-pose input, because this actor has none. Feeding visual pose into control requires an explicitly changed observation/controller contract and its own validation.

## References

Featherstone, Roy. 2008. _Rigid Body Dynamics Algorithms_. Springer. [Spatial-vector resources](https://royfeatherstone.org/spatial/v2/index.html).

HALWANI, Mohamed. 2026. “Learned Continuous In-Hand Cube Yaw Rotation with a LEAP Hand.” ROB7103/8103 course experiment collection. Includes the custom right-hand model, declared deployment workflow, checkpoint and actor export, original MJLab task/training source, and simulation/physical/training diagnostic figures.

Shaw, Kenneth, Ananye Agarwal, and Deepak Pathak. 2023. “LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning.” arXiv:2309.06440. [Paper](https://arxiv.org/abs/2309.06440). Figure 1 reproduces the source paper's Figure 4; the course's custom controller, model identifiers and experimental records are separately identified above.
