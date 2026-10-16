# GentleHumanoid: Whole body Motion Tracking with Compliance - Inference and Deploy

[![Home Page](https://img.shields.io/badge/Project-Website-C27185.svg)](https://gentle-humanoid.axell.top/#/) 
[![arXiv](https://img.shields.io/badge/Arxiv-2511.04679-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2511.04679) 
[![Video](https://img.shields.io/badge/Video-Demo-FF0000.svg?logo=youtube)](https://www.youtube.com/watch?v=rF6N2o0IQJg)
[![Online Demo](https://img.shields.io/badge/Online-Demo-3B82F6.svg?logo=demo)](https://gentle-humanoid.axell.top/#/demo)

This is an official implementation of GentleHumanoid, more details please check our [Project](https://gentle-humanoid.axell.top) page. 

This repo provides the codebase for deploying GentleHumanoid policies on both simulation and real robots. For training code and data, please check [here](https://github.com/Axellwppr/gentle-humanoid-training).

**GentleHumanoid** learns a universal whole-body motion control policy with upper-body compliance and adjustable force limits, allowing smooth, stable, and safe interactions with humans and objects.

**Key Features**:
1. Compliance: Coordinated responses across the shoulder, elbow, and wrist for adaptive motion.  
2. Unified interaction modeling: Handles both resistive and human-guided contact forces.  
3. Safety-aware control: Supports tunable force thresholds to ensure safe human–robot interaction.  
4. Universal and Robust: Demonstrated in simulation and on the Unitree G1, generalizing across diverse motions and interactions.  

Try Our [Online Demo](https://gentle-humanoid.axell.top/#/demo)

https://github.com/user-attachments/assets/60412d44-9c35-4934-ac14-2a906f16c37c

## TODO
- [x] Release sim2sim, sim2real code 
- [x] Release pretrained model, that can deploy customized motions to G1
- [x] Release training code and data
- [ ] Release full pipeline from RGB video to G1 deployment
- [ ] Release locomotion and motion-tracking switching module
- [ ] More upon request

## Getting Started 
Clone the repo:
```
git clone https://github.com/Axellwppr/gentle-humanoid
cd gentle-humanoid
``` 

## Install 
- Create conda environment
  ```
  conda create -n gentle python=3.10
  conda activate gentle
  ```
- Install the Unitree SDK2 Python bindings in virtual environment (follow the [official Unitree guide](https://github.com/unitreerobotics/unitree_sdk2_python))
- Install Python deps:
  ```bash
  pip install -r requirements.txt
  ```
Tested on Ubuntu 22.04 with Python 3.10

## Run Sim2Sim
1. Start the simulator (state publisher + keyboard bridge):
   ```bash
   python3 src/sim2sim.py --xml_path assets/g1/g1.xml
   ```
   Leave the terminal focused so the keyboard mapping works.
2. In another terminal launch the high-level controller:
   ```bash
   python3 src/deploy.py --net lo --sim2sim
   ```
3. Flow:
   - Controller waits in zero-torque mode until it receives the simulated state.
   - Press `s` in the sim terminal to let the robot move to the default pose.
   - Press `a` in the sim terminal to start tracking policy
   - See [Motion Switching](#motion-switching) to replay different motions.
   - Use `u` and `d` to increase/decrease the force threshold (default 10N).
   - Press `x` to exit gracefully.

You can double-click a link in the simulation window and Ctrl + right-drag to apply an external force to that link.

### Configurable Impulse Testing

After the tracking policy has started, open a third terminal to configure and
apply a repeatable force pulse:

```bash
python3 src/impulse_select.py
```

The sender starts with a 30 N force applied to `torso_link` for 0.1 s at the
body-frame origin, pointing along world `+X`. Type `help` for all commands. A
typical test is:

```text
body left_elbow_link
force 30
duration 0.1
point 0.05 0 0
direction 1 0 0
frame world
preview
fire
```

- `point` is expressed in the selected body's local frame, in metres.
- `direction` is normalized automatically and can be interpreted in either the
  `world` or `body` frame.
- The requested duration is rounded to the nearest 5 ms physics step. The
  sender reports the actual duration and impulse in N*s.
- Only one pulse can be active. Use `cancel` to stop it at the next physics
  step before firing another pulse.
- Yellow markers are previews; red markers indicate an active force pulse.
- Full 200 Hz state logs are written to `assets/logs/` by default. Logging can
  be selected explicitly with `--impulse-log` or disabled with
  `--no-impulse-log` on `sim2sim.py`. Use `--impulse-log-dir <path>` to choose
  another directory.

```bash
# Record CSV (this is also the default)
python3 src/sim2sim.py --impulse-log

# Run without creating an impulse CSV file
python3 src/sim2sim.py --no-impulse-log
```

The paper's simulated per-link force range is capped at 30 N. The impulse tool
allows larger values for robustness experiments, but reports them as
out-of-distribution tests. This simulator-only tool is not a safety mechanism
for a real robot. The existing Ctrl + right-drag force remains independently
limited to 30 N.

The UDP endpoint defaults to `127.0.0.1:28563`. If it needs to be changed, pass
matching `--impulse-host` and `--impulse-port` options to `sim2sim.py`, and
`--host` and `--port` options to `impulse_select.py`.

https://github.com/user-attachments/assets/19c929eb-9731-4734-b021-d22356a839c9

## Run Sim2Real
Please read [official document](https://support.unitree.com/home/en/G1_developer/remote_control) before you work on G1. 
1. Power on G1 and connect to your PC. 
2. Turn on Controller, press L2+R2 for debugging mode. Run Sim2Real: 
   ```bash
   python3 src/deploy.py --net <robot_iface> --real
   ```
3. The state machine matches Sim2Sim but with `physical remote controller` input
   - Zero torque
   - (Press `start`) → move to default pose
   - Place robot on the ground
   - (Press `A`) → run the active policy
   - See [Motion Switching](#motion-switching) to replay different motions.
   - Use `up` and `down` buttons to increase/decrease the force threshold (default 10N).
   - (Press `select`) → exit gracefully

⚠️ Please always test motions in Sim2Sim before deploying to real robot.  
⚠️ Please do not blindly trust the RL policy. Always have emergency stop (`select`) ready.

## Motion Switching
- The tracking policy accepts motion change commands while it is active.
- Open a terminal and run the motion selector:
  ```bash
  python3 src/motion_select.py
  ```
- Usage tips:
  - Type the motion name or its index (`list` prints the menu). Press Enter with an empty line to resend the previous choice.
  - `r` reloads the YAML file if you edit it; `q` exits the selector.
- Selection rules:
  - The policy only starts a new motion when the current clip has finished and the robot is in the `default` clip (or you explicitly request `default`).
  - Sending `default` always fades back to the idle pose.

https://github.com/user-attachments/assets/25c3596e-5b79-45cd-a4fd-3974b314e01d

## Acknowledgments

- Visualization is built upon [MuJoCo](https://mujoco.org/) and [MUJOCO WASM](https://github.com/zalo/mujoco_wasm
).  
- RL framework from [active-adaptation](https://github.com/btx0424/active-adaptation). 
- Sim2Real implementation is based on [Stanford-TML/real_g1](https://github.com/Stanford-TML/real_g1).  
- Motion retargeting uses [GMR](https://github.com/YanjieZe/GMR).  
- SMPL-X motion estimation from video uses [PromptHMR](https://github.com/yufu-wang/PromptHMR).  
- Training datasets include [AMASS](https://amass.is.tue.mpg.de/), [InterX](https://liangxuy.github.io/inter-x/), and [LAFAN](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).  

<!-- ## LICENSE -->
