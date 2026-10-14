# Z01_std_robocup (English translation)

Translated from `Z01_std_robocup/README.md` in
[Zvalley-Robotics/zv_robot_assets](https://github.com/Zvalley-Robotics/zv_robot_assets) at commit
`244246c` (2026-10-13). Translator's notes are marked **[TN]**.

---

RoboCup multi-robot scene: **six Z01_std_parallel_ankle robots on one soccer field**, loaded by
`src/robot_sim/sim_z01_std_robocup` (one MuJoCo model with six robots, six depth cameras and six
sets of `/robot_0X/*` topics).

**[TN]** `sim_z01_std_robocup` and `mj_sim` are in Zvalley's own simulation workspace, which is not
public.

## Generated files: don't edit by hand

```
xml/Z01_std_robocup.xml   Main file (~120 lines): global settings + six <include>s + field / lights / ball
xml/robots/robot_0X.xml   A complete sub-model per robot (~270 lines each): body + actuator/sensor/tendon/equality
```

Both are produced by the generator:

```bash
python3 robot_assets/utils/gen_robocup_multi_xml.py          # regenerate (main file + 6 fragments)
python3 robot_assets/utils/gen_robocup_multi_xml.py --check  # validate only (duplicate names / joint addresses / frustum / loading)
```

**Why six included sub-models:** MuJoCo 3.1.1 treats duplicate names as a hard error and has no
"reuse with renaming" mechanism (`<attach>` / `<replicate>` need 3.2+ / 3.4+; 3.1.1 reports
`unrecognized element`). So each robot needs its own names with a `robot_0X_` prefix. Each robot
lives in its own `robots/robot_0X.xml` (root tag `<mujoco>`, a **complete sub-model**, not a body
fragment). `<include>` merges its worldbody / actuator / sensor / tendon / equality into the
matching sections of the main model (verified on 3.1.1; bodies, actuators and sensors keep include
order and qpos order = document order). The main XML is therefore only 120 lines instead of six
near-duplicate blocks. The single source of truth for the robot is still
`Z01_std_parallel_ankle/xml/Z01_std_parallel_ankle.xml`: change it once and rerun the generator to
update all six.

## What the model is made of

| Part | Source |
| --- | --- |
| Robot body | `Z01_std_parallel_ankle/xml/Z01_std_parallel_ankle.xml`, generated into `xml/robots/robot_0X.xml` (names prefixed `robot_0X_`, including site / tendon / equality references and each robot's actuators / sensors) |
| Soccer field (turf, lines, both goals, ball, placement preview marker) | `Z01_std_football_player/xml/scene/soccer_field.xml` (**inlined**, so the ball can be put on the centre spot) |
| Textures / materials / centre-circle mesh | `<include file="../../Z01_std_football_player/xml/scene/soccer_field_assets.xml"/>` |
| Mesh files | `<compiler meshdir="../../Z01_std_football_player/meshes/"/>` (that directory holds both the robot STLs and `center_circle_ring.obj`, so no meshes are duplicated) |

Two deliberate differences from `Z01_std_football_player`:

1. **The ball sits on the centre spot `(0, 0, 0.11)`.** football_player hides it at `z = -100` and
   shows it in its "football mode"; in the RoboCup scene the ball is always visible, so the
   inlined field puts it at the centre.
2. **The placement preview marker `ball_preview` is hidden 1 m below the ground** (not at
   `z = -100`). At −100 it stretched the model's bounding box to 107 m (`stat.extent`), which in
   turn widened the depth cameras' near/far planes; at −1 m `stat.extent ≈ 23`, matching the field.

## Key figures (`--check` output)

```
nq=247 nv=240 nu=174 ncam=6 ngeom=392
robot_01..06 base free joint: qpos = 0 / 40 / 80 / 120 / 160 / 200
ball free joint:              qpos = 240
```

- Each robot has 40 qpos (7 floating base + 33 hinge joints); the six robots fill the front of
  qpos and the ball comes last. Simulation code always **looks addresses up by name** and never
  assumes offsets.
- `<size njmax="4000" nconmax="2000"/>`: with six humanoids colliding we measured `ncon ≈ 340`,
  `nefc ≈ 1600` (one robot alone: ~57 / 277), so these must be raised. The arena is therefore
  ~900 MB of **virtual** memory, but RSS is only ~95 MB (measured on the 3.1.1 library;
  `maxuse_arena ≈ 2.1 MB` after 1000 steps).

## Before regenerating

- If the robot body changes, edit `Z01_std_parallel_ankle` (or its `urdf_to_xml.yaml`) and rerun
  this generator.
- If the field changes (`soccer_field.xml`), rerun the generator too: the field is inlined, not
  included.
- After adding an XML, `mj_sim`'s `file(GLOB */xml/*.xml)` needs a reconfigure, or
  `robot_header/Z01_std_robocup.h` won't be generated:
  `colcon build --cmake-force-configure --packages-select mj_sim`.
- `header_generator.py` only reads the **first** `<actuator>` / `<sensor>` block and the first body
  with a free joint, so all names in the generated `Z01_std_robocup.h` are robot_01's. The sim node
  **doesn't** use that header for names; it builds them from the `robot_0X_` prefix plus
  `Z01_std_parallel_ankle.h` (see `RobotModelSpec` in `sim_z01_std_robocup`).
