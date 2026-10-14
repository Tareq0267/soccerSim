# Zvalley Robot SDK (English translation)

Translated from the Chinese `README.md` of
[Zvalley-Robotics/zv_robot_sdk](https://github.com/Zvalley-Robotics/zv_robot_sdk) at commit `4ce4906`
(2026-06-25, "Add IMU data format and example program"). Code blocks are unchanged except for
translated comments. Translator's notes are marked **[TN]**.

---

# Application development

## SDK overview

The Zvalley Robot SDK is a high-performance robot-control SDK built on DDS (Data Distribution
Service). It gives robot systems a standard communication and control interface. It's written in
modern C++ and uses ROS 2–compatible data formats.

## Getting the SDK

### About the SDK

The Zvalley Robot SDK is the development kit Zvalley (中科云谷) ships for its new generation of
robots. It wraps both low-level motor control and high-level motion control behind one interface,
and comes with standard interfaces and example projects so developers can extend the robot and
build on it quickly: mobile platforms, control algorithms and applications.

### Download

https://github.com/Zvalley-Robotics/zv_robot_sdk.git

Follow the README in the repository to install the SDK on your development machine.

## Quick start

### Requirements

| Item       | Recommended     |
| :--------- | :-------------- |
| OS         | Ubuntu 22.04 LTS |
| CPU        | x86_64          |
| CPU cores  | 4               |
| RAM        | 16 GB           |

**[TN]** The prebuilt libraries (`libs/cpp/lib/*.so`, `libzv_robot_sdk.a`, and the Python
extension for CPython 3.9–3.12) are **Linux x86_64 only**. There are no macOS, Windows or aarch64
builds, and no source for `libzv_robot_sdk`.

### Using the SDK

To build your own application, go into the `examples` directory, write an example program, build
it, and run the binary to check it works.

```
cd examples
mkdir build && cd build
cmake ..
make
```

The executables are written to `build/`; run them directly.

# Services

The SDK offers both **high-level services** and **low-level services**, using different
communication patterns to meet different real-time needs.

- **High-level services** control the robot's overall behaviour: mode switching, omnidirectional
  walking, composite motions, head control and so on. These are called as RPC-style
  request/response and suit one-off or low-rate commands.
- **Low-level services** handle real-time data and direct control: high-rate motor and IMU data,
  and driving the motors directly. These use DDS publish/subscribe for continuous, fast exchange.

**[TN]** Despite "RPC", the high-level interface documented below is plain ROS 2 topics
(`/motion_cmd`, `/change_cmd`). The SDK does contain `zv_api::msg::dds_::Request_`/`Response_`
types (`zv/robot/internal/`), but they aren't documented.

## High-level service interface

### DDS communication interface

#### About Cyclone DDS

DDS is a distributed communication framework implementing the OMG DDS specification. Nodes
exchange data directly by publish/subscribe without a central server, with QoS policies providing
reliability and real-time behaviour.

![](https://raw.githubusercontent.com/Zvalley-Robotics/zv_robot_sdk/main/images/dds.png)

#### Interface

`zv::robot` provides DDS communication for sending and receiving messages inside the robot or
between the robot and external applications. Main classes: `ChannelFactory`, `ChannelPublisher`,
`ChannelSubscriber`.

---

**zv::robot::ChannelFactory**

A singleton that creates channels backed by DDS topics. It must be initialised before use, which
sets up the underlying DDS configuration.

| Function | Prototype | Purpose | Notes |
|---|---|---|---|
| `Instance` | `static zv::robot::ChannelFactory* Instance()` | Get the singleton pointer | — |
| `Release` | `void Release()` | Free the factory's static resources | `ChannelFactory::Instance()->Release()` |
| `Init` | `void Init(const nlohmann::json& config)` | Initialise the factory and its DDS factory from a JSON config | — |
| `Init` | `void Init(int32_t domainId, const std::string& networkInterface = "")` | Initialise with a DDS domain id and network interface | `domainId` is the domain of the default DdsParticipant. If `networkInterface` is empty a default, auto-selected config is generated. **When developing external applications, set `enableSharedMemory` to false.** |
| `CreateSendChannel` | `CreateSendChannel<MSG>(const std::string& name)` | Create a channel that sends messages of type MSG | Returns a templated channel pointer |
| `CreateRecvChannel` | `CreateRecvChannel<MSG>(const std::string& name, std::function<void(const void*)> callback, int32_t queuelen = 0)` | Create a channel that receives MSG and calls `callback` | `queuelen` = 0 disables the queue. If the callback does heavy work, enable the queue so the DDS callback thread isn't blocked. |

---

**zv::robot::ChannelPublisher\<MSG\>**

Publishes messages of a given type. Constructor: `explicit ChannelPublisher(const std::string& channelName)`.

| Function | Prototype | Purpose |
|---|---|---|
| `InitChannel` | `void InitChannel()` | Initialise the channel for sending |
| `CloseChannel` | `void CloseChannel()` | Close the channel |
| `Write` | `bool Write(const MSG& msg, int64_t wait_microsec = 0)` | Send a message; returns true on success |

**DDS publish example – Hello World** (`examples/helloworld/publisher`)

```cpp
#include <zv/robot/channel/channel_publisher.hpp>
#include "String_.hpp"

// Topic to publish on
static const std::string HELLO_WORLD_TOPIC = "rt/hello_world";

using namespace zv::robot;
using namespace zv::common;

int main() {
    // Initialise the channel factory (once per process)
    ChannelFactory::Instance()->Init(0);

    // Create and initialise the publisher
    ChannelPublisher<std_msgs::msg::dds_::String_> publisher(HELLO_WORLD_TOPIC);
    publisher.InitChannel();
    int count = 0;
    while (true)
    {
        // Build a message and write it to the channel
        std_msgs::msg::dds_::String_ msg("HelloWorld. " + std::to_string(count++));
        publisher.Write(msg);
        // 1 Hz
        sleep(1);
    }
    return 0;
}
```

---

**zv::robot::ChannelSubscriber\<MSG\>**

Subscribes to messages of a given type. Constructor: `explicit ChannelSubscriber(const std::string& channelName)`.

| Function | Prototype | Purpose | Notes |
|---|---|---|---|
| `InitChannel` | `void InitChannel(const std::function<void(const void*)>& handler, int64_t queuelen = 0)` | Initialise the channel and register the handler | Enable the queue if the handler is slow, so the DDS callback thread isn't blocked |
| `CloseChannel` | `void CloseChannel()` | Close the channel | — |
| `GetLastDataAvailableTime` | `int64_t GetLastDataAvailableTime()` | Time the last message arrived | −1 if the channel isn't initialised; otherwise a monotonic timestamp since boot, in microseconds |

**DDS subscribe example – Hello World** (`examples/helloworld/subscriber`)

```cpp
#include <zv/robot/channel/channel_subscriber.hpp>
#include "String_.hpp"
#include <iostream>

// Topic to subscribe to
static const std::string HELLO_WORLD_TOPIC = "rt/chatter";

using namespace zv::robot;
using namespace zv::common;

// 1. Callback; the SDK calls it when a message arrives on the topic
void Handler(const void* msg)
{
    // Cast the generic pointer to the concrete String_ type
    const std_msgs::msg::dds_::String_* hello_world_msg =
        (const std_msgs::msg::dds_::String_*)msg;
    std::cout << "message:" << hello_world_msg->data_() << std::endl;
}

int main()
{
    // 2. Initialise the channel factory (manages the SDK's communication resources)
    ChannelFactory::Instance()->Init(0);

    // 3. Create the subscriber, binding topic and message type
    ChannelSubscriber<std_msgs::msg::dds_::String_> subscriber(HELLO_WORLD_TOPIC);

    // 4. Initialise the channel and register Handler
    subscriber.InitChannel(Handler);

    // 5. Keep the process alive; the subscriber calls the handler on an internal thread
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
    return 0;
}
```

**[TN]** The `rt/` prefix is how ROS 2 maps a topic name onto a raw DDS topic, so `rt/chatter` is
the ROS 2 topic `/chatter`. The DDS type names (`std_msgs::msg::dds_::String_`,
`robot_msgs::msg::dds_::AllJointCmd_`, …) also follow the ROS 2 convention, which is why the SDK
can talk to ordinary ROS 2 nodes when they use Cyclone DDS and have matching `.msg` definitions.

---

## High-level motion control interface

#### Overall design — robot state machine

![](https://raw.githubusercontent.com/Zvalley-Robotics/zv_robot_sdk/main/images/yk.png)

**[TN]** The diagram shows: `INIT → READY`; `READY → RL_GAIT` or `READY → STOP`;
`RL_GAIT → RL_RULE` or `RL_GAIT → STOP`; `RL_RULE → STOP`; `STOP → INIT`.

#### Control interfaces

**1. `/motion_cmd` — motion control**

Controls the robot's basic motion (velocity, gait, …).

Topic: `/motion_cmd`

Message type (`RemoteControlMotion.msg`):

```
string token
float32 vx
float32 vy
float32 vyaw
float32 gait_step
bool is_continous
```

| Field | Type | Meaning |
| --- | --- | --- |
| token | string | Token / authentication id |
| vx | float32 | Forward/back linear velocity (m/s), positive = forward |
| vy | float32 | Left/right linear velocity (m/s), positive = left |
| vyaw | float32 | Angular velocity (rad/s), positive = turn left |
| gait_step | float32 | Gait stride parameter |
| is_continous | bool | Whether the motion is continuous |

`is_continous` (default `false`):

| Value | Meaning |
| --- | --- |
| true | Keep executing this command |
| false | Execute for at most 1 s |

Command-line example:

```bash
# Walk straight forward at 1 m/s for 1 s
ros2 topic pub /change_cmd robot_msgs/msg/RemoteControlMotion "{vx: 1, vy:  0, vyaw: 0, gait_step: 1, is_continous: false}" -1
```

**[TN]** The example publishes to `/change_cmd`, which looks like a typo for `/motion_cmd` (it uses
the `RemoteControlMotion` type). `is_continous` is spelled that way in the message.

**2. `/change_cmd` — switching**

Switches the robot's:

- state
- motion model
- identity

Topic: `/change_cmd`

Message type (`RemoteControlChange`):

```
string key
string val
```

Supported `key` values:

| key | Meaning |
| --- | --- |
| status | Switch robot state |
| model | Switch motion model |
| name | Set the robot's identity |

`status` values:

| Value | Meaning |
| --- | --- |
| init | Initialising |
| ready | Ready |
| rl_gait | Model-based (RL policy) |
| rl_rule | Rule-based |
| stop | Stop |

`model` values (enum `ModelType`), used to switch the robot's motion model:

| Model | Meaning |
| --- | --- |
| stil | Stand still |
| stand | Stand |
| run | Run |
| hand2 | Wave |
| mimic | Imitate motion |
| hiphop | Hip-hop dance |
| mojito | "Mojito" dance |
| warmup | Warm-up |
| newdance | New dance |
| kick | Kick |
| beyond | "Beyond" dance |
| year | New Year greeting |
| shrug | Shrug |
| bow | Bow |
| salute | Salute |
| gentleman | Gentleman gesture |
| introduce | Self-introduction |
| dualhello | Two-person greeting |
| show | Show-off motion |
| flat | Balance motion |

## Low-level service interface

Low-level communication handles data exchange between the user's PC and the robot body, over DDS.

#### Interface

Subscribe and publish with the DDS interface described above. Topic data is stored in
IDL-defined structs. Common structs:

| Struct | Meaning |
| --- | --- |
| `JointCmd_` | Command for one joint: target position, velocity, torque and gains |
| `JointState_` | Feedback for one joint: position, velocity, torque and status flags |
| `AllJointCmd_` | Container for a whole-robot joint command: metadata (timestamp, sequence number) plus one `JointCmd_` per joint |
| `AllJointState_` | Container for whole-robot joint feedback: metadata plus the list of joint states |
| `String_` | A single string field, for simple text such as debug output or status |
| `Time_` | Timestamp base type: seconds (`sec`) and nanoseconds (`nanosec`) |

**Low-level command example** (`examples/lowcmd/publisher.cpp`)

Publishes whole-body joint commands (`AllJointCmd`) to `rt/all_joint_cmd`; each message holds the
target position, velocity, torque and gains of every joint.

```cpp
#include <thread>
#include <zv/robot/channel/channel_publisher.hpp>
#include "AllJointCmd_.hpp"

static const std::string LOW_CMD_TOPIC = "rt/all_joint_cmd";

using namespace zv::robot;
using namespace zv::common;

int main()
{
    // 1. Initialise the channel factory (global communication resources)
    ChannelFactory::Instance()->Init(0);
    // 2. Create the publisher, binding topic and message type
    ChannelPublisher<robot_msgs::msg::dds_::AllJointCmd_> publisher(LOW_CMD_TOPIC);
    // 3. Initialise the channel
    publisher.InitChannel();

    while (true)
    {
        // 4. Build an AllJointCmd_ message
        robot_msgs::msg::dds_::AllJointCmd_ msg;
        // 5. Per-joint control parameters
        const std::vector<double> joint_pos { /* 23 values, see the repo */ };
        const std::vector<double> kp { /* 23 values */ };
        const std::vector<double> kd { /* 25 values */ };
        // 6. Container for the joint commands
        auto& cmds = msg.joint_cmds_();
        cmds.clear();
        cmds.reserve(joint_pos.size());
        // 7. One command per joint
        for (int i = 0; i < 23; i++)
        {
            cmds.emplace_back();
            auto& c = cmds.back();
            c.joint_pos_(joint_pos[i]);
            c.joint_vel_(0);
            c.joint_torque_(0);
            c.kp_(kp[i]);
            c.kd_(kd[i]);
        }
        // 8. Publish
        publisher.Write(msg);
        std::cout << "test [tx] " << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return 0;
}
```

**[TN]** The example sends **23** joint commands, but the Z01 models have 29 (or 33 with the
parallel ankle) joints, and its `kd` list has 25 entries. The joint order of `rt/all_joint_cmd` is
not documented anywhere. Ask Zvalley before using this on a real robot.

**Low-level state example** (`examples/lowcmd/subscriber.cpp`)

Subscribes to joint states (`AllJointState`) on `rt/all_joint_state` and prints each joint's
position and ids in the callback.

```cpp
#include <chrono>
#include <thread>
#include <zv/robot/channel/channel_subscriber.hpp>
#include "AllJointState_.hpp"

// Topic (whole-robot joint state)
static const std::string LOW_STATE_TOPIC = "rt/all_joint_state";

using namespace zv::robot;
using namespace zv::common;

// 1. Callback, called by the SDK when an AllJointState_ arrives
void Handler(const void* msg)
{
    const robot_msgs::msg::dds_::AllJointState_* joint_state_msg =
        (const robot_msgs::msg::dds_::AllJointState_*)msg;
    // Timestamp, sequence number and "used for control" flag
    int64_t timestamp = joint_state_msg->timestamp_();
    int64_t index = joint_state_msg->index_();
    uint8_t used_for_ctrl = joint_state_msg->used_for_ctrl_();
    // Joint states
    std::vector<robot_msgs::msg::dds_::JointState_> joint_states = joint_state_msg->joint_states_();
    std::cout << "[Subscriber] Message received msg: " << index << ", timestamp: " << timestamp
              << ", used_for_ctrl: " << static_cast<int>(used_for_ctrl) << std::endl;
    // Print every joint (cast uint8_t to int so it isn't printed as a char)
    for (int i = 0; i < joint_states.size(); i++)
    {
        const auto& js = joint_states[i];
        std::cout << "    joint " << i << " pos: " << js.joint_pos_()
                  << ", model_id: " << static_cast<int>(js.model_id_())
                  << ", motor_id: " << static_cast<int>(js.motor_id_())
                  << ", joint_error: " << static_cast<int>(js.joint_error_()) << std::endl;
    }
}

int main()
{
    // 2. Initialise the channel factory
    ChannelFactory::Instance()->Init(0);
    // 3. Create the subscriber
    ChannelSubscriber<robot_msgs::msg::dds_::AllJointState_> subscriber(LOW_STATE_TOPIC);
    // 4. Initialise the channel and register the callback
    subscriber.InitChannel(Handler);
    // 5. Keep the process alive
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
    return 0;
}
```

**IMU / navigation subscribe example** (`examples/imu_nav/subscriber.cpp`)

Subscribes to fused IMU + integrated-navigation data (`NavAll_`) on `rt/nav_all` and reads
acceleration, angular velocity, Euler angles, quaternion, GNSS longitude/latitude/altitude, ENU
velocity, UTC time, fusion/GNSS status, temperature and pressure.

```cpp
static const std::string NAV_ALL_TOPIC = "rt/nav_all";

void Handler(const void* msg)
{
    const yesense_interface::msg::dds_::NavAll_* nav_msg =
        (const yesense_interface::msg::dds_::NavAll_*)msg;
    const auto& acc = nav_msg->acc_();
    const auto& gyro = nav_msg->gyro_();
    const auto& euler = nav_msg->euler_();
    const auto& quat = nav_msg->quat_();
    // ... pos_, vel_, status_, temp_, pressure_, utc_, tid_ ...
}

int main()
{
    ChannelFactory::Instance()->Init(0);
    ChannelSubscriber<yesense_interface::msg::dds_::NavAll_> subscriber(NAV_ALL_TOPIC);
    subscriber.InitChannel(Handler);
    while (true) std::this_thread::sleep_for(std::chrono::seconds(10));
}
```

> This example is the `imuNavSubscriber` target in `examples/CMakeLists.txt`.

#### Message types

**Single joint command (`JointCmd_`)**

| Field | Type | Meaning |
| --- | --- | --- |
| joint_pos_ | double | Target position |
| joint_vel_ | double | Target velocity |
| joint_torque_ | double | Target (feed-forward) torque |
| kp_ | double | Position gain |
| kd_ | double | Velocity gain |

**Single joint state (`JointState_`)**

| Field | Type | Meaning |
| --- | --- | --- |
| model_id_ | uint8_t | Model / part id |
| motor_id_ | uint8_t | Motor id |
| joint_pos_ | double | Joint position |
| joint_vel_ | double | Joint velocity |
| joint_torque_ | double | Joint torque |
| temp_motor_ | int32_t | Motor temperature |
| temp_mos_ | int32_t | MOSFET temperature |
| error_ | int32_t | Error code |
| joint_error_ | uint8_t | Joint error flag |

**[TN]** The original table labels the state's `joint_pos_`/`joint_vel_`/`joint_torque_` as
"target" values (copied from `JointCmd_`); in a state message they're measured values.
`joint_error_` constants from the header: `JOINT_NORMAL=0, JOINT_ERROR=1, JOINT_INTER_DISORDER=2,
JOINT_INVALID_ANGLE=3, JOINT_INVALID_CMD=4, JOINT_ZERO_ABSENCE=5`.

**Whole-robot joint command (`AllJointCmd_`)**

| Field | Type | Meaning |
| --- | --- | --- |
| timestamp_ | int64_t | Time the command was generated |
| index_ | int64_t | Sequence number |
| cmd_type_ | uint8_t | Control mode |
| joint_cmds_ | std::vector<JointCmd_> | One command per joint |

**Whole-robot joint state (`AllJointState_`)**

| Field | Type | Meaning |
| --- | --- | --- |
| timestamp_ | int64_t | Time the state was generated |
| index_ | int64_t | Sequence number |
| used_for_ctrl_ | uint8_t | "Used for control" flag |
| joint_states_ | std::vector<JointState_> | One state per joint |

**Remote-control parameter change (`RemoteControlChange_`)**

| Field | Type | Meaning |
| --- | --- | --- |
| key_ | std::string | Parameter key |
| val_ | std::string | Parameter value |

**Remote-control motion (`RemoteControlMotion_`)**

| Field | Type | Meaning |
| --- | --- | --- |
| token_ | std::string | Token / authentication id |
| vx_ | float | Linear velocity along X |
| vy_ | float | Linear velocity along Y |
| vyaw_ | float | Yaw rate |
| gait_step_ | float | Gait stride |
| is_continous_ | bool | Continuous-motion flag |

#### IMU / navigation message types

These live in namespace `yesense_interface::msg::dds_` and carry fused IMU + navigation data
(topic `rt/nav_all`).

**`NavAll_`**

| Field | Type | Meaning |
| --- | --- | --- |
| tid_ | Tid_ | Frame / thread id |
| acc_ | ThreeAxis_ | 3-axis acceleration |
| gyro_ | ThreeAxis_ | 3-axis angular velocity |
| euler_ | EulerAngle_ | Euler angles |
| quat_ | Quat_ | Quaternion |
| temp_ | SensorTemp_ | Sensor temperature |
| pos_ | GnssPos_ | GNSS position |
| status_ | NavStatus_ | Navigation status |
| vel_ | Vel_ | Velocity (east-north-up) |
| utc_ | Utc_ | UTC time |
| pressure_ | Pressure_ | Air pressure |

**Sub-structs**

| Struct | Fields |
| --- | --- |
| Tid_ | tid_ (uint16_t) |
| ThreeAxis_ | x_ / y_ / z_ (float) |
| EulerAngle_ | pitch_ / roll_ / yaw_ (float) |
| Quat_ | q0_ / q1_ / q2_ / q3_ (double) |
| GnssPos_ | longitude_ (double), latitude_ (double), altitude_ (float) |
| Vel_ | vel_e_ / vel_n_ / vel_u_ (float) |
| NavStatus_ | fusion_status_ (uint8_t), gnss_status_ (uint8_t) |
| SensorTemp_ | temp_ (float) |
| Utc_ | year_ (uint16_t), month_/day_/hour_/min_/sec_ (uint8_t), ms_ (uint32_t) |
| Pressure_ | val_ (float) |

---

## [TN] Types in the SDK that the README doesn't document

Read from the generated headers in `libs/cpp/include/zv/idl/`:

| Type (DDS name) | Fields |
| --- | --- |
| `robot_msgs::msg::dds_::RemoteControlHead_` | `pos_roll` (float), `pos_pitch` (float), `is_absolute` (bool) |
| `robot_msgs::msg::dds_::HeadControl_` | `header` (ZVHeader_), `change_motion_state` (uint8), `motion_type` (uint8), `target` (float[]), `velocity` (float) |
| `robot_msgs::msg::dds_::RobotState_` | `header`, `robot_state` (uint8), `err_info` (string). States: `INITIALIZING=0, ERROR=1, STOP=2, CHECK_MOTORS=3, DO_READY=4, READY=5, MODEL=6, RULE=7, CUSTOM_MODEL=8, CUSTOM_RULE=9` |
| `robot_msgs::msg::dds_::HeadJointCmd_` / `LegJointCmd_` / `ArmJointCmd_` / `WaistJointCmd_` | Same layout as `AllJointCmd_`, per body part |
| `yesense_interface::msg::dds_::RobotLord_` | `tid`, `acc`, `gyro`, `quat` (a slimmer IMU message) |
| `sensor_msgs` `Image_` / `CompressedImage_`, `nav_msgs` `OccupancyGrid_`, `geometry_msgs` `PoseStamped_` | Standard ROS 2 layouts |

The topic names for head control, robot state and the camera are not documented.

The Python binding (`zv_robot_sdk_python`) only exposes publishers for `AllJointCmd_` and
`String_`, and subscribers for `AllJointState_`, `NavAll_`, `OccupancyGrid_`, `PoseStamped_` and
`String_`.
