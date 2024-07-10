# PyTorch-Deep-Reinforcement-Learning-Framework-for-Autonomous-Moon-Landings
Implementing the Deep Deterministic Policy Gradient (DDPG) algorithm in PyTorch, utilizing OpenAI Gym for environment simulation, to develop reinforcement learning strategies for autonomous and efficient moon landings.

## Project Overview

### Deep Deterministic Policy Gradient (DDPG) Agent for Continuous Control in Lunar Lander Environment
By May Cooper
<br>
![lunar_lander](https://github.com/MayCooper/PyTorch-Deep-Reinforcement-Learning-Framework-for-Autonomous-Moon-Landings/assets/82129870/87950feb-1d09-4350-8cec-ffa10af02bdc)

This project implements a Deep Deterministic Policy Gradient (DDPG) agent to solve the Lunar Lander Continuous-v2 environment from OpenAI Gym. The DDPG algorithm is an actor-critic method designed for environments with continuous action spaces, making it suitable for complex control tasks like lunar landing.

### Components

#### Experience Replay Buffer
- **TransitionMemory**: A buffer to store and sample transitions (state, action, reward, next state, done) to break the temporal correlations during training and stabilize the learning process.

#### DDPG Architecture
- **PolicyNetwork (Actor)**: Responsible for selecting actions based on the current state.
- **ValueNetwork (Critic)**: Evaluates the actions taken by the actor by estimating the Q-values.

#### DDPG Agent Class
- **DDPGAgent**: Manages the actor and critic networks, handles action selection with added exploration noise, and updates the networks based on sampled experiences from the replay buffer.

#### Reinforcement Learning Environment Setup and Training Loop
- Sets up the Lunar Lander environment, initializes hyperparameters, and runs the training loop. The agent interacts with the environment, stores experiences in the replay buffer, and updates the networks.

#### Plotting and Statistics Calculation
- Functions and scripts to calculate running averages of rewards and episode lengths, and plot these statistics to visualize the agent's performance over time.

### Lunar Lander Environment Overview

**LunarLander** is a challenging environment provided by OpenAI Gym. In this environment, the objective is for the agent to safely navigate a lunar lander to a designated landing pad. The agent's performance is evaluated through a reward system:

- **Navigational Reward**: The agent receives positive rewards for successfully navigating towards the landing pad and turning off the engine at the appropriate times.
- **Safe Landing Reward**: A substantial reward is granted for achieving a safe landing on the pad, indicating the agent has effectively learned to control the lander.
- **Crash Penalty**: A significant negative reward is given for an unsafe landing or crash, encouraging the agent to avoid such outcomes.
- **Engine Usage Penalty**: To promote efficient use of the lander's engines, the agent incurs a small negative reward each time the engines are used.

#### Discrete and Continuous Versions
The LunarLander environment is available in two versions:

- **Discrete Version**: In this version, the agent selects from a set of discrete actions to control the lander's engines.
- **Continuous Version**: This version offers a more complex challenge where the agent must control two engines with continuous action values ranging from -1 to 1. This requires the agent to determine the precise thrust needed for each engine to achieve a safe landing.

The continuous version is particularly suited for testing advanced reinforcement learning algorithms like Deep Deterministic Policy Gradient (DDPG), which can handle the nuances of continuous action spaces effectively.

### DDPG Architecture

The actor network in Deep Deterministic Policy Gradient (DDPG) utilizes deterministic policy gradients for training. DDPG is an off-policy algorithm that incorporates methods from Deep Q Networks (DQN), including updating target networks and using an experience replay buffer. Unlike DQN, DDPG is designed to handle continuous action spaces. In this example, we employ DDPG to train a network on the continuous Lunar Lander problem.

### DDPGAgent Class

The `DDPGAgent` class implements the Deep Deterministic Policy Gradient (DDPG) algorithm, which is a model-free, off-policy reinforcement learning method tailored specifically for environments that feature continuous action spaces. This class encapsulates all necessary functionalities required to train an agent using both policy-based and value-based strategies within such contexts.

#### Key Features

- **Actor-Critic Architecture**: The DDPG algorithm adopts an actor-critic structure. The actor is responsible for updating the policy distribution in the direction suggested by the critic, which evaluates the quality of actions taken by the actor. This methodology helps in refining the policy towards optimal actions.
- **Target Networks**: Stability during training is crucial, especially in complex environments. DDPG enhances stability by maintaining separate, slow-moving target networks for both the actor and the critic. These networks provide a fixed baseline for updating the main networks and are updated using a soft update mechanism.
- **Experience Replay**: To mitigate issues related to the correlation of consecutive observations and to improve sample efficiency, the `DDPGAgent` employs an experience replay mechanism. This strategy stores past transitions, which are later used for training the agent asynchronously from the environment's current state.
- **Batch Normalization**: Batch normalization is used within the network to handle varying scales of inputs effectively. It normalizes the inputs to each layer so that the network remains effective across a wide range of input values without needing significant changes to the hyperparameters.

The `DDPGAgent` class provides a structured approach to applying the DDPG algorithm in various simulation environments, facilitating both research and practical applications in fields such as robotics and automated systems.

### Overview of the Reinforcement Learning (RL) Training Process

The training loop in reinforcement learning is a structured sequence where we iteratively process episodes and their respective timesteps. This loop continues until a predefined stopping criterion is met, which might be reaching the maximum number of episodes, exceeding a limit on timesteps, or achieving a specific performance threshold by the agent.

Within each episode, the process unfolds timestep by timestep. At every step, the agent performs the following actions:

- **Action Selection**: The agent chooses an action based on the current state, guided by its policy.
- **Environment Interaction**: The chosen action is executed in the environment using the `env.step()` function, which returns the new state, reward, and a flag indicating whether the episode has ended.
- **Data Storage**: The agent stores the state transition (current state, action, reward, next state, and done flag) in the replay buffer. This buffer holds a collection of such transitions, allowing the agent to learn from a wider range of experiences.
- **Network Training**: Concurrently with data accumulation, the agent trains its neural network, which involves adjusting the network parameters to better predict the value of actions given the current state.
- **Target Network Updates**: Periodically, the agent updates its target network. This network helps stabilize the learning process by providing a consistent target for the training network's output. The updates are typically less frequent and use a weighted average of the training network's parameters to maintain stability.

Whenever an episode concludes, indicated by the `done` flag from the environment, the environment is reset to start a new episode, ensuring continuous and diverse learning opportunities.

### Images

1. **Initial State of the Environment**:
![image](https://github.com/MayCooper/PyTorch-Deep-Reinforcement-Learning-Framework-for-Autonomous-Moon-Landings/assets/82129870/55895edc-d123-4e2b-b8ea-5ce91a77110b)
   - "Initial state of the lunar lander environment."

2. **End of Episode State**:
![image](https://github.com/MayCooper/PyTorch-Deep-Reinforcement-Learning-Framework-for-Autonomous-Moon-Landings/assets/82129870/2bd4b7ac-8cc8-4e0b-939f-2c766943fbe5)
![image](https://github.com/MayCooper/PyTorch-Deep-Reinforcement-Learning-Framework-for-Autonomous-Moon-Landings/assets/82129870/1129d59e-be8d-45f6-9c97-4c8994b1f77c)
   - "State of the lunar lander environment at the end of an episode."

3. **Output for Episode Iterations**:
![image](https://github.com/MayCooper/PyTorch-Deep-Reinforcement-Learning-Framework-for-Autonomous-Moon-Landings/assets/82129870/0a77a80e-78dd-4867-88dd-3e8e6c0d8e7c)

### Plots

1. **Reward Trends Per Episode and Episode Duration Trends**:
![image](https://github.com/MayCooper/PyTorch-Deep-Reinforcement-Learning-Framework-for-Autonomous-Moon-Landings/assets/82129870/9d1ff093-8ac9-4ce6-bc79-40505926c556)
   - "Reward trends over episodes, showing both the sliding average and per episode rewards."
   - "Episode duration trends over time, showing both the sliding average and per episode durations."

2. **Distribution of Episode Rewards**:
![image](https://github.com/MayCooper/PyTorch-Deep-Reinforcement-Learning-Framework-for-Autonomous-Moon-Landings/assets/82129870/113be4a4-8b59-47c5-bf7d-31390b7ccdf4)
![image](https://github.com/MayCooper/PyTorch-Deep-Reinforcement-Learning-Framework-for-Autonomous-Moon-Landings/assets/82129870/5be8ae4b-5ad4-4c1d-84d8-373b93e4c1fe)
   - "Histogram showing the distribution of rewards across all episodes."

3. **Episode Length Over Time**:
![image](https://github.com/MayCooper/PyTorch-Deep-Reinforcement-Learning-Framework-for-Autonomous-Moon-Landings/assets/82129870/37988ae5-c86f-4677-aae7-f411ce73ee92)
   - "Episode lengths plotted over time to visualize changes in episode duration."

4. **Smoothed Episode Length Over Time**:
![image](https://github.com/MayCooper/PyTorch-Deep-Reinforcement-Learning-Framework-for-Autonomous-Moon-Landings/assets/82129870/18fe3ecf-a4a5-41b7-8b8a-dd1be0d51422)
   - "Smoothed episode lengths over time using a moving average to highlight trends."

5. **Correlation Between Episode Duration and Rewards**:
![image](https://github.com/MayCooper/PyTorch-Deep-Reinforcement-Learning-Framework-for-Autonomous-Moon-Landings/assets/82129870/7df9003a-5195-46ea-aad0-26bad8bdc378)
   - "Scatter plot showing the correlation between episode duration and rewards."

6. **Reward Distribution Across Episode Batches**:
![image](https://github.com/MayCooper/PyTorch-Deep-Reinforcement-Learning-Framework-for-Autonomous-Moon-Landings/assets/82129870/0ed66d8f-dcb0-4eed-b5fe-d19415465e14)
   - "Box plot showing the distribution of rewards across different episode batches."

7. **Seasonal Decomposition of Episode Rewards**:
![image](https://github.com/MayCooper/PyTorch-Deep-Reinforcement-Learning-Framework-for-Autonomous-Moon-Landings/assets/82129870/1957b979-2361-43eb-8970-bb388124372c)
   - "Decomposition of episode rewards into trend, seasonal, and residual components."

8. **Agent Path in Environment**:
![image](https://github.com/MayCooper/PyTorch-Deep-Reinforcement-Learning-Framework-for-Autonomous-Moon-Landings/assets/82129870/efe47ec2-0713-4d0e-b6b7-e9d6cbb87c20)
   - "Visualization of the agent's path within the environment."

9. **Average Agent Path in Environment**:
![image](https://github.com/MayCooper/PyTorch-Deep-Reinforcement-Learning-Framework-for-Autonomous-Moon-Landings/assets/82129870/938b3225-1db6-4b4e-af82-f5f43b948a36)
![image](https://github.com/MayCooper/PyTorch-Deep-Reinforcement-Learning-Framework-for-Autonomous-Moon-Landings/assets/82129870/e4a6a90e-ebfb-4ee7-9fb6-a27a164fa7c9)

   - "Average path taken by the agent across multiple episodes."
   - "Highlights the consistency and variance in the agent's navigation."
