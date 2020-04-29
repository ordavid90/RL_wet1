import numpy as np
from cartpole_cont import CartPoleContEnv
from lqr import find_lqr_control_input
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
# Part (3):
    env = CartPoleContEnv(initial_theta=np.pi * 0.1)
    actual_state = env.reset()
    env.render()
    xs, us, Ks = find_lqr_control_input(env)
    is_done = False
    iteration = 0
    is_stable_all = []
    s1 = []
    while not is_done:
        predicted_theta = xs[iteration].item(2)
        actual_theta = actual_state[2]
        s1.append(actual_theta)
        predicted_action = us[iteration].item(0)
        actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1)).item(0)
#        actual_action = us[iteration].item(0)
        actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
        actual_action = np.array([actual_action])
        actual_state, reward, is_done, _ = env.step(actual_action)
        is_stable = reward == 1.0
        is_stable_all.append(is_stable)
        env.render()
        iteration += 1
    env.close()
    s1.append(actual_state[2])

    env = CartPoleContEnv(initial_theta=np.pi * 0.05)
    actual_state = env.reset()
    env.render()
    xs, us, Ks = find_lqr_control_input(env)
    is_done = False
    iteration = 0
    is_stable_all = []
    s2 = []
    while not is_done:
        predicted_theta = xs[iteration].item(2)
        actual_theta = actual_state[2]
        s2.append(actual_theta)
        predicted_action = us[iteration].item(0)
        actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1)).item(0)
#        actual_action = us[iteration].item(0)
        actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
        actual_action = np.array([actual_action])
        actual_state, reward, is_done, _ = env.step(actual_action)
        is_stable = reward == 1.0
        is_stable_all.append(is_stable)
        env.render()
        iteration += 1
    env.close()
    s2.append(actual_state[2])

    env = CartPoleContEnv(initial_theta=np.pi * 0.11)
    actual_state = env.reset()
    env.render()
    xs, us, Ks = find_lqr_control_input(env)
    is_done = False
    iteration = 0
    is_stable_all = []
    s3 = []
    while not is_done:
        predicted_theta = xs[iteration].item(2)
        actual_theta = actual_state[2]
        s3.append(actual_theta)
        predicted_action = us[iteration].item(0)
        actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1)).item(0)
#        actual_action = us[iteration].item(0)
        actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
        actual_action = np.array([actual_action])
        actual_state, reward, is_done, _ = env.step(actual_action)
        is_stable = reward == 1.0
        is_stable_all.append(is_stable)
        env.render()
        iteration += 1
    env.close()
    s3.append(actual_state[2])

# Data for plotting
t = np.linspace(0, env.tau * env.planning_steps, env.planning_steps + 1)

fig, ax = plt.subplots()

ax.plot(t, s1, label='theta = 0.1pi')
ax.plot(t, s2, label='theta = 0.05pi')
ax.plot(t, s3, label='theta = 0.11pi')
plt.legend()
ax.set(xlabel='time (s)', ylabel='angle (rad)',
       title='Angle / Time')
ax.grid()

fig.savefig("test.png")
plt.show()