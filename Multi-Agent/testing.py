import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import os
from time import perf_counter
import cv2

# To load data
loaded_data = np.load(os.path.join(os.getcwd(), "Results", "sim_data.npz"), allow_pickle=True)
data_stack_np = loaded_data["data_stack"]

# data_stack_np[timestep, column (agents, dialect_groups)]



# data_stack_np[step, column(agent object, dialect clusters)][agent_idx]
agent = data_stack_np[0, 0][0]


agent_id = agent.id
y = []; x = []
for step in data_stack_np:
    agent = [agent for agent in step[0] if agent.id == agent_id]
    if len(agent) > 0: 
        agent = agent[0]
        x.append(agent.position[0])
        y.append(agent.position[1])
    else: break

li = list(zip(x, y))
img = cv2.imread(os.path.join(os.getcwd(), "environment.png"))
img = cv2.resize(img, (1280, 720), interpolation = cv2.INTER_AREA)
plt.imshow(img, origin='upper')

plt.plot(*zip(*li), color='tab:red')
plt.plot(*zip(*li), color='tab:green')
plt.show()



for step in data_stack_np:
    dialects = step[1]
    print(dialects)