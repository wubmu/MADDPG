import numpy as np
import torch


class ReplayBuffer(object):
    """
    用于并行部署的多智能体（MARL）的重放缓冲区
    """

    def __init__(self, args,
                 max_steps,
                 num_agents,
                 obs_dims,
                 ac_dims):
        """

        Args:
            args:
            max_steps (int): 最大容量
            num_agents (int):环境中智能体的个数
            obs_dims (list of ints):单个智能体的观测状态的维度
            ac_dims (list of ints):每个智能体的动作维度
        """
        self.max_steps = max_steps
        self.num_agents = num_agents

        self.obs_buffers = []
        self.action_buffers = []
        self.reward_buffers = []
        self.next_obs_buffers = []
        self.done_buffers = []

        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffers.append(np.zeros((max_steps, odim)))
            self.action_buffers.append(np.zeros(max_steps, ac_dims))
            self.reward_buffers.append(np.zeros(max_steps))
            self.next_obs_buffers.append(np.zeros((max_steps, odim)))
            self.done_buffers.append(np.zeros(max_steps))

        self.current_index = 0  # 当前索引写入（覆盖最旧的数据）
        self.filled_index = 0

    def __len__(self):
        return self.filled_index

    def push(self, observations, actions, rewards, next_observations, dones):
        """
        Args:
            observations: (batch_size,num_agents, obs_dims)
            actions: (batch_size, num_agents, action_dims)
            rewards: (batch_size ,num_agents)
            next_observations:
            dones:

        Returns:

        """

        counts = observations.shape[0]  # 处理多个并行环境

        # 剩余容量不够存储 counts个条目的数据
        if self.current_index + counts > self.max_steps:
            rollover = self.max_steps - self.current_index # 向左滚动 [1,2,3,4] -->roll 2-->[3,4,1 ,2]
            for agent_index in range(self.num_agents):
                self.obs_buffers[agent_index] = np.roll(self.obs_buffers[agent_index],
                                                        rollover, axis=0)
                self.action_buffers[agent_index] = np.roll(self.action_buffers[agent_index],
                                                           rollover, axis=0)
                self.reward_buffers[agent_index] = np.roll(self.reward_buffers[agent_index],
                                                           rollover)
                self.next_obs_buffers[agent_index] = np.roll(self.next_obs_buffers[agent_index],
                                                             rollover, axis=0)
                self.done_buffers[agent_index] = np.roll(self.done_buffers[agent_index],
                                                         rollover)
                # 在末尾补满之后，current_index回到开头
                self.current_index = 0
                self.filled_index = self.max_steps

        # 正常处理
        for agent_index in range(self.num_agents):
            self.obs_buffers[agent_index][self.current_index: self.current_index + counts] = np.vstack(
                observations[:, agent_index]
            )
            # todo: 理解下,acion_buffers的处理
            # self.action_buffers[agent_index][self.current_index: self.current_index + counts] = np.vstack(
            #     actions[:,agent_index]
            # )
            self.action_buffers[agent_index][self.current_index: self.current_index + counts] = actions[agent_index]
            self.reward_buffers[agent_index][self.current_index: self.current_index + counts] = rewards[:,agent_index]
            self.next_obs_buffers[agent_index][self.current_index: self.current_index + counts] = np.vstack(
                next_observations[:,agent_index]
            )
            self.done_buffers[agent_index][self.current_index:self.current_index+ counts] = dones[:agent_index]


        self.current_index += counts
        if self.filled_index  < self.max_steps:
            self.filled_index += counts
        if self.current_index == self.max_steps:
            self.current_index = 0


    def sample(self, sample_size, to_gpu = False):
        indexs = np.random.choice(np.arange(self.filled_index), size=sample_size
                                  ,replace=True) # repalce 可以重复采样

        