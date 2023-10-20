import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import random
from collections import deque
import itertools

from env_cfg import Config, TestDemand, Agent


def get_init_len(init):
    """
    Calculate total number of elements in a 1D array or list of lists.
    :type init: iterable, list or (list of lists)
    :rtype: int
    """
    is_init_array = all([isinstance(x, (float, int, np.int64)) for x in init])
    if is_init_array:
        init_len = len(init)
    else:
        init_len = len(list(itertools.chain.from_iterable(init)))
    return init_len



class BeerGame(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, n_agents=4, n_turns_per_game=100,test_mode=False):
        super().__init__()
        c = Config()
        config, unparsed = c.get_config()
        self.config = config
        self.test_mode = test_mode
        if self.test_mode:
            self.test_demand_pool = TestDemand()

        self.curGame = 1 # The number associated with the current game (counter of the game)
        self.curTime = 0
        self.m = 10             #window size
        self.totIterPlayed = 0  # total iterations of the game, played so far in this and previous games
        self.players = self.createAgent()  # create the agents
        self.T = 0
        self.demand = []
        self.orders = []
        self.shipments = []
        self.rewards = []
        self.cur_demand = 0

        self.ifOptimalSolExist = self.config.ifOptimalSolExist
        self.getOptimalSol()

        self.totRew = 0    # it is reward of all players obtained for the current player.
        self.totalReward = 0
        self.n_agents = n_agents

        self.n_turns = n_turns_per_game
        seed  = random.randint(0,1000000)
        self.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.totalTotal = 0

        # Agent 0 has 5 (-2, ..., 2) + AO
        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(5),gym.spaces.Discrete(5),gym.spaces.Discrete(5),gym.spaces.Discrete(5)]))

        ob_spaces = {}
        for i in range(self.m):
            ob_spaces[f'current_stock_minus{i}'] = spaces.Discrete(5)
            ob_spaces[f'current_stock_plus{i}'] = spaces.Discrete(5)
            ob_spaces[f'OO{i}'] = spaces.Discrete(5)
            ob_spaces[f'AS{i}'] = spaces.Discrete(5)
            ob_spaces[f'AO{i}'] = spaces.Discrete(5)

        # Define the observation space, x holds the size of each part of the state
        x = [750, 750, 170, 45, 45]
        oob = []
        for _ in range(self.m):
          for ii in range(len(x)):
            oob.append(x[ii])
        self.observation_space = gym.spaces.Tuple(tuple([spaces.MultiDiscrete(oob)] * 4))

        print("Observation space:")
        print(self.observation_space)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def createAgent(self):
      agentTypes = self.config.agentTypes
      return [Agent(i,self.config.ILInit[i], self.config.AOInit, self.config.ASInit[i],
                                self.config.c_h[i], self.config.c_p[i], self.config.eta[i],
                                agentTypes[i],self.config) for i in range(self.config.NoAgent)]


    def resetGame(self, demand, ):
        self.demand = demand
        self.playType='test'
        self.curTime = 0
        self.curGame += 1
        self.totIterPlayed += self.T
        self.T = self.planHorizon()         #now fixed
        self.totalReward = 0

        self.deques = []
        for i in range(self.n_agents):
            deques = {}
            deques[f'current_stock_minus'] = deque([0.0] * self.m, maxlen=self.m)
            deques[f'current_stock_plus'] = deque([0.0] * self.m, maxlen=self.m)
            deques[f'OO'] = deque([0] * self.m, maxlen=self.m)
            deques[f'AS'] = deque([0] * self.m, maxlen=self.m)
            deques[f'AO'] = deque([0] * self.m, maxlen=self.m)
            self.deques.append(deques)

        # reset the required information of player for each episode
        for k in range(0,self.config.NoAgent):
            self.players[k].resetPlayer(self.T)

        # update OO when there are initial IL,AO,AS
        self.update_OO()


    def reset(self):
        if self.test_mode:
            demand = self.test_demand_pool.next()
            if not self.test_demand_pool:           #if run out of testing data
                self.test_demand_pool = TestDemand()
        else:
            demand = [random.randint(0,2) for _ in range(102)]

        self.resetGame(demand)
        observations = [None] * self.n_agents

        self.deques = []
        for i in range(self.n_agents):
            deques = {}
            deques[f'current_stock_minus'] = deque([0.0] * self.m, maxlen=self.m)
            deques[f'current_stock_plus'] = deque([0.0] * self.m, maxlen=self.m)
            deques[f'OO'] = deque([0] * self.m, maxlen=self.m)
            deques[f'AS'] = deque([0] * self.m, maxlen=self.m)
            deques[f'AO'] = deque([0] * self.m, maxlen=self.m)
            self.deques.append(deques)

        # prepend current observation
        # get current observation, prepend to deque
        for i in range(self.n_agents):
            curState = self.players[i].getCurState(self.curTime)
            self.deques[i]['current_stock_minus'].appendleft(int(curState[0]))
            self.deques[i]['current_stock_plus'].appendleft(int(curState[1]))
            self.deques[i]['OO'].appendleft(int(curState[2]))
            self.deques[i]['AS'].appendleft(int(curState[3]))
            self.deques[i]['AO'].appendleft(int(curState[4]))

        # return entire m observations
        obs = [[], [], [], []]
        for i in range(self.n_agents):
            spaces = {}
            for j in range(self.m):
                obs[i].append(self.deques[i]['current_stock_minus'][j])
                obs[i].append(self.deques[i]['current_stock_plus'][j])
                obs[i].append(self.deques[i]['OO'][j])
                obs[i].append(self.deques[i]['AS'][j])
                obs[i].append(self.deques[i]['AO'][j])
                # spaces[f'current_stock_minus{j}'] = self.deques[i]['current_stock_minus'][j]
                # spaces[f'current_stock_plus{j}'] = self.deques[i]['current_stock_plus'][j]
                # spaces[f'OO{j}'] = self.deques[i]['OO'][j]
                # spaces[f'AS{j}'] = self.deques[i]['AS'][j]
                # spaces[f'AO{j}'] = self.deques[i]['AO'][j]

            # observations[i] = spaces

        obs_array = np.array([np.array(row) for row in obs])
        return obs_array  # observations #self._get_observations()


    def step(self, action:list):
        if get_init_len(action) != self.n_agents:
            raise error.InvalidAction(f'Length of action array must be same as n_agents({self.n_agents})')
        if any(np.array(action) < 0):
            raise error.InvalidAction(f"You can't order negative amount. You agents actions are: {action}")

        self.handleAction(action)
        self.next()

        self.orders = action

        for i in range(self.n_agents):
            self.players[i].getReward()
        self.rewards = [1 * self.players[i].curReward for i in range(0, self.config.NoAgent)]

        if self.curTime == self.T+1:
            self.done = [True] * 4
        else:
            self.done = [False] * 4


        # get current observation, prepend to deque
        for i in range(self.n_agents):
            curState = self.players[i].getCurState(self.curTime)
            self.deques[i]['current_stock_minus'].appendleft(int(curState[0]))
            self.deques[i]['current_stock_plus'].appendleft(int(curState[1]))
            self.deques[i]['OO'].appendleft(int(curState[2]))
            self.deques[i]['AS'].appendleft(int(curState[3]))
            self.deques[i]['AO'].appendleft(int(curState[4]))

        # return entire m observations
        obs = [[],[],[],[]]
        observations = [None] * self.n_agents
        for i in range(self.n_agents):
            spaces = {}
            for j in range(self.m):
              obs[i].append(self.deques[i]['current_stock_minus'][j])
              obs[i].append(self.deques[i]['current_stock_plus'][j])
              obs[i].append(self.deques[i]['OO'][j])
              obs[i].append(self.deques[i]['AS'][j])
              obs[i].append(self.deques[i]['AO'][j])

        obs_array = np.array([np.array(row) for row in obs])
        state = obs_array #observations #self._get_observations()
        return state, self.rewards, self.done, {}



    def handleAction(self, action):
        # get random lead time
        leadTime = random.randint(self.config.leadRecOrderLow[0], self.config.leadRecOrderUp[0])
        self.cur_demand = self.demand[self.curTime]
        # set AO
        BS = False
        self.players[0].AO[self.curTime] += self.demand[self.curTime]       #orders from customer, add directly to the retailer arriving order
        for k in range(0, self.config.NoAgent):
            if k >= 0:  #recording action
                self.players[k].action = np.zeros(5)        #one-hot transformation
                self.players[k].action[action[k]] = 1
                BS = False
            else:
                raise NotImplementedError
                self.getAction(k)
                BS = True

            # updates OO and AO at time t+1
            self.players[k].OO += self.players[k].actionValue(self.curTime, self.playType, BS = BS)     #open order level update
            leadTime = random.randint(self.config.leadRecOrderLow[k], self.config.leadRecOrderUp[k])        #order
            if self.players[k].agentNum < self.config.NoAgent-1:
                if k>=0:
                    self.players[k + 1].AO[self.curTime + leadTime] += self.players[k].actionValue(self.curTime,
                                                                                                   self.playType,
                                                                                                   BS=False)  # TODO(yan): k+1 arrived order contains my own order and the order i received from k-1
                else:
                    raise NotImplementedError
                    self.players[k + 1].AO[self.curTime + leadTime] += self.players[k].actionValue(self.curTime,
                                                                                                   self.playType,
                                                                                                   BS=True)  # open order level update

    def next(self):
        # get a random leadtime for shipment
        leadTimeIn = random.randint(self.config.leadRecItemLow[self.config.NoAgent - 1],
                                    self.config.leadRecItemUp[self.config.NoAgent - 1])

        # handle the most upstream recieved shipment
        self.players[self.config.NoAgent-1].AS[self.curTime + leadTimeIn] += self.players[self.config.NoAgent-1].actionValue(self.curTime, self.playType, BS=True)
                                                                #the manufacture gets its ordered beer after leadtime

        self.shipments = []
        for k in range(self.config.NoAgent-1,-1,-1): # [3,2,1,0]

            # get current IL and Backorder
            current_IL = max(0, self.players[k].IL)
            current_backorder = max(0, -self.players[k].IL)

            # increase IL and decrease OO based on the action, for the next period
            self.players[k].recieveItems(self.curTime)

            # observe the reward
            possible_shipment = min(current_IL + self.players[k].AS[self.curTime],
                                    current_backorder + self.players[k].AO[self.curTime])       #if positive IL, ship all beer or all they needs, if backorders, ship all k-1 needs
            self.shipments.append(possible_shipment)

            # plan arrivals of the items to the downstream agent
            if self.players[k].agentNum > 0:
                leadTimeIn = random.randint(self.config.leadRecItemLow[k-1], self.config.leadRecItemUp[k-1])
                self.players[k-1].AS[self.curTime + leadTimeIn] += possible_shipment

            # update IL
            self.players[k].IL -= self.players[k].AO[self.curTime]

            # observe the reward
            self.players[k].getReward()
            rewards = [-1 * self.players[i].curReward for i in range(0, self.config.NoAgent)]

            # update next observation
            self.players[k].nextObservation = self.players[k].getCurState(self.curTime + 1)

        if self.config.ifUseTotalReward:  # default is false
            # correction on cost at time T
            if self.curTime == self.T:
                self.getTotRew()

        self.curTime += 1

    def getAction(self, k):
        self.players[k].action = np.zeros(self.config.actionListLenOpt)

        if self.config.demandDistribution == 2:
            if self.curTime and self.config.use_initial_BS <= 4:
                self.players[k].action[np.argmin(np.abs(np.array(self.config.actionListOpt) - \
                                                        max(0, (self.players[k].int_bslBaseStock - (
                                                                    self.players[k].IL + self.players[k].OO -
                                                                    self.players[k].AO[self.curTime])))))] = 1
            else:
                self.players[k].action[np.argmin(np.abs(np.array(self.config.actionListOpt) - \
                                                        max(0, (self.players[k].bsBaseStock - (
                                                                    self.players[k].IL + self.players[k].OO -
                                                                    self.players[k].AO[self.curTime])))))] = 1
        else:
            self.players[k].action[np.argmin(np.abs(np.array(self.config.actionListOpt) - \
                                                    max(0, (self.players[k].bsBaseStock - (
                                                                self.players[k].IL + self.players[k].OO -
                                                                self.players[k].AO[self.curTime])))))] = 1

    def getTotRew(self):
      totRew = 0
      for i in range(self.config.NoAgent):
        # sum all rewards for the agents and make correction
        totRew += self.players[i].cumReward

      for i in range(self.config.NoAgent):
        self.players[i].curReward += self.players[i].eta*(totRew - self.players[i].cumReward) #/(self.T)


    def planHorizon(self):
      # TLow: minimum number for the planning horizon # TUp: maximum number for the planning horizon
      #output: The planning horizon which is chosen randomly.
      return random.randint(self.n_turns, self.n_turns)# self.config.TLow,self.config.TUp)

    def update_OO(self):
        for k in range(0,self.config.NoAgent):
            if k < self.config.NoAgent - 1:
                self.players[k].OO = sum(self.players[k+1].AO) + sum(self.players[k].AS)
            else:
                self.players[k].OO = sum(self.players[k].AS)

    def getOptimalSol(self):
        # if self.config.NoAgent !=1:
        if self.config.NoAgent != 1 and 1 == 2:
            # check the Shang and Song (2003) condition.
            for k in range(self.config.NoAgent - 1):
                if not (self.players[k].c_h == self.players[k + 1].c_h and self.players[k + 1].c_p == 0):
                    self.ifOptimalSolExist = False

            # if the Shang and Song (2003) condition satisfied, it runs the algorithm
            if self.ifOptimalSolExist == True:
                calculations = np.zeros((7, self.config.NoAgent))
                for k in range(self.config.NoAgent):
                    # DL_high
                    calculations[0][k] = ((self.config.leadRecItemLow + self.config.leadRecItemUp + 2) / 2 \
                                          + (self.config.leadRecOrderLow + self.config.leadRecOrderUp + 2) / 2) * \
                                         (self.config.demandUp - self.config.demandLow - 1)
                    if k > 0:
                        calculations[0][k] += calculations[0][k - 1]
                    # probability_high
                    nominator_ch = 0
                    low_denominator_ch = 0
                    for j in range(k, self.config.NoAgent):
                        if j < self.config.NoAgent - 1:
                            nominator_ch += self.players[j + 1].c_h
                        low_denominator_ch += self.players[j].c_h
                    if k == 0:
                        high_denominator_ch = low_denominator_ch
                    calculations[2][k] = (self.players[0].c_p + nominator_ch) / (
                                self.players[0].c_p + low_denominator_ch + 0.0)
                    # probability_low
                    calculations[3][k] = (self.players[0].c_p + nominator_ch) / (
                                self.players[0].c_p + high_denominator_ch + 0.0)
                # S_high
                calculations[4] = np.round(np.multiply(calculations[0], calculations[2]))
                # S_low
                calculations[5] = np.round(np.multiply(calculations[0], calculations[3]))
                # S_avg
                calculations[6] = np.round(np.mean(calculations[4:6], axis=0))
                # S', set the base stock values into each agent.
                for k in range(self.config.NoAgent):
                    if k == 0:
                        self.players[k].bsBaseStock = calculations[6][k]

                    else:
                        self.players[k].bsBaseStock = calculations[6][k] - calculations[6][k - 1]
                        if self.players[k].bsBaseStock < 0:
                            self.players[k].bsBaseStock = 0
        elif self.config.NoAgent == 1:
            if self.config.demandDistribution == 0:
                self.players[0].bsBaseStock = np.ceil(
                    self.config.c_h[0] / (self.config.c_h[0] + self.config.c_p[0] + 0.0)) * ((
                                                                                                         self.config.demandUp - self.config.demandLow - 1) / 2) * self.config.leadRecItemUp
        elif 1 == 1:
            f = self.config.f
            f_init = self.config.f_init
            for k in range(self.config.NoAgent):
                self.players[k].bsBaseStock = f[k]
                self.players[k].int_bslBaseStock = f_init[k]

    def render(self, mode='human'):
        # if mode != 'human':
        #     raise NotImplementedError(f'Render mode {mode} is not implemented yet')
        # print("")
        print('\n' + '=' * 20)
        print('Turn:     ', self.curTime)
        stocks = [p.IL for p in self.players]
        print('Stocks:   ', ", ".join([str(x) for x in stocks]))
        print('Orders:   ', self.orders)
        print('Shipments:', self.shipments)
        print('Rewards:', self.rewards)
        print('Customer demand: ', self.cur_demand)

        AO = [p.AO[self.curTime] for p in self.players]
        AS = [p.AS[self.curTime] for p in self.players]

        print('Arrived Order: ', AO)
        print('Arrived Shipment: ', AS)

        OO = [p.OO for p in self.players]
        print('Working Order: ', OO)


        # print('Last incoming orders:  ', self.next_incoming_orders)
        # print('Cum holding cost:  ', self.cum_stockout_cost)
        # print('Cum stockout cost: ', self.cum_holding_cost)
        # print('Last holding cost: ', self.holding_cost)
        # print('Last stockout cost:', self.stockout_cost)



if __name__ == "__main__":
    env = BeerGame()
    obs = env.reset()
    env.render()
    done = False

    while not done:
        rnd_action = env.action_space.sample()
        next_obs, reward, done_list, _ = env.step(rnd_action)
        done = all(done_list)
        env.render()


    print(1)