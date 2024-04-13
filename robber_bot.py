import torch
import random
import numpy as np
from collections import deque
from model import QTrainer, Linear_QNet

MAX_MEMORY = 10000
LR = 0.001
BATCH_SIZE = 1000
NUM_GAMES = 100000


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0
        self.action_list = []
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(126, 80, 19)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def save_model(self, model_name='model.pth'):
        self.model.save(model_name)

    def load_model(self, model_name='model.pth'):
        self.model.load(model_name)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states)

    def train_short_memory(self, state, action, reward, next_state):
        self.trainer.train_step(state, action, reward, next_state)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 150 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 19)
        else:
            print("Using Model:")
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            robber_hexes = state[107:126]
            # change to find previous robber hex
            if move == state[-1]:
                prediction[move] = -1
                move = torch.argmax(prediction).item()
        return move


def train():
    from server import JSettlersServer

    # TODO: plot these as needed.
    placements = [0, 0, 0, 0]
    placement_history = deque(maxlen=10) #last 5 games
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    avg_placement = []

    agent = Agent()
    game = JSettlersServer("localhost", 2004, agent, timeout=120)
    while True:
        if agent.n_games >= NUM_GAMES:
            break
        feat_vector = game.get_message()
        if feat_vector is None:
            # print("Msg skipped: ")
            if game.final_place != -1:
                agent.n_games += 1
                reward = 0
                if game.final_place == 1:
                    reward = 10
                    placements[0] += 1
                    placement_history.append(1)
                elif game.final_place == 2:
                    reward = 5
                    placements[1] += 1
                    placement_history.append(2)
                elif game.final_place == 3:
                    reward = -3
                    placements[2] += 1
                    placement_history.append(3)
                elif game.final_place == 4:
                    reward = -6
                    placements[3] += 1
                    placement_history.append(4)
                else:
                    print("game not finished yet")

                for elem in agent.action_list:
                    agent.remember(elem[0], elem[1], elem[2] + reward, elem[3])

                print("Placed " + str(game.final_place))

                agent.action_list.clear()

                agent.train_long_memory()
                print("Finished training long term memory")
                agent.model.save()

                agent.n_games += 1

                placement_history.append(game.final_place)
                avg_placement.append(sum(placement_history) / len(placement_history))
                with open('placement_history.txt', 'a') as file:
                    file.write(str(game.final_place) + '\n')

                game.reset()

        else:
            cur_state = feat_vector

            action = agent.get_action(feat_vector)

            print("Action: " + str(action))
            reward = 0

            # state before action
            my_settlements = feat_vector[19:38]
            player2_settlements = feat_vector[38:57]
            player3_settlements = feat_vector[57:76]
            player4_settlements = feat_vector[76:95]
            other_players_resources = feat_vector[100:103]

            # short-term reward based on whether or not we stole a resource and whether or not we blocked ourself
            blocked_self = 1 if my_settlements[action] == 1 else 0
            stole_resource = 0
            
            if (player2_settlements[action] == 1 and other_players_resources[0] > 0) or (player3_settlements[action] == 1 and other_players_resources[1] > 0) \
            or (player4_settlements[action] == 1 and other_players_resources[2] > 0):
                stole_resource = 1
            
            reward += 0.5 * (stole_resource - 0.3 * blocked_self)
        
            # state after action
            if stole_resource == 0:
                new_state = np.array(feat_vector[0:107].tolist() + [0 for i in range(19)])
                new_state[107 + action] = 1
            else:
               new_state = np.array(feat_vector[0:107].tolist() + [0 for i in range(19)])
               new_state[107 + action] = 1
               new_state[100] = new_state[100] + 1


            agent.action_list.append((cur_state, action, reward, new_state))
            agent.train_short_memory(cur_state, action, reward, new_state)


if __name__ == '__main__':
    train()