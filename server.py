import socket
import os
import numpy as np
from collections import deque
import time

OBSERVATION_SPACE_SIZE = (1,22)
ACTION_SPACE_SIZE = 2
#  Stats settings
AGGREGATE_STATS_EVERY = 2  # episodes


class JSettlersServer:
    def __init__(self, host, port, agent, timeout=None):
        # Used for training agent
        self.agent = agent
        self.host = host
        self.port = port
        self.timeout = timeout
        self.prev_vector = None
        self.last_action = None
        self.cur_state = None
        self.conn = None
        self.addr = None

        self.final_place = -1 # change from -1 to final placing after game finishes

        # Used for logging models and stats
        self.ep_rewards = [0]
        self.curr_episode = 1
        self.standing_log = "agent_standings.csv"
        self.standing_results = [0,0,0,0]

        self.soc = socket.socket()  # Create a socket object
        if self.timeout:
            self.soc.settimeout(self.timeout)
        try:
            print(str(self.host) + " " + str(self.port))
            self.soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.soc.bind((self.host, self.port))

        except socket.error as err:
            print('Bind failed. Error Code : '.format(err))

        self.soc.listen(10)
        print("Socket Listening ... ")


    def reset(self):
        self.prev_vector = None
        self.last_action = None
        self.cur_state = None
        self.conn = None
        self.addr = None

        self.final_place = -1 # change from -1 to final placing after game finishes

        # Used for logging models and stats
        self.ep_rewards = [0]
        self.curr_episode += 1



    def run(self):
        while True:
            try:
                feat_vector = self.get_message()
                if feat_vector is None:
                    continue
                action = self.agent.get_action(feat_vector)
                self.conn.send((str(action) + '\n').encode(encoding='UTF-8'))
                # print("Result: " + str(action) + "\n")
            except socket.timeout:
                print("Timeout or error occured. Exiting ... ")
                break

    def get_message(self):
        try:
            self.conn, self.addr = self.soc.accept()     # Establish connection with client
            length_of_message = int.from_bytes(self.conn.recv(2), byteorder='big')
            msg = self.conn.recv(length_of_message).decode("UTF-8")
            print(msg)
            msg_args = msg.split("|")
            if msg_args[0] == 'end':
                print("Ended")
                is_over = str(msg_args[1])
                final_placing = int(msg_args[2])
                self.final_place = final_placing
                print(self.final_place)
                return None

            elif msg_args[0] == 'robber':
                dice_values = [int(x) for x in msg_args[1].split(",")]
                my_settlements = [int(x) for x in msg_args[2].split(",")]
                p2_settlements = [int(x) for x in msg_args[3].split(",")]
                p3_settlements = [int(x) for x in msg_args[4].split(",")]
                p4_settlements = [int(x) for x in msg_args[5].split(",")]
                my_dev_cards = int(msg_args[6])
                others_dev_cards = [int(x) for x in msg_args[7].split(",")]
                my_res = int(msg_args[8])
                others_res = [int(x) for x in msg_args[9].split(",")]
                my_vp = int(msg_args[10])
                others_vp = [int(x) for x in msg_args[11].split(",")]
                robber_hexes = [int(x) for x in msg_args[12].split(",")]

                feat_vector = np.array(dice_values + my_settlements + p2_settlements + p3_settlements + p4_settlements + [my_dev_cards] + others_dev_cards + [my_res] + others_res + [my_vp] + others_vp + robber_hexes)
                return feat_vector
                
            return None

        except:
            print("Timeout or error occured. Exiting ... ")

    def play_step(self, msg_type, action):
        if msg_type == 'robber':
            # print("Sending Msg: " + str(action))
            self.conn.send((str(action) + '\n').encode(encoding='UTF-8'))
            reward = 0
            return reward
        else:
            return 0


if __name__ == "__main__":
    from robber_bot import Agent
    trading_agent = Agent()
    server = JSettlersServer("localhost", 2004, trading_agent, timeout=120)
    server.run()