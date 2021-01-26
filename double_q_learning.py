import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6
ACTIONS = ['left','right']
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
MAX_EPSISODES = 15
FRESH_TIME = 0.1
SHOW_TIME = 1

def build_q_table(n_states,actions):
    q_table = pd.DataFrame(
        np.zeros((n_states,len(actions))),
        columns = ACTIONS
    )
    return q_table


def choose_action(state,q_table,episode):
    all_actions = q_table.iloc[state,:]
    if(np.random.uniform()<EPSILON/(1+episode)) or ((all_actions==0).all()):
        action = np.random.choice(ACTIONS)
    else:
        action = all_actions.idxmax()

    return action

def get_env_feedback(S,A):
    if A == 'right':
        if S == N_STATES-2:
            S_ = N_STATES-1
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
        R = 0
    return S_,R


def update_env(S,episode,step_counter):
    env_list = ['-']*(N_STATES-1)+['T']
    if S == N_STATES-1:
        env_list[N_STATES-1] = 'o'
        print(''.join(env_list))
        print("Episode:{},total_steps:{}".format(episode+1,step_counter))
        time.sleep(SHOW_TIME)
    else:
        env_list[S] = 'o'
        print(''.join(env_list))
        time.sleep(FRESH_TIME)


def rl():
    q_table_A = build_q_table(N_STATES,ACTIONS)
    q_table_B = build_q_table(N_STATES,ACTIONS)
    q_table = build_q_table(N_STATES,ACTIONS)
    for episode in range(MAX_EPSISODES):
        print('episode:{} started'.format(episode+1))
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S,episode,step_counter)
        while not is_terminated:
            A = choose_action(S,q_table,episode)
            S_,R = get_env_feedback(S,A)
            update_a = np.random.rand()<0.5
            if update_a:
                A_star = q_table_A.iloc[S_].idxmax()
                q_table_A.loc[S,A] += ALPHA*(R+GAMMA*q_table_B.loc[S_,A_star]-q_table_A.loc[S,A])
            else:
                B_star = q_table_B.iloc[S_].idxmax()
                q_table_B.loc[S,A] += ALPHA*(R+GAMMA*q_table_A.loc[S_,B_star]-q_table_B.loc[S,A])
            if S_ == N_STATES-1:
                is_terminated = True
            # print('q_table_A:\n',q_table_A)
            # print('q_table_B:\n',q_table_B)
            q_table = (q_table_A + q_table_B)/2
            S = S_
            step_counter += 1
            update_env(S,episode,step_counter)
    return q_table

if __name__ == '__main__':
    q_table = rl()
    print("\r\nUltimate Q-table:\n")
    print(q_table)