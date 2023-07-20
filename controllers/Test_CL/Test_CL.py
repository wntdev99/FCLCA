STATE_SIZE = 24
MAX_SPEED = 1.57
MAX_EPISODE = 30
MAX_FRAME = 3
MAX_LENGHT = 0.9
MIN_DISTANCE = 0.35
INPUT_ONE_FRAME = 8
NORMALIZATION_SIZE = 100
ARRIVE_STANDARD = 0.1
REPLAY_CYCLE = 2000
TARGET_NETWORK_CYCLE = 5
GOAL_X = 0
GOAL_Y = 0 
OBSTACLE_COUNT = 0
MODIFY_NUM = 7
MODEL_NAME = "Curriculum 1"

import os
import math
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dqn_agent import DqnAgent
from replay_buffer import ReplayBuffer
from controller import Supervisor

DAY_NUM = str(datetime.date.today())

# 1. 초기 세팅
robot = Supervisor()
agent = DqnAgent()
buffer = ReplayBuffer()
# 1-1. 현재 world timestep
timestep = int(robot.getBasicTimeStep())
# 1-1-1. e-puck 모터 정보
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
# 1-1-2. 로봇 모터 다음 명령 있을 때 까지 전 모터 상태 유지
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
# 1-1-3. 로봇 속도 0
left_motor.setVelocity(0)
right_motor.setVelocity(0)
# 1-2. 로봇 노드 정보
ep_node = robot.getFromDef('ep')
# 1-3. 로봇 위치 필드 정보
translation_field = ep_node.getField('translation')
rotation_field = ep_node.getField('rotation')
# 1-3-1. 장애물 필드 정보
ob_field = []    
for i in range(OBSTACLE_COUNT):
    tmp = robot.getFromDef(f'ob{i}').getField('translation').value
    ob_field.append(tmp[0:2])
ob_field.append([0,0])
#1-4. e-puck proximity sensor 정보
ps = []
psNames = [
    'ps0', 'ps1', 'ps2', 'ps3',
    'ps4', 'ps5', 'ps6', 'ps7'
]
for i in range(8):
    ps.append(robot.getDevice(psNames[i]))
    ps[i].enable(timestep)
# 1-5. 초기화
state = np.zeros((STATE_SIZE))
next_state = np.zeros((STATE_SIZE))
ep = []
storage = []                                                            # storage 는 state를 받기 위한 임시 저장소
loss_data = []
reward_data = []
done_storage = []                                                       # done,collision 은 도착, 충돌의 비율을 시각화 하기 위함 
collision_storage = []
action = 0
avg_reward = 0                                                          # avg_reward 는 평균 보상 그래프 출력을 위함
count_state = 0              # one state 를 n개의 frame으로 만들기 위함
count_experience = 0      
set_count = 0
x_min, x_max = -MAX_LENGHT, MAX_LENGHT
y_min, y_max = -MAX_LENGHT, MAX_LENGHT
# 1-5-1. 최대 에피소드
max_episodes = MAX_EPISODE
# 1-5-2. goal 초기화
goal = [GOAL_X,GOAL_Y,0]
# 1-5-3. input 초기화
input = INPUT_ONE_FRAME

# 2. 함수    
# 2-1. one frame get
def environment():
    # 2-1-1. location get
    ep = translation_field.value
    # 2-1-2. heading get
    orientation = ep_node.getOrientation()
    heading = rotated_point(orientation)
    # 2-1-3. perfect_angle get    
    perfect_angle = point_slope(goal)                           
    # 2-1-4. theta get
    #theta = abs(perfect_angle - heading)
    theta = heading - perfect_angle
    if abs(theta) > 180:
        if theta < 0:
            theta = theta + 360
        elif theta > 0:
            theta = theta - 360
    # 2-1-5. radius get ep 
    goal_radius = math.sqrt(pow(goal[0] - ep[0],2) + pow(goal[1] - ep[1],2))
    # 2-1-6. radius get ob
    storage.append(goal_radius)
    storage.append(math.radians(theta))
    # 2-1-6-1. sensor value
    storage.append((ps[6].value)/100)
    storage.append((ps[7].value)/100)
    storage.append((ps[0].value)/100)
    storage.append((ps[1].value)/100)
    storage.append((ps[5].value)/100)
    storage.append((ps[2].value)/100)
    
# 2-2. Collect experiences
def collect_experiences(state,next_state,action,reward,done,buffer):
    buffer.store_experience(state,next_state,reward,action,done)
   
# 2-3. Select action 
def Action(action):
    if action == 0:
        left_motor.setVelocity(MAX_SPEED)
        right_motor.setVelocity(MAX_SPEED)
    elif action == 1:
        left_motor.setVelocity(MAX_SPEED)
        right_motor.setVelocity(-MAX_SPEED)
    elif action == 2:
        left_motor.setVelocity(-MAX_SPEED)
        right_motor.setVelocity(MAX_SPEED)

# 2-4. Reward structure
def Reward(state,next_state):
    total = 0
    for i in range(MAX_FRAME):                                          # 각 프레임에 대해 모두 진행
        if next_state[i * input] < ARRIVE_STANDARD:
            total += 10
        if state[i * input + 1] < 0 and action == 2:
            total += 0.02
        if state[i * input + 1] < 0 and action == 1:
            total -= 0.1
        if state[i * input + 1] > 0 and action == 1:
            total += 0.02
        if state[i * input + 1] > 0 and action == 2:
            total -= 0.1
        if state[i * input] + 0.0002 < next_state[i * input] and action == 0:
            total -= 0.15
        if next_state[i * input] + 0.0002< state[i * input] and action == 0:
            total += 0.005
        if next_state[i * input] + 0.0004 < state[i * input] and action == 0:
            total += 0.005
        if next_state[i * input] + 0.0005 < state[i * input] and action == 0:
            total += 0.005
        if next_state[i * input] + 0.0006 < state[i * input] and action == 0:
            total += 0.005
        if next_state[i * input] + 0.0007 < state[i * input] and action == 0:
            total += 0.005
        total -= 0.0001
    return total
    
# 2.5. Done check
def Done():
    for i in range(MAX_FRAME):
        # location get
        if next_state[i * input] <= ARRIVE_STANDARD:
            done = True     # 그만
        else:
            done = False    # 더해    
            
        return done
    
# 2.6. Collision check
def collision_check():
    global set_count 
    for j in range(MAX_FRAME):
        for i in range(4):
            if 5 < next_state[j * input + 2 + i] :
                setting()
                set_count = 1
                collision_storage.append(1)
                break
            if i == 3:
                collision_storage.append(0)

# 2.7. Setting
def setting(): 
    cc = 0
    while True:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        for i, j in ob_field:
            street = math.sqrt(pow(x - i,2) + pow(y - j,2))
            if street < MIN_DISTANCE:
                cc = cc + 1
                break
        if cc == 1:
            cc = 0
            continue
        else:
            r1 = np.random.uniform(-3,3)                                                                            
            translation_field.setSFVec3f([translation_field.value[0],translation_field.value[1],5])
            translation_field.setSFVec3f([x,y,5])
            translation_field.setSFVec3f([x,y,0])
            rotation_field.setSFRotation([0,0,-1,r1])
            robot.simulationResetPhysics()
            break
    return


# 2.8. e-puck  perfect_angle
"""
1. 목적지를 바라보는 방향을 기준으로 어느정도 각도로 보고 있냐는 정도
2. 사분면에 따라 달라짐
"""
def point_slope(target):
    ep = translation_field.value
    slope = (target[1] - ep[1]) / (target[0] - ep[0])
    if slope < 0:
        result = np.degrees(math.atan(slope))+180
        if target[0] < ep[0] and target[1] > ep[1]:
            return result
        elif target[0] > ep[0] and target[1] < ep[1]:
            return result + 180
        elif target[0] > ep[0] and target[1] > ep[1]:
            return result
        elif target[0] < ep[0] and target[1] < ep[1]:
            return result + 180
        else:
            return result

    else:
        result = np.degrees(math.atan(slope))
        if target[0] < ep[0] and target[1] > ep[1]:
            return result
        elif target[0] > ep[0] and target[1] < ep[1]:
            return (result + 180)
        elif target[0] > ep[0] and target[1]> ep[1]:
            return result 
        elif target[0] < ep[0] and target[1] < ep[1]:
            return (result + 180)
        else:
            return result
        
# 2.9. coordinate transformation
def rotated_point(orientation):
    angle = np.radians(-90)
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]])
    point = np.array(orientation[:3])
    rotated_point = np.dot(rot_matrix,point)
    heading = math.atan2(rotated_point[0], rotated_point[1])
    heading_degrees = np.degrees(heading) + 180
    return heading_degrees

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")
        
        
# 3. Train
for episode_cnt in range(1,max_episodes):
    # 3-1. one experience
    while robot.step(timestep) != -1:
        # 3-2. 1개 프레임 가져오기
        count_state += 1  
        environment()
        # 3-3. 3개 프레임 가져오기
        if count_state == MAX_FRAME:
            # setiing 하게 되면 초기화 버그 해결
            if set_count == 1:
                next_state = np.array(storage)
                count_state = 0
                set_count = 0
                storage = []
                continue
            # state = previous_state
            state = np.array(next_state)
            count_state = 0  
            # next state = current_state  
            next_state = np.array(storage)
            storage = []
            # state, next state에 따라 reward
            reward = Reward(state,next_state)
            avg_reward += reward
            # reach check (next_state , current_state)
            done = Done()
            # store experiences
            collect_experiences(state,next_state,action,reward,done,buffer)
            # collision check
            collision_check()
            # done check -> setting or pass
            if done == True:
                done_storage.append(1)
                setting()
                set_count = 1
            else:
                done_storage.append(0)
            # current state에 따라 action
            action = agent.collect_policy(max_episodes,episode_cnt, next_state)
            Action(action)
            # count experiences
            count_experience += 1

            # experience replay
            if count_experience == REPLAY_CYCLE:
                count_experience = 0
                experience_batch = buffer.sample_batch()
                avg_reward = avg_reward / len(experience_batch[0])
                loss = agent.train(experience_batch)
                # avg_reward = evaluate_training_result(env,agent)
                
                print('Episode {0}/{1} and so far the performance is {2} and '
                      'loss is {3}'.format(episode_cnt, max_episodes,
                                           avg_reward, loss[0]))
                reward_data.append(avg_reward)
                avg_reward = 0
                loss_data.append(loss[0])
                if episode_cnt % TARGET_NETWORK_CYCLE == 0:
                    agent.update_target_network()
                break

# 4. 결과 저장
createDirectory(f"data")
createDirectory(f"data/{DAY_NUM}")
# 4-1. loss graph
x_data = list(range(len(loss_data)))
loss_min = np.min(loss_data)
loss_max = np.max(loss_data)
plt.ylim([loss_min-0.01, loss_max + 0.01])
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.plot(x_data,loss_data,c='red',label = "loss")
plt.savefig(f'data/{DAY_NUM}/loss_{MODIFY_NUM}.png')
plt.cla()
# 4-2. reward graph
plt.xlabel('Epoche')
plt.ylabel('Reward')
reward_min = np.min(reward_data)
reward_max = np.max(reward_data)
plt.ylim([reward_min-0.01, reward_max+0.01])
plt.plot(x_data,reward_data,c='blue',label = "reward")
plt.savefig(f'data/{DAY_NUM}/reward_{MODIFY_NUM}.png')
plt.cla()
# 4-3. done graph
done_data = list(range(len(done_storage)))
plt.xlabel('Epoche')
plt.ylabel('success')
plt.ylim([0, 2])
plt.plot(done_data,done_storage,c='green',label = "done_storage")
plt.savefig(f'data/{DAY_NUM}/done_{MODIFY_NUM}.png')
plt.cla()
# 4-4. collision graph
collision_data = list(range(len(collision_storage)))
plt.xlabel('Epoche')
plt.ylabel('collision')
plt.ylim([0, 2])
plt.plot(collision_data,collision_storage,c='green',label = "collision_storage")
plt.savefig(f'data/{DAY_NUM}/collision_{MODIFY_NUM}.png')

# 5. model save
agent.q_net.save(f'data/{DAY_NUM}/{MODEL_NAME}_{MODIFY_NUM}')
original_model = agent.q_net
loaded_model = tf.keras.models.load_model(f'data/{DAY_NUM}/{MODEL_NAME}_{MODIFY_NUM}')
# Check if the weights of the two models are the same
for i, (original_weight, loaded_weight) in enumerate(zip(original_model.weights, loaded_model.weights)):
    tf.debugging.assert_equal(original_weight, loaded_weight,
                              message=f"Weight {i} is different.")
print("The weights of the two models are the same.")