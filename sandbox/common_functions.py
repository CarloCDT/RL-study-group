import cv2
import numpy as np

def process_state_image(state, env):
    state_image = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state_image = state_image.astype(float)
    state_image /= 255.0

    ## Sensors
    car_front = [65, 48]
    # Speed
    speed = np.sqrt(np.square(env.car.hull.linearVelocity[0]) + np.square(env.car.hull.linearVelocity[1]))

    # Front Sensor
    front_sensor = car_front[0]
    for i in range(car_front[0]):
        pixel = state[car_front[0]-i, car_front[1], :]

        if pixel[1]>pixel[0]*1.3 and pixel[1]>pixel[2]*1.3:
            front_sensor = i
            break

    # Front left sensor
    left_sensor = car_front[0]
    for i in range(car_front[0]):
        pixel = state[car_front[0]-i, car_front[1]-i//4, :]
        if pixel[1]>pixel[0]*1.3 and pixel[1]>pixel[2]*1.3:
            left_sensor = i
            break
        
    # Full left sensor
    full_left_sensor = car_front[1]
    for i in range(car_front[1]):
        pixel = state[car_front[0], car_front[1]-i, :]
        if pixel[1]>pixel[0]*1.3 and pixel[1]>pixel[2]*1.3:
            full_left_sensor =  i
            break

    # Front Right Sensor
    right_sensor = car_front[0]
    for i in range(car_front[0]):
        pixel = state[car_front[0]-i, car_front[1]+i//4, :]
        if pixel[1]>pixel[0]*1.3 and pixel[1]>pixel[2]*1.3:
            right_sensor = i
            break
    
    # Full right sensor
    full_right_sensor = 96-car_front[1]
    for i in range(96-car_front[1]):
        pixel = state[car_front[0], car_front[1]+i, :]
        if pixel[1]>pixel[0]*1.3 and pixel[1]>pixel[2]*1.3:
            full_right_sensor = i
            break

    state_sensors = [round(speed, 2), front_sensor, left_sensor, full_left_sensor, right_sensor, full_right_sensor]

    return [state_image, state_sensors]

def generate_state_frame_stack_from_queue(deque):
    state_image = []
    state_sensors = []
    for a in deque:
        state_image.append(a[0])
        state_sensors.append(a[1])

    frame_stack_image = np.array(state_image)
    frame_stack_sensor = np.array(state_sensors)

    # Move stack dimension to the channel dimension (stack, x, y) -> (x, y, stack)
    return [np.transpose(frame_stack_image, (1, 2, 0)), np.transpose(frame_stack_sensor, (1, 0))]
