import cv2
import numpy as np

def process_state_image(state, env):
    state_image = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state_image = state_image.astype(float)
    state_image /= 255.0
    return state_image

def generate_state_frame_stack_from_queue(deque):

    return np.transpose(np.array(deque), (1, 2, 0))