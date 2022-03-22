import numpy as np
import threading
import time

class RadReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, image_shape, proprioception_shape, action_shape, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        self.ignore_image = True
        self.ignore_state = True

        if image_shape[-1] != 0:
            if image_shape[-1] == 1:
                self.images = np.empty((capacity, *image_shape[:2]), dtype=np.uint8)
                self.next_images = np.empty((capacity, *image_shape[:2]), dtype=np.uint8)
            else:
                self.images = np.empty((capacity, *image_shape), dtype=np.uint8)
                self.next_images = np.empty((capacity, *image_shape), dtype=np.uint8)
            self.ignore_image = False

        if proprioception_shape[-1] != 0:
            self.states = np.empty((capacity, *proprioception_shape), dtype=np.float32)
            self.next_states = np.empty((capacity, *proprioception_shape), dtype=np.float32)
            self.ignore_state = False

        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False
        self.count = 0

    def add(self, image, state, action, reward, next_image, next_state, done):
        if not self.ignore_image:
            self.images[self.idx] = image
            self.next_images[self.idx] = next_image
        if not self.ignore_state:
            self.states[self.idx]= state
            self.next_states[self.idx]= next_state
        self.actions[self.idx]= action
        self.rewards[self.idx]= reward
        self.dones[self.idx]= done

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        self.count = self.capacity if self.full else self.idx

    def sample(self):
        idxs = np.random.randint(
            0, self.count, size=min(self.count, self.batch_size)
        )
        if self.ignore_image:
            images = None
            next_images = None
        else:
            images = self.images[idxs]
            next_images = self.next_images[idxs]
            
        if self.ignore_state:
            states = None
            next_states = None
        else:
            states = self.states[idxs]
            next_states = self.next_states[idxs]
        
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        dones = self.dones[idxs]

        return images, states, actions, rewards, next_images, next_states, dones

class AsyncRadReplayBuffer(RadReplayBuffer):
    def __init__(self, image_shape, proprioception_shape, action_shape, capacity, batch_size,
                 sample_queue, minibatch_queue, init_steps, max_updates_per_step):
        super(AsyncRadReplayBuffer, self).__init__(image_shape, proprioception_shape, action_shape, capacity, batch_size)
        self.init_steps = init_steps
        self.step = 0
        self.send_count = 0
        self.max_updates_per_step = max_updates_per_step
        self.sample_queue = sample_queue
        self.minibatch_queue = minibatch_queue

        self.start_thread()

    def start_thread(self):
        threading.Thread(target=self.recv_from_env).start()
        threading.Thread(target=self.send_to_update).start()

    def recv_from_env(self):
        while True:
            self.add(*self.sample_queue.get())
            self.step += 1

    def send_to_update(self):
        while True:
            if self.send_count > (self.step - self.init_steps) * self.max_updates_per_step:
                time.sleep(0.1)
            else:
                self.minibatch_queue.put(tuple(self.sample()))
                self.send_count += 1
