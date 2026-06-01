import math


class CosineDecayLR:
    def __init__(self, base_lr, flat_steps, total_steps, min_lr, new_alg = True, last_step=-1):
        """
        Initializes the CosineDecayLR scheduler.

        Args:
            base_lr (float): The starting (maximum) learning rate.
            flat_steps (int): Number of steps with a flat (constant) learning rate.
            total_steps (int): Total number of steps for the cosine decay.
            min_lr (float): Minimum learning rate at the end of decay.
            last_step (int, optional): The index of the last completed step. Defaults to -1.
        """
        self.base_lr = base_lr
        self.flat_steps = flat_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        self.new_alg = new_alg
        self.last_step = last_step

    def step(self):
        """
        Updates the step counter and returns the current learning rate.
        """
        self.last_step += 1
        return self.get_lr()

    def get_lr(self):
        """
        Computes the learning rate at the current step.

        Returns:
            float: The current learning rate.
        """
        current_step = self.last_step + 1

        if current_step < self.flat_steps:
            return self.base_lr
        elif current_step <= self.total_steps:
            # Cosine decay
            decay_steps = current_step - self.flat_steps
            total_decay_steps = self.total_steps - self.flat_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_steps / total_decay_steps))
            return max(self.min_lr, self.base_lr * cosine_decay)
        else:
            # After total_steps, maintain min_lr
            return self.min_lr