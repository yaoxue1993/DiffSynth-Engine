import torch
import matplotlib.pyplot as plt

def linear_beta_schedule(beta_start: float = 0.00085, beta_end: float = 0.0120, num_train_steps: int = 1000):
    """
    DDPM Schedule
    """
    return torch.linspace(beta_start, beta_end, num_train_steps)

def scaled_linear_beta_schedule(beta_start: float = 0.00085, beta_end: float = 0.0120, num_train_steps: int = 1000):
    """ 
    Stable Diffusion Schedule
    """
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_steps) ** 2

# Generate the schedules
num_train_steps = 1000
linear_betas = linear_beta_schedule(num_train_steps=num_train_steps)
scaled_linear_betas = scaled_linear_beta_schedule(num_train_steps=num_train_steps)

# Plotting
plt.figure(figsize=(10, 6))

# Linear schedule
plt.plot(range(num_train_steps), linear_betas, label='Linear (DDPM)')

# Scaled linear schedule
plt.plot(range(num_train_steps), scaled_linear_betas, label='Scaled Linear (Stable Diffusion)', linestyle='--')


# Define a function to mark and annotate points
def mark_and_annotate_points(x_values, y_values, color):
    for i, x in enumerate(x_values):
        point = (x, y_values[x])
        plt.scatter(point[0], point[1], color=color)
        plt.annotate(f'({point[0]}, {point[1]:.5f})',
                     xy=point,
                     xytext=(10, (-30 if i == 0 else 30)),  # Adjust position of annotation
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

# Mark and annotate start and end points for both schedules
mark_and_annotate_points([0, num_train_steps-1], linear_betas, 'orange')


plt.title('Beta Schedule over Timesteps')
plt.xlabel('Timestep (t)')
plt.ylabel('Beta Value')
plt.legend()
plt.grid(True)
plt.savefig('beta_schedule.png', format='png', dpi=600) 
plt.show()
