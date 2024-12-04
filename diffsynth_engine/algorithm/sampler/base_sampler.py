class EpsilonSampler:
    def initialize(self, latents, timesteps, sigmas):                
        raise NotImplementedError()
    
    def step(self, latents, model_output, current_step):
        raise NotImplementedError()

    def _scaling(self, x, sigma):
        sigma_data = 1
        return x * (sigma ** 2 + sigma_data ** 2) ** 0.5

    def _unscaling(self, x, sigma):
        sigma_data = 1
        return x * 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5        

    def _to_denoised(self, sigma, model_outputs, x):
        # eps prediction
        return x - sigma * model_outputs