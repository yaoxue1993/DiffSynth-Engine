class EpsilonSampler:
    def _scaling(self, x, sigma):
        sigma_data = 1
        return x * (sigma ** 2 + sigma_data ** 2) ** 0.5

    def _unscaling(self, x, sigma):
        sigma_data = 1
        return x * 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5        

    def _to_denoised(self, sigma, model_pred, x):
        # eps prediction
        return x - sigma * model_pred