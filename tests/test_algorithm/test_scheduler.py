import unittest
import torch
from ..common.test_case import ImageTestCase
from diffsynth_engine.algorithm.noise_scheduler import ScaledLinearScheduler, ExponentialScheduler, KarrasScheduler, BetaScheduler
from diffsynth_engine.algorithm.noise_scheduler.flow_match.recifited_flow import RecifitedFlowScheduler
from diffsynth_engine.pipelines.flux_image import calculate_shift

class TestScheduler(ImageTestCase):
    def test_linear_scheduler(self):
        scheduler = ScaledLinearScheduler()
        sigmas, timesteps = scheduler.schedule(20)
        expect_tensors = self.get_expect_tensor("algorithm/scaled_linear_20steps.safetensors")
        self.assertTensorEqual(sigmas, expect_tensors["sigmas"])
        self.assertTensorEqual(timesteps, expect_tensors["timesteps"])

    def test_karras_scheduler(self):
        scheduler = KarrasScheduler()
        sigmas, timesteps = scheduler.schedule(20)
        expect_tensors = self.get_expect_tensor("algorithm/karras_20steps.safetensors")
        self.assertTensorEqual(sigmas, expect_tensors["sigmas"])
        self.assertTensorEqual(timesteps, scheduler.sigma_to_t(expect_tensors["sigmas"][:-1]))

    def test_exponential_scheduler(self):
        scheduler = ExponentialScheduler()
        sigmas, timesteps = scheduler.schedule(20)
        expect_tensors = self.get_expect_tensor("algorithm/exponential_20steps.safetensors")
        self.assertTensorEqual(sigmas, expect_tensors["sigmas"])
        self.assertTensorEqual(timesteps, scheduler.sigma_to_t(expect_tensors["sigmas"][:-1]))

    def test_beta_scheduler(self):
        scheduler = BetaScheduler()
        sigmas, timesteps = scheduler.schedule(20)
        expect_tensors = self.get_expect_tensor("algorithm/beta_20steps.safetensors")        
        self.assertTensorEqual(sigmas, expect_tensors["sigmas"])
        self.assertTensorEqual(timesteps, scheduler.sigma_to_t(expect_tensors["sigmas"][:-1]))

    def test_recifited_flow_scheduler(self):
        # FLUX
        width = 1024
        height = 1024
        num_inference_steps = 20
        sigmas = torch.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        scheduler = RecifitedFlowScheduler(use_dynamic_shifting=True)
        sigmas, timesteps = scheduler.schedule(num_inference_steps, mu=calculate_shift(width//16 * height//16), sigmas=sigmas)
        expect_tensors = self.get_expect_tensor("algorithm/recifited_flow_20steps_flux.safetensors")
        self.assertTensorEqual(sigmas, expect_tensors["sigmas"])
        self.assertTensorEqual(timesteps, expect_tensors["timesteps"])

        # SD3/SD3.5
        scheduler = RecifitedFlowScheduler(shift=3, use_dynamic_shifting=False)
        sigmas, timesteps = scheduler.schedule(20)
        expect_tensors = self.get_expect_tensor("algorithm/recifited_flow_20steps_sd3.safetensors")
        self.assertTensorEqual(sigmas, expect_tensors["sigmas"])
        self.assertTensorEqual(timesteps, expect_tensors["timesteps"])
    

if __name__ == "__main__":
    unittest.main()
