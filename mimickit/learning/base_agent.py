import abc
import enum
import gymnasium.spaces as spaces
import json
import numbers
import numpy as np
import os
import random
import tempfile
import time
import torch

import envs.base_env as base_env
import learning.experience_buffer as experience_buffer
import learning.mp_optimizer as mp_optimizer
import learning.normalizer as normalizer
import learning.return_tracker as return_tracker
from util.logger import Logger
import util.mp_util as mp_util
import util.torch_util as torch_util
import util.logger as logger
import util.tb_logger as tb_logger
import util.wandb_logger as wandb_logger

import learning.distribution_gaussian_diag as distribution_gaussian_diag

class AgentMode(enum.Enum):
    TRAIN = 0
    TEST = 1

class BaseAgent(torch.nn.Module):
    CHECKPOINT_VERSION = 2

    def __init__(self, config, env, device):
        super().__init__()

        self._env = env
        self._device = device
        self._iter = 0
        self._sample_count = 0
        self._config = config
        self._load_params(config)

        self._build_normalizers()
        self._build_model(config)
        self.to(self._device)

        self._build_optimizer(config)

        self._build_exp_buffer(config)
        self._build_return_tracker()
        
        self._mode = AgentMode.TRAIN
        self._curr_obs = None
        self._curr_info = None
        self._elapsed_train_time = 0.0
        self._last_test_info = {
            "mean_return": 0.0,
            "mean_ep_len": 0.0,
            "num_eps": 0,
        }
        self._resume_pending = False
        self._resume_exp_total_samples = 0
        self._resume_exp_sampling_state = None
        self._resume_count = 0
        self._last_output_sample_count = 0
        self._last_output_wall_time = 0.0
        self._checkpoint_context = {}
        return

    def set_checkpoint_context(self, context):
        self._checkpoint_context = dict(context)
        return

    def train_model(self, max_samples, out_dir, save_int_models, logger_type):
        resume_run = self._resume_pending
        start_time = time.time() - self._elapsed_train_time

        out_model_file = os.path.join(out_dir, "model.pt")
        out_checkpoint_file = os.path.join(out_dir, "checkpoint.pt")
        log_file = os.path.join(out_dir, "log.txt")
        self._train_metrics_file = os.path.join(
            out_dir, "train_metrics.jsonl")
        if (mp_util.is_root_proc() and not resume_run):
            os.makedirs(out_dir, exist_ok=True)
            with open(self._train_metrics_file, "w"):
                pass
        self._logger = self._build_logger(logger_type, log_file, self._config,
                                          append=resume_run)
        
        if (save_int_models):
            int_out_dir = os.path.join(out_dir, "int_models")
            if (mp_util.is_root_proc() and not os.path.exists(int_out_dir)):
                os.makedirs(int_out_dir, exist_ok=True)
        else:
            int_out_dir = ""
        
        self._curr_obs, self._curr_info = self._reset_envs()
        self._init_train()
        test_info = self._last_test_info

        while self._sample_count < max_samples:
            train_info = self._train_iter()
            
            self._sample_count = self._update_sample_count()
            output_iter = (self._iter % self._iters_per_output == 0) or (self._sample_count >= max_samples)

            if (output_iter):
                test_info = self.test_model(self._test_episodes)
                self._last_test_info = test_info
            
            env_diag_info = self._env.record_diagnostics()
            self._log_train_info(train_info, test_info, env_diag_info, start_time) 
            self._logger.print_log()

            if (output_iter):
                self._logger.write_log()
                self._write_train_metrics_jsonl()
                self._elapsed_train_time = time.time() - start_time
                self._last_output_sample_count = self._sample_count
                self._last_output_wall_time = self._elapsed_train_time
                self._output_train_model(self._iter, out_model_file,
                                         out_checkpoint_file, int_out_dir)

                self._train_return_tracker.reset()
                self._curr_obs, self._curr_info = self._reset_envs()
            
            self._iter += 1

        return

    def test_model(self, num_episodes):
        self.eval()
        self.set_mode(AgentMode.TEST)
        
        num_procs = mp_util.get_num_procs()
        num_eps_proc = int(np.ceil(num_episodes / num_procs))

        with torch.no_grad():
            self._curr_obs, self._curr_info = self._reset_envs()
            test_info = self._rollout_test(num_eps_proc)

        return test_info
    
    def get_action_size(self):
        a_space = self._env.get_action_space()
        if (isinstance(a_space, spaces.Box)):
            a_size = np.prod(a_space.shape)
        elif (isinstance(a_space, spaces.Discrete)):
            a_size = 1
        else:
            assert(False), "Unsuppoted action space: {}".format(a_space)
        return a_size
    
    def set_mode(self, mode):
        self._mode = mode
        if (self._mode == AgentMode.TRAIN):
            self._env.set_mode(base_env.EnvMode.TRAIN)
        elif (self._mode == AgentMode.TEST):
            self._env.set_mode(base_env.EnvMode.TEST)
        else:
            assert(False), "Unsupported agent mode: {}".format(mode)
        return

    def get_num_envs(self):
        return self._env.get_num_envs()

    def save(self, out_file):
        if (mp_util.is_root_proc()):
            state_dict = self.state_dict()
            self._atomic_torch_save(state_dict, out_file)
        return

    def load(self, in_file):
        state_dict = self._torch_load(in_file)
        if (self._is_training_checkpoint(state_dict)):
            state_dict = state_dict["model_state_dict"]
        self.load_state_dict(state_dict)
        self._sync_optimizer()
        Logger.print("Loaded model parameters from {:s}".format(str(in_file)))
        return

    def save_checkpoint(self, out_file, next_iter=None):
        if (not mp_util.is_root_proc()):
            return

        if (next_iter is None):
            next_iter = self._iter

        checkpoint = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "metadata": {
                "agent_class": self.__class__.__name__,
                "num_envs": int(self.get_num_envs()),
                "world_size": max(1, int(mp_util.get_num_procs())),
                "checkpoint_context": dict(self._checkpoint_context),
            },
            "model_state_dict": self.state_dict(),
            "optimizer_state_dicts": {
                name: optimizer.state_dict()
                for name, optimizer in self._get_optimizers().items()
            },
            "normalizer_training_states": {
                name: module.training_state_dict()
                for name, module in self.named_modules()
                if isinstance(module, normalizer.Normalizer)
            },
            "trainer_state": {
                "next_iter": int(next_iter),
                "sample_count": int(self._sample_count),
                "exp_total_samples": int(self._exp_buffer.get_total_samples()),
                "elapsed_train_time": float(self._elapsed_train_time),
                "last_test_info": dict(self._last_test_info),
                "resume_count": int(self._resume_count),
                "last_output_sample_count": int(
                    self._last_output_sample_count),
                "last_output_wall_time": float(
                    self._last_output_wall_time),
            },
            "exp_buffer_sampling_state": (
                self._exp_buffer.sampling_state_dict()),
            "rng_state": self._get_rng_state(),
            "replay_buffer_states": {
                name: buffer.state_dict()
                for name, buffer in self._get_replay_buffers().items()
            },
        }
        self._atomic_torch_save(checkpoint, out_file)
        return

    def resume(self, in_file):
        checkpoint = self._torch_load(in_file)
        if (not self._is_training_checkpoint(checkpoint)):
            raise ValueError(
                "{} is a weights-only model. Use --model_file for evaluation "
                "or resume from checkpoint.pt.".format(in_file))

        version = int(checkpoint["checkpoint_version"])
        if (version != self.CHECKPOINT_VERSION):
            raise ValueError("Unsupported checkpoint version: {}".format(version))

        metadata = checkpoint.get("metadata", {})
        expected_metadata = {
            "agent_class": self.__class__.__name__,
            "num_envs": int(self.get_num_envs()),
            "world_size": max(1, int(mp_util.get_num_procs())),
        }
        if (expected_metadata["world_size"] != 1):
            raise ValueError(
                "Strict checkpoint resume currently supports one training "
                "process only; per-rank RNG and replay state are intentionally "
                "not approximated.")
        for key, expected_val in expected_metadata.items():
            saved_val = metadata.get(key, expected_val)
            if (saved_val != expected_val):
                raise ValueError(
                    "Checkpoint {} mismatch: saved {!r}, current {!r}."
                    .format(key, saved_val, expected_val))

        saved_context = metadata.get("checkpoint_context", {})
        if saved_context != self._checkpoint_context:
            raise ValueError(
                "Checkpoint configuration mismatch: saved context {!r}, "
                "current context {!r}.".format(
                    saved_context, self._checkpoint_context))

        self.load_state_dict(checkpoint["model_state_dict"])

        normalizers = {
            name: module
            for name, module in self.named_modules()
            if isinstance(module, normalizer.Normalizer)
        }
        saved_normalizers = checkpoint.get("normalizer_training_states", {})
        if set(normalizers.keys()) != set(saved_normalizers.keys()):
            raise ValueError(
                "Normalizer mismatch: checkpoint has {}, current agent has {}."
                .format(sorted(saved_normalizers.keys()),
                        sorted(normalizers.keys())))
        for name, module in normalizers.items():
            module.load_training_state_dict(saved_normalizers[name])

        optimizers = self._get_optimizers()
        saved_optimizers = checkpoint["optimizer_state_dicts"]
        if (set(optimizers.keys()) != set(saved_optimizers.keys())):
            raise ValueError(
                "Optimizer mismatch: checkpoint has {}, current agent has {}."
                .format(sorted(saved_optimizers.keys()),
                        sorted(optimizers.keys())))
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(saved_optimizers[name])

        replay_buffers = self._get_replay_buffers()
        saved_replay_buffers = checkpoint.get("replay_buffer_states", {})
        if (set(replay_buffers.keys()) != set(saved_replay_buffers.keys())):
            raise ValueError(
                "Replay-buffer mismatch: checkpoint has {}, current agent has {}."
                .format(sorted(saved_replay_buffers.keys()),
                        sorted(replay_buffers.keys())))
        for name, buffer in replay_buffers.items():
            buffer.load_state_dict(saved_replay_buffers[name])

        trainer_state = checkpoint["trainer_state"]
        self._iter = int(trainer_state["next_iter"])
        self._sample_count = int(trainer_state["sample_count"])
        self._resume_exp_total_samples = int(
            trainer_state["exp_total_samples"])
        self._resume_exp_sampling_state = checkpoint.get(
            "exp_buffer_sampling_state", None)
        self._elapsed_train_time = float(
            trainer_state.get("elapsed_train_time", 0.0))
        self._last_test_info = dict(trainer_state.get(
            "last_test_info", self._last_test_info))
        self._resume_count = int(trainer_state.get("resume_count", 0)) + 1
        self._last_output_sample_count = int(trainer_state.get(
            "last_output_sample_count", self._sample_count))
        self._last_output_wall_time = float(trainer_state.get(
            "last_output_wall_time", self._elapsed_train_time))

        self._sync_optimizer()
        # Restore RNG last so future synchronization implementations cannot
        # perturb the saved continuation stream.
        self._set_rng_state(checkpoint["rng_state"])
        self._resume_pending = True
        Logger.print(
            "Resuming training from {:s} at iteration {:d}, sample {:d}."
            .format(str(in_file), self._iter, self._sample_count))
        return

    def calc_num_params(self):
        params = self.parameters()
        num_params = sum(p.numel() for p in params if p.requires_grad)
        return num_params
    
    def _load_params(self, config):
        self._discount = config["discount"]
        self._iters_per_output = config["iters_per_output"]
        self._normalizer_samples = config.get("normalizer_samples", np.inf)
        self._test_episodes = config["test_episodes"]
        
        self._steps_per_iter = config["steps_per_iter"]
        self._use_mixed_precision = config.get("use_mixed_precision", False)
        return

    def _build_normalizers(self):
        obs_space = self._env.get_obs_space()
        obs_dtype = torch_util.numpy_dtype_to_torch(obs_space.dtype)
        self._obs_norm = normalizer.Normalizer(obs_space.shape, clip=10.0, device=self._device, dtype=obs_dtype)

        self._a_norm = self._build_action_normalizer()
        return
    
    def _build_action_normalizer(self):
        a_space = self._env.get_action_space()
        a_dtype = torch_util.numpy_dtype_to_torch(a_space.dtype)

        if (isinstance(a_space, spaces.Box)):
            a_mean = torch.tensor(0.5 * (a_space.high + a_space.low), device=self._device, dtype=a_dtype)
            a_std = torch.tensor(0.5 * (a_space.high - a_space.low), device=self._device, dtype=a_dtype)
            
            # ensure initialized std is strictly greater than 0 to avoid degenerate normalizer
            assert (a_std > 0).all().item(), "init_std must be > 0 for action normalizer (Box action space wrong! Check your XML file. Joints must have 'limited=true' and non-zero bounds.)"

            a_norm = normalizer.Normalizer(a_mean.shape, device=self._device, init_mean=a_mean, 
                                                 init_std=a_std, dtype=a_dtype)
        elif (isinstance(a_space, spaces.Discrete)):
            a_mean = torch.tensor([0], device=self._device, dtype=a_dtype)
            a_std = torch.tensor([1], device=self._device, dtype=a_dtype)
            a_norm = normalizer.Normalizer(a_mean.shape, device=self._device, init_mean=a_mean, 
                                                 init_std=a_std, min_std=0, dtype=a_dtype)
        else:
            assert(False), "Unsupported action space: {}".format(a_space)

        return a_norm
    
    def _build_exp_buffer(self, config):
        buffer_length = self._get_exp_buffer_length()
        batch_size = self.get_num_envs()
        self._exp_buffer = experience_buffer.ExperienceBuffer(buffer_length=buffer_length, batch_size=batch_size,
                                                              device=self._device)
        return

    def _build_return_tracker(self):
        self._train_return_tracker = return_tracker.ReturnTracker(self.get_num_envs(), self._device)
        self._test_return_tracker = return_tracker.ReturnTracker(self.get_num_envs(), self._device)
        return

    def _build_logger(self, logger_type, log_file, config, append=False):
        if (logger_type == "txt"):
            log = logger.Logger()
        elif (logger_type == "tb"):
            log = tb_logger.TBLogger()
        elif (logger_type == "wandb"):
            log = wandb_logger.WandbLogger("mimickit", config)
        else:
            assert(False), "Unsupported logger: {:s}".format(logger_type)

        log.set_step_key("Samples")
        if (mp_util.is_root_proc()):
            if (logger_type == "txt"):
                log.configure_output_file(log_file, append=append)
            else:
                log.configure_output_file(log_file)
        
        return log

    def _update_sample_count(self):
        sample_count = self._exp_buffer.get_total_samples()
        sample_count = mp_util.reduce_sum(sample_count)
        return sample_count
    
    def _init_train(self):
        if (self._resume_pending):
            # Simulator state and the partial on-policy rollout are deliberately
            # not checkpointed.  Resume starts a fresh rollout at the saved
            # iteration boundary while preserving the global sample schedule.
            self._exp_buffer.set_total_samples(
                self._resume_exp_total_samples)
            if (self._resume_exp_sampling_state is not None):
                self._exp_buffer.load_sampling_state_dict(
                    self._resume_exp_sampling_state)
            self._resume_exp_sampling_state = None
            self._resume_pending = False
        else:
            self._iter = 0
            self._sample_count = 0
            self._elapsed_train_time = 0.0
            self._resume_count = 0
            self._last_output_sample_count = 0
            self._last_output_wall_time = 0.0
            self._exp_buffer.clear()
        self._train_return_tracker.reset()
        self._test_return_tracker.reset()
        return

    def _train_iter(self):
        self._init_iter()
        
        self.eval()
        self.set_mode(AgentMode.TRAIN)

        with torch.no_grad():
            self._rollout_train(self._steps_per_iter)
        
        data_info = self._build_train_data()
        train_info = self._update_model()
        
        if (self._need_normalizer_update()):
            self._update_normalizers()

        info = {**train_info, **data_info}
        
        info["mean_return"] = self._train_return_tracker.get_mean_return().item()
        info["mean_ep_len"] = self._train_return_tracker.get_mean_ep_len().item()
        info["num_eps"] = self._train_return_tracker.get_episodes()
        
        return info

    def _init_iter(self):
        return

    def _rollout_train(self, num_steps):
        for i in range(num_steps):
            action, action_info = self._decide_action(self._curr_obs, self._curr_info)
            self._record_data_pre_step(self._curr_obs, self._curr_info, action, action_info)

            next_obs, r, done, next_info = self._step_env(action)
            self._train_return_tracker.update(r, done)
            self._record_data_post_step(next_obs, r, done, next_info)
            
            self._curr_obs, self._curr_info = self._reset_done_envs(done)
            self._exp_buffer.inc()
        return
    
    def _rollout_test(self, num_episodes):
        self._test_return_tracker.reset()

        if (num_episodes == 0):
            test_info = {
                "mean_return": 0.0,
                "mean_ep_len": 0.0,
                "num_eps": 0
            }
        else:
            num_envs = self.get_num_envs()
            # minimum number of episodes to collect per env
            # this is mitigate bias in the return estimate towards shorter episodes
            min_eps_per_env = int(np.ceil(num_episodes / num_envs))

            while self._env.is_running():
                action, action_info = self._decide_action(self._curr_obs, self._curr_info)

                next_obs, r, done, next_info = self._step_env(action)
                self._test_return_tracker.update(r, done)
            
                self._curr_obs, self._curr_info = self._reset_done_envs(done)
            
                eps_per_env = self._test_return_tracker.get_eps_per_env()
                if (torch.all(eps_per_env > min_eps_per_env - 1)):
                    break
        
            test_return = self._test_return_tracker.get_mean_return()
            test_ep_len = self._test_return_tracker.get_mean_ep_len()
            test_info = {
                "mean_return": test_return.item(),
                "mean_ep_len": test_ep_len.item(),
                "num_eps": self._test_return_tracker.get_episodes()
            }
        return test_info

    def _step_env(self, action):
        obs, r, done, info = self._env.step(action)
        return obs, r, done, info

    def _record_data_pre_step(self, obs, info, action, action_info):
        self._exp_buffer.record("obs", obs)
        self._exp_buffer.record("action", action)
        
        if (self._need_normalizer_update()):
            self._obs_norm.record(obs)
        return

    def _record_data_post_step(self, next_obs, r, done, next_info):
        self._exp_buffer.record("next_obs", next_obs)
        self._exp_buffer.record("reward", r)
        self._exp_buffer.record("done", done)
        return

    def _reset_done_envs(self, done):
        done_indices = (done != base_env.DoneFlags.NULL.value).nonzero(as_tuple=False)
        env_ids = torch.flatten(done_indices)
        obs, info = self._reset_envs(env_ids)
        return obs, info

    def _reset_envs(self, env_ids=None):
        obs, info = self._env.reset(env_ids)
        return obs, info

    def _need_normalizer_update(self):
        return self._sample_count < self._normalizer_samples

    def _update_normalizers(self):
        self._obs_norm.update()
        return

    def _build_train_data(self):
        return dict()

    def _compute_succ_val(self):
        r_succ = self._env.get_reward_succ()
        val_succ = r_succ / (1.0 - self._discount)
        return val_succ
    
    def _compute_fail_val(self):
        r_fail = self._env.get_reward_fail()
        val_fail = r_fail / (1.0 - self._discount)
        return val_fail

    def _log_train_info(self, train_info, test_info, env_diag_info, start_time):
        wall_time_secs = time.time() - start_time
        wall_time_hrs = wall_time_secs / (60 * 60) # store time in hours
        
        self._logger.log("Iteration", self._iter, collection="1_Info")
        self._logger.log("Wall_Time", wall_time_hrs, collection="1_Info")
        self._logger.log("Samples", self._sample_count, collection="1_Info")
        interval_samples = self._sample_count - self._last_output_sample_count
        interval_time = wall_time_secs - self._last_output_wall_time
        interval_sps = interval_samples / max(interval_time, 1e-8)
        self._logger.log("Samples_Per_Second", interval_sps,
                         collection="1_Info", quiet=True)
        self._logger.log("Resume_Count", self._resume_count,
                         collection="1_Info", quiet=True)

        peak_gpu_memory_mb = 0.0
        if (torch.cuda.is_available()
                and torch.device(self._device).type == "cuda"):
            peak_gpu_memory_mb = (
                torch.cuda.max_memory_allocated(self._device) / (1024 * 1024))
        self._logger.log("Peak_GPU_Memory_MB", peak_gpu_memory_mb,
                         collection="1_Info", quiet=True)

        test_return = test_info["mean_return"]
        test_ep_len = test_info["mean_ep_len"]
        test_eps = test_info["num_eps"]
        test_eps = mp_util.reduce_sum(test_eps)

        train_return_key, test_return_key = self._get_return_log_keys()
        self._logger.log(test_return_key, test_return, collection="0_Main")
        self._logger.log("Test_Episode_Length", test_ep_len, collection="0_Main", quiet=True)
        self._logger.log("Test_Episodes", test_eps, collection="1_Info", quiet=True)

        train_return = train_info.pop("mean_return")
        train_ep_len = train_info.pop("mean_ep_len")
        train_eps = train_info.pop("num_eps")
        train_eps = mp_util.reduce_sum(train_eps)

        self._logger.log(train_return_key, train_return, collection="0_Main")
        self._logger.log("Train_Episode_Length", train_ep_len, collection="0_Main", quiet=True)
        self._logger.log("Train_Episodes", train_eps, collection="1_Info", quiet=True)

        for k, v in train_info.items():
            val_name = k.title()
            if torch.is_tensor(v):
                v = v.item()
            self._logger.log(val_name, v)

        for k, v in env_diag_info.items():
            val_name = k.title()
            if torch.is_tensor(v):
                v = v.item()
            self._logger.log(val_name, v, collection="2_Env", quiet=True)
        
        obs_norm_mean = self._obs_norm.get_mean()
        obs_norm_std = self._obs_norm.get_std()

        if (obs_norm_mean.dtype == torch.float32):
            obs_norm_mean = torch.mean(torch.abs(obs_norm_mean)).item()
            obs_norm_std = torch.mean(obs_norm_std).item()
            self._logger.log("Obs_Norm_Mean", obs_norm_mean, quiet=True)
            self._logger.log("Obs_Norm_Std", obs_norm_std, quiet=True)
        
        return

    def _get_return_log_keys(self):
        """Return logger keys for environment rewards tracked during stepping."""
        return "Train_Return", "Test_Return"
    
    def _compute_action_bound_loss(self, norm_a_dist):
        loss = None
        action_space = self._env.get_action_space()

        if (isinstance(action_space, spaces.Box)):
            a_low = action_space.low
            a_high = action_space.high
            valid_bounds = np.all(np.isfinite(a_low)) and np.all(np.isfinite(a_high))

            if (valid_bounds):
                assert(isinstance(norm_a_dist, distribution_gaussian_diag.DistributionGaussianDiag))
                # assume that actions have been normalized between [-1, 1]
                bound_min = -1
                bound_max = 1
                violation_min = torch.clamp_max(norm_a_dist.mode - bound_min, 0.0)
                violation_max = torch.clamp_min(norm_a_dist.mode - bound_max, 0)
                violation = torch.sum(torch.square(violation_min), dim=-1) \
                            + torch.sum(torch.square(violation_max), dim=-1)
                loss = violation

        return loss

    def _output_train_model(self, iter, out_model_file, out_checkpoint_file,
                            int_out_dir):
        self.save(out_model_file)
        self.save_checkpoint(out_checkpoint_file, next_iter=iter + 1)

        if (int_out_dir != ""):
            int_model_file = os.path.join(int_out_dir, "model_{:010d}.pt".format(iter))
            self.save(int_model_file)
        return

    def _write_train_metrics_jsonl(self):
        if (not mp_util.is_root_proc()):
            return

        row = {
            "iteration": int(self._iter),
            "samples": int(self._sample_count),
            "resume_segment": int(self._resume_count),
        }
        for key, entry in self._logger.log_current_row.items():
            val = entry.val
            if (torch.is_tensor(val) and val.numel() == 1):
                val = val.item()
            if (isinstance(val, numbers.Integral)):
                row[key] = int(val)
            elif (isinstance(val, numbers.Real)):
                row[key] = float(val)

        with open(self._train_metrics_file, "a") as metrics_file:
            metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
        return

    def _get_optimizers(self):
        return {
            name: val
            for name, val in vars(self).items()
            if isinstance(val, mp_optimizer.MPOptimizer)
        }

    def _get_replay_buffers(self):
        return {
            name: val
            for name, val in vars(self).items()
            if (name != "_exp_buffer"
                and isinstance(val, experience_buffer.ExperienceBuffer))
        }

    def _get_rng_state(self):
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if (torch.cuda.is_available()):
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _set_rng_state(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if (torch.cuda.is_available() and "cuda" in state):
            torch.cuda.set_rng_state_all(
                [rng_state.cpu() for rng_state in state["cuda"]])
        return

    def _torch_load(self, in_file):
        # Stage on CPU so a large replay snapshot does not temporarily occupy
        # a second full copy on the training GPU. Explicit weights_only=False
        # is required for RNG tuples on torch 2.6+.
        try:
            return torch.load(in_file, map_location="cpu",
                              weights_only=False)
        except TypeError:
            return torch.load(in_file, map_location="cpu")

    def _atomic_torch_save(self, state, out_file):
        out_dir = os.path.dirname(os.path.abspath(out_file))
        os.makedirs(out_dir, exist_ok=True)
        file_handle, tmp_file = tempfile.mkstemp(
            prefix=".checkpoint_", suffix=".tmp", dir=out_dir)
        os.close(file_handle)
        try:
            torch.save(state, tmp_file)
            os.replace(tmp_file, out_file)
        finally:
            if (os.path.exists(tmp_file)):
                os.remove(tmp_file)
        return

    def _is_training_checkpoint(self, state):
        return (isinstance(state, dict)
                and "checkpoint_version" in state
                and "model_state_dict" in state)
    
    @abc.abstractmethod
    def _build_model(self, config):
        return
    
    @abc.abstractmethod
    def _build_optimizer(self, config):
        return

    @abc.abstractmethod
    def _get_exp_buffer_length(self):
        return 0
    
    @abc.abstractmethod
    def _sync_optimizer(self):
        return

    @abc.abstractmethod
    def _decide_action(self, obs, info):
        a = None
        a_info = dict()
        return a, a_info
    
    @abc.abstractmethod
    def _update_model(self):
        return
