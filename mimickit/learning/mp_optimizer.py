import torch

import util.mp_util as mp_util

class MPOptimizer():
    CHECK_SYNC_STEPS = 1000

    def __init__(self, config, param_list):
        self._param_list = param_list
        self._grad_clip = float(config.get("grad_clip", 0.0))
        self._optimizer = self._build_optimizer(config, param_list)
        self._steps = 0
        self._last_grad_norm = 0.0
        
        if (mp_util.enable_mp()):
            self._param_buffer = self._build_param_buffer()

        self.sync()
        return
    
    def step(self, loss):
        self._optimizer.zero_grad()
        loss.backward()
        
        if (mp_util.enable_mp()):
            self._aggregate_mp_grads()

        self._last_grad_norm = self._calc_grad_norm()

        if (self._enable_grad_clip()):
            self._clip_grads(self._grad_clip)

        self._optimizer.step()
        
        if (mp_util.enable_mp() and (self.get_steps() % self.CHECK_SYNC_STEPS == 0)):
            assert(self._check_synced()), "Network parameters desynchronized"

        self._steps += 1
        return self._last_grad_norm

    def get_steps(self):
        return self._steps

    def get_last_grad_norm(self):
        return self._last_grad_norm

    def state_dict(self):
        """Return all state required to continue optimizer updates."""
        return {
            "optimizer": self._optimizer.state_dict(),
            "steps": self._steps,
        }

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict["optimizer"])
        self._steps = int(state_dict.get("steps", 0))

        # Optimizer tensors are not parameters and older torch versions do not
        # always follow the module's map_location when loading them.
        if (len(self._param_list) > 0):
            device = self._param_list[0].device
            for state in self._optimizer.state.values():
                for key, val in state.items():
                    if (torch.is_tensor(val)):
                        state[key] = val.to(device=device)
        return

    def sync(self):
        with torch.no_grad():
            for param in self._param_list:
                global_param = mp_util.broadcast(param)
                param.copy_(global_param)
        return

    def _build_optimizer(self, config, param_list):
        lr = float(config["learning_rate"])
        weight_decay = float(config.get("weight_decay", 0.0))
        optimizer_type = config["type"]

        if (optimizer_type == "SGD"):
            optimizer = torch.optim.SGD(param_list, lr, momentum=0.9, weight_decay=weight_decay)
        elif (optimizer_type == "Adam"):
            optimizer = torch.optim.AdamW(param_list, lr, weight_decay=weight_decay)
        else:
            assert(False), "Unsupported optimizer type: " + optimizer_type
        return optimizer
    
    def _build_param_buffer(self):
        buffer = torch.nn.utils.parameters_to_vector(self._param_list).clone().detach()
        return buffer
    
    def _check_synced(self):
        synced = True
        for param in self._param_list:
            global_param = mp_util.broadcast(param)
            param_synced = torch.equal(param, global_param)
            if (not param_synced):
                synced = False
        
        device = self._param_list[0].device
        buffer = torch.tensor([synced], dtype=torch.int, device=device)
        mp_util.reduce_min(buffer)
        synced = buffer.item() != 0

        return synced

    def _aggregate_mp_grads(self):
        grad_list = [p.grad for p in self._param_list]
        self._param_buffer[:] = torch.nn.utils.parameters_to_vector(grad_list)
        mp_util.reduce_inplace_mean(self._param_buffer)
        torch.nn.utils.vector_to_parameters(self._param_buffer, grad_list)
        return
    
    def _enable_grad_clip(self):
        return self._grad_clip > 0.0

    def _calc_grad_norm(self):
        grad_sq_sum = None
        for param in self._param_list:
            if (param.grad is not None):
                curr_sum = torch.sum(torch.square(param.grad.detach()))
                grad_sq_sum = (curr_sum if grad_sq_sum is None
                               else grad_sq_sum + curr_sum)
        if (grad_sq_sum is None):
            return 0.0
        return torch.sqrt(grad_sq_sum).item()
    
    def _clip_grads(self, max_norm):
        torch.nn.utils.clip_grad_norm_(self._param_list, max_norm)
        return
