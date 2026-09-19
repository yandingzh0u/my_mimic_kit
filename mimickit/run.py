import hashlib
import numpy as np
import os
import shutil
import subprocess
import sys
import time

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.arg_parser as arg_parser
from util.logger import Logger
import util.mp_util as mp_util
import util.util as util

import torch

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)
    return

def load_args(argv):
    args = arg_parser.ArgParser()
    args.load_args(argv[1:])

    arg_file = args.parse_string("arg_file")
    if (arg_file != ""):
        succ = args.load_file(arg_file)
        assert succ, Logger.print("Failed to load args from: " + arg_file)

    return args

def build_env(args, num_envs, device, visualize):
    env_file = args.parse_string("env_config")
    engine_file = args.parse_string("engine_config")
    record_video = args.parse_bool("video", False)
    
    env = env_builder.build_env(env_file, engine_file, num_envs, device, visualize=visualize, record_video=record_video)
    return env

def build_agent(args, env, device):
    agent_file = args.parse_string("agent_config")
    agent = agent_builder.build_agent(agent_file, env, device)
    return agent

def _file_sha256(filename):
    digest = hashlib.sha256()
    with open(filename, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def build_checkpoint_context(args):
    """Content identity that prevents resuming under another experiment."""
    context = {}
    for key in ("env_config", "agent_config", "engine_config"):
        filename = args.parse_string(key)
        if filename != "":
            context[key + "_sha256"] = _file_sha256(filename)
    return context

def train(agent, max_samples, out_dir, save_int_models, logger_type):
    agent.train_model(max_samples=max_samples, out_dir=out_dir, 
                      save_int_models=save_int_models, logger_type=logger_type)
    return

def test(agent, test_episodes):
    result = agent.test_model(num_episodes=test_episodes)
    
    Logger.print("Mean Return: {}".format(result["mean_return"]))
    Logger.print("Mean Episode Length: {}".format(result["mean_ep_len"]))
    Logger.print("Episodes: {}".format(result["num_eps"]))
    return result

def save_config_files(args, out_dir):
    engine_file = args.parse_string("engine_config")
    if (engine_file != ""):
        copy_file_to_dir(engine_file, "engine_config.yaml", out_dir)

    env_file = args.parse_string("env_config")
    if (env_file != ""):
        copy_file_to_dir(env_file, "env_config.yaml", out_dir)

    agent_file = args.parse_string("agent_config")
    if (agent_file != ""):
        copy_file_to_dir(agent_file, "agent_config.yaml", out_dir)
    return

def create_output_dir(out_dir):
    if (mp_util.is_root_proc()):
        if (out_dir != "" and (not os.path.exists(out_dir))):
            os.makedirs(out_dir, exist_ok=True)
    return

def copy_file_to_dir(in_path, out_filename, output_dir):
    out_file = os.path.join(output_dir, out_filename)
    shutil.copy(in_path, out_file)
    return

def set_rand_seed(args):
    rand_seed_key = "rand_seed"

    if (args.has_key(rand_seed_key)):
        rand_seed = args.parse_int(rand_seed_key)
    else:
        rand_seed = np.uint64(time.time() * 256)
        
    rand_seed += np.uint64(41 * mp_util.get_proc_rank())
    print("Setting seed: {}".format(rand_seed))
    util.set_rand_seed(rand_seed)
    return

def run(rank, num_procs, device, master_port, args):
    mode = args.parse_string("mode", "train")
    num_envs = args.parse_int("num_envs", 1)
    visualize = args.parse_bool("visualize", True)
    logger_type = args.parse_string("logger", "txt")
    model_file = args.parse_string("model_file", "")
    resume_file = args.parse_string("resume_file", "")

    if (model_file != "" and resume_file != ""):
        raise ValueError("--model_file and --resume_file are mutually exclusive.")
    if (resume_file != "" and mode != "train"):
        raise ValueError("--resume_file is only valid in train mode.")

    out_dir = args.parse_string("out_dir", "output/")
    save_int_models = args.parse_bool("save_int_models", False)
    max_samples = args.parse_int("max_samples", np.iinfo(np.int64).max)

    if (resume_file != ""):
        resume_dir = os.path.dirname(os.path.abspath(resume_file))
        requested_dir = os.path.abspath(out_dir)
        if requested_dir not in (os.path.abspath("output"),
                                 os.path.abspath("output/"),
                                 resume_dir):
            raise ValueError(
                "--resume_file must resume in its original directory: {}"
                .format(resume_dir))
        out_dir = resume_dir

    mp_util.init(rank, num_procs, device, master_port)

    set_rand_seed(args)
    set_np_formatting()
    create_output_dir(out_dir)

    env = build_env(args, num_envs, device, visualize)
    agent = build_agent(args, env, device)
    agent.set_checkpoint_context(build_checkpoint_context(args))

    if (model_file != ""):
        agent.load(model_file)
    elif (resume_file != ""):
        agent.resume(resume_file)
        # Diagnostic only: match the phase-zero reset following an output
        # checkpoint in uninterrupted training. This does not restore PhysX.
        resume_initial_reset = args.parse_string("resume_initial_reset", "train")
        if resume_initial_reset not in ("train", "test"):
            raise ValueError("resume_initial_reset must be train or test")
        if resume_initial_reset == "test":
            agent.eval()
            agent.set_mode(type(agent._mode).TEST)
        print("Resume initial reset mode: {}".format(resume_initial_reset), flush=True)

    if (mode == "train"):
        if (resume_file == ""):
            save_config_files(args, out_dir)
        train(agent=agent, max_samples=max_samples, out_dir=out_dir, 
              save_int_models=save_int_models, logger_type=logger_type)
        
    elif (mode == "test"):
        test_episodes = args.parse_int("test_episodes", np.iinfo(np.int64).max)
        test(agent=agent, test_episodes=test_episodes)

    else:
        assert(False), "Unsupported mode: {}".format(mode)

    return

def check_pcie_link():
    """Warn when a GPU is running on a degraded PCIe link.

    Simulation throughput and host<->device transfers collapse when a card
    negotiates fewer lanes than it supports (e.g. a x16 card sitting in a x4
    slot, or a riser that only wires one lane).  nvidia-smi reports this
    directly, so the training entry point checks it once and reports it here
    instead of leaving it to a console warning buried in the startup log.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,pcie.link.gen.current,pcie.link.gen.max,"
             "pcie.link.width.current,pcie.link.width.max",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
    except Exception:
        return
    for line in out.stdout.strip().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        index, name, _, _, width_cur, width_max = fields
        if not (width_cur.isdigit() and width_max.isdigit()):
            continue
        if int(width_cur) < int(width_max):
            Logger.print(
                "WARNING: GPU {} ({}) negotiated PCIe x{} but supports x{}. "
                "Check the slot (prefer a CPU x16 slot), any riser/adapter and "
                "the BIOS bifurcation setting; effective bandwidth is reduced "
                "up to {}x.".format(index, name, width_cur, width_max,
                                    int(width_max) // max(int(width_cur), 1)))


def main(argv):
    root_rank = 0
    args = load_args(argv)
    check_pcie_link()
    master_port = args.parse_int("master_port", None)
    devices = args.parse_strings("devices", ["cuda:0"])
    
    num_workers = len(devices)
    assert(num_workers > 0)
    
    # if master port is not specified, then pick a random one
    if (master_port is None):
        master_port = np.random.randint(6000, 7000)

    torch.multiprocessing.set_start_method("spawn")

    processes = []
    for rank in range(1, num_workers):
        curr_device = devices[rank]
        proc = torch.multiprocessing.Process(target=run, args=[rank, num_workers, curr_device, master_port, args])
        proc.start()
        processes.append(proc)
    
    root_device = devices[0]
    run(root_rank, num_workers, root_device, master_port, args)

    for proc in processes:
        proc.join()
       
    return

if __name__ == "__main__":
    main(sys.argv)
