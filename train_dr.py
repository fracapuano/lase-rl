from pprint import pprint
import argparse
import pdb
import sys
import socket
import os
import pickle
import gc
import random
import string

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch
import wandb
from stable_baselines3.common.env_util import make_vec_env
from laserenv.RandomVecEnv import (
    RandomDummyVecEnv,
    RandomSubprocVecEnv
)

import laserenv
from utils import (
    get_random_string,
    set_seed,
    get_run_name,
    create_dirs,
    save_config,
    get_actor_critic_obs_keys,
    make_wrapped_environment,
    save_object,
    create_dir,
    load_config,
    type_convert
)
from dr.doraemon import DomainRandDistribution
from policy.policy import Policy
from dr.doraemon import (
    TrainingSubRtn, 
    DomainRandDistribution, 
    DORAEMON
)

from laserenv.env_utils import EnvParametrization


def main():
    args.eval_freq = max(args.eval_freq // args.now, 1)   # Making eval_freq behave w.r.t. global timesteps, so it follows --timesteps convention
    torch.set_num_threads(max(4, args.now))  # Avoids taking up all CPUs when parallelizing with multiple environments and processes

    assert args.dr_percentage <= 1 and args.dr_percentage >= 0
    assert args.env is not None
    assert args.test_env is None, 'source and target domains should be the same. As of right now, test_env is used to test the policy on the final target DR distribution'
    if args.test_env is None:
        args.test_env = args.env


    gt_task = gym.make(args.env, **env_kwargs).unwrapped.get_task()  # ground truth dynamics parameters (static vector)
    if args.rand_all_but is not None:  # args.rand_all_but overwrites args.rand_only
        args.rand_only = np.arange(len(gt_task)).tolist()
        del args.rand_only[args.rand_all_but]


    ### Configs and Wandb
    random_string = get_random_string(5)
    run_name = "DORAEMON_"+ args.algo +'_seed'+str(args.seed)+'_'+random_string
    print(f'========== RUN_NAME: {run_name} ==========')
    pprint(vars(args))
    set_seed(args.seed)
    wandb.init(config=vars(args),
               project="DORAEMON-dev",
               name=run_name,
               save_code=True,
               tags=None,
               notes=args.notes,
               mode=args.wandb)

    run_path = "runs/"+str(args.env)+"/"+get_run_name(args)+"_"+random_string+"/"
    print('Run path:', run_path)
    create_dirs(run_path)
    save_config(vars(args), run_path)

    wandb.config.path = run_path
    wandb.config.hostname = socket.gethostname()

    # randomizing B and GDD only
    bounds_low, bounds_high = EnvParametrization().get_bounds(parameters=["B", "GDD"])
    init_distr_parameters = []
    target_distr_parameters = []

    for m, M in zip(bounds_low, bounds_high):
        # starts with sub-Gaussian(m, M)
        init_distr_parameters.append(
            {'m': m, 'M': M, 'a': args.init_beta_param, 'b': args.init_beta_param}
        )
        # targets uniform(m, M) for max entropy  
        target_distr_parameters.append(
            {'m': m, 'M': M, 'a': 1, 'b': 1}
        )

    init_distribution = DomainRandDistribution(
        dr_type='beta', 
        distr=init_distr_parameters
    )
    target_distribution = DomainRandDistribution(
        dr_type='beta', 
        distr=target_distr_parameters
    )
    
    print('init distr:')
    init_distribution.print()
    print('target distr:')
    target_distribution.print()

    # Actor & Critic input observation are masked to train with privileged information
    actor_obs_keys, critic_obs_keys = get_actor_critic_obs_keys(args)

    # Set up training
    env = make_vec_env(
        args.env, 
        n_envs=args.now, 
        seed=args.seed,
        vec_env_cls=RandomSubprocVecEnv if args.multiprocessing else RandomDummyVecEnv,
        wrapper_class=make_wrapped_environment, 
        wrapper_kwargs={'args': args, 'wrapper': 'doraemon'}, 
        env_kwargs=env_kwargs
    )

    # Evaluation episodes are not used for DORAEMON (dynamics and return samples not tracked)
    eval_env = make_vec_env(
        args.env,
        n_envs=args.now,
        seed=args.seed,
        vec_env_cls=RandomSubprocVecEnv if args.multiprocessing else RandomDummyVecEnv,
        wrapper_class=make_wrapped_environment,
        wrapper_kwargs={'args': args, 'wrapper': 'returnTracker'},
        env_kwargs=env_kwargs
    )

    max_ts_per_iter = int(args.timesteps / args.n_iters)
    assert max_ts_per_iter//args.now >= gym.make(args.env, **env_kwargs)._max_episode_steps, 'ERROR! Atleast one episode needs to be collected in between each DORAEMON iteration.'
    eval_freq = min(int(max_ts_per_iter/args.now/2), args.eval_freq)  # make sure you at least evaluate the policy 2 times per iteration

    training_subroutine = TrainingSubRtn(env,
                                     eval_env=eval_env,
                                     algo=args.algo,
                                     lr=args.lr,
                                     gamma=args.gamma,
                                     device=args.device,
                                     seed=args.seed,
                                     actor_obs_mask=actor_obs_keys,
                                     critic_obs_mask=critic_obs_keys,
                                     n_eval_episodes=args.eval_episodes,
                                     eval_freq=eval_freq,
                                     run_path=run_path,
                                     gradient_steps=args.gradient_steps,
                                     verbose=args.verbose)



    # DORAEMON loop
    assert args.performance_lb is not None
    performance_lower_bound = args.performance_lb
    
    if args.stop_at_reward_threshold:
        raise NotImplementedError('Not yet implemented!')
        performance_lb_margin = compute_abs_reward_threshold_margin(args.reward_threshold_perc_margin)
        print('Reward threshold margin:', performance_lb_margin)
    else:
        performance_lb_margin = 0  # dummy value

    doraemon = DORAEMON(training_subrtn=training_subroutine,
                performance_lower_bound=performance_lower_bound,
                kl_upper_bound=args.kl_ub,
                init_distr=init_distribution,
                target_distr=target_distribution,
                budget=args.timesteps,
                max_training_steps=max_ts_per_iter,
                stopAtRewardThreshold=args.stop_at_reward_threshold,
                reward_threshold_margin=performance_lb_margin,
                test_episodes=(args.test_episodes if not args.debug else 1),
                training_subrtn_kwargs={},
                train_until_performance_lb=args.train_until_lb,
                hard_performance_constraint=args.hard_performance_constraint,
                robust_estimate=args.robust_estimate,
                alpha_ci=args.alpha_ci,
                performance_lb_percentile=args.performance_lb_percentile,
                success_rate_condition=args.success_rate_condition,
                prior_constraint=args.prior_constraint,
                force_success_with_returns=args.force_success_with_returns,
                init_beta_param=args.init_beta_param,
                verbose=args.verbose)

    while doraemon.is_there_budget_for_iter():
        doraemon.step(ckpt_dir=run_path)

    last_policy = doraemon.previous_policy  # policy at last iteration
    eff_n_iters = len(doraemon.distr_history)


    ### Plot distributions
    fig, ax = plt.subplots(nrows=1, ncols=init_distribution.ndims, figsize=(8,5))
    alpha_step = 1/(eff_n_iters+1)
    for i, distr in enumerate(doraemon.distr_history):
        if i == len(doraemon.distr_history) - 1:  # last distribution
            target_distribution.visualize_distr(ax, alpha=0.9, color='red', label='Target')
            distr.visualize_distr(ax, alpha=0.9, color='#FFEB3B', label='Last')
            doraemon.best_distr.visualize_distr(ax, alpha=0.9, color='#43A047', label='Best')
        else:
            distr.visualize_distr(ax, alpha=(int(i+1)*alpha_step), color='blue', label=None)
    plt.legend()
    wandb.log({"distr_history": wandb.Image(fig)})
    plt.savefig(os.path.join(run_path, 'doraemon_distr_history.png'))
    plt.close()


    ### Save distributions to disk
    distr_dir = os.path.join(run_path, 'distributions')
    create_dir(distr_dir)
    save_object(doraemon.distr_history, save_dir=distr_dir, filename='distr_history')
    save_object(doraemon.best_distr, save_dir=distr_dir, filename='best_distr')


    ### Free up some memory
    del training_subroutine
    del doraemon
    del env
    gc.collect()


    ### Evaluation on target environment
    test_env = make_vec_env(
        args.test_env, 
        n_envs=args.now, 
        seed=args.seed,
        vec_env_cls=RandomSubprocVecEnv if args.multiprocessing else RandomDummyVecEnv,
        wrapper_class=make_wrapped_environment, 
        wrapper_kwargs={'args': args}, 
        env_kwargs=env_kwargs
    )
    test_env.set_dr_distribution(
        dr_type='uniform', 
        distr=[x for i in range(len(gt_task)) for x in (bounds_low[i], bounds_high[i])]
    )
    test_env.set_dr_training(True)
    policy = Policy(
        algo=args.algo, 
        env=test_env, 
        device=args.device, 
        seed=args.seed, 
        actor_obs_mask=actor_obs_keys, 
        critic_obs_mask=critic_obs_keys
    )
    
    policy.load_state_dict(last_policy)

    mean_reward, std_reward = policy.eval(n_eval_episodes=args.test_episodes)
    print('Test reward and stdev:', mean_reward, std_reward)

    wandb.run.summary["test_mean_reward"] = mean_reward
    wandb.run.summary["test_std_reward"] = std_reward


    ### Compute joint 2D heatmap values
    del test_env
    if args.compute_final_heatmap or True:
        print('\n--- Computing joint 2D heatmap values')
        compute_joint_2dheatmap_data(last_policy, run_path)

    wandb.finish()


def compute_abs_reward_threshold_margin(reward_threshold_perc_margin):
    """Compute absolute reward threshold margin given
    the percentage w.r.t. (threshold - random_policy_reward)
    """
    env = gym.make(args.test_env, **env_kwargs)
    policy = Policy(algo=args.algo, env=env, device=args.device, seed=args.seed)
    mean_reward, _ = policy.eval(n_eval_episodes=(10 if not args.debug else 1))  # random policy performance

    abs_reward = reward_threshold_perc_margin * (args.performance_lb - mean_reward)
    
    return max(abs_reward, 0)


def compute_joint_2dheatmap_data(test_policy, run_path):
    """Compute data for joint 2d-heatmap visualization"""
    dyn_pair = [0,1]

    save_dir = os.path.join(run_path, 'joint_avg_return_per_dyn')
    create_dirs(save_dir)
    target_filename = os.path.join(
        save_dir, 
        f'joint_return_per_dyns_{dyn_pair[0]}_{dyn_pair[1]}.npy'
    )

    test_env = make_vec_env(
        args.test_env,
        n_envs=args.now,
        seed=args.seed,
        vec_env_cls=RandomSubprocVecEnv if args.multiprocessing else RandomDummyVecEnv,
        wrapper_class=make_wrapped_environment,
        wrapper_kwargs={'args': args},
        env_kwargs=args.env_kwargs
    )

    actor_obs_mask, critic_obs_mask = get_actor_critic_obs_keys(args)
    policy = Policy(
        algo=args.algo, 
        env=test_env, 
        device=args.device, 
        seed=args.seed, 
        actor_obs_mask=actor_obs_mask, 
        critic_obs_mask=critic_obs_mask
    )

    policy.load_state_dict(test_policy)

    n_points_per_task_dim = 50 if not args.debug else 5
    return_per_dyn = np.empty((n_points_per_task_dim, n_points_per_task_dim))

    gt_task = gym.make(args.test_env, **env_kwargs).get_wrapper_attr("get_task")()
    lower_bounds = None
    test_bounds = gym.make(
        args.test_env, **env_kwargs
        ).get_wrapper_attr(
            "get_uniform_dr_by_percentage"
        )(
            **dict(
                percentage=args.dr_percentage,
                nominal_values=gt_task,
                lower_bounds=lower_bounds
            )
        )
    
    bounds_low, bounds_high = test_bounds[::2], test_bounds[1::2]

    test_tasks_1 = np.linspace(bounds_low[dyn_pair[0]], bounds_high[dyn_pair[0]], n_points_per_task_dim) # (50,)
    test_tasks_2 = np.linspace(bounds_low[dyn_pair[1]], bounds_high[dyn_pair[1]], n_points_per_task_dim) # (50,)

    curr_task = gt_task.copy()
    for j, test_task_1 in enumerate(test_tasks_1):
        for k, test_task_2 in enumerate(test_tasks_2):
            curr_task[dyn_pair] = [test_task_1, test_task_2]  # Change two params at a time, and keep others to the nominal values
            repeated_curr_task = np.repeat(curr_task[np.newaxis, :], args.now, axis=0)  # duplicate task args.now times to handle vec envs
            test_env.set_task(repeated_curr_task)
            mean_reward, std_reward = policy.eval(n_eval_episodes=(10 if not args.debug else 1))
            return_per_dyn[j, k] = mean_reward

            # Show progress
            print(f'[{j+1}/{n_points_per_task_dim}, {k+1}/{n_points_per_task_dim}]: {round(mean_reward, 2)} +- {round(std_reward,2)}', end="\r")

    # Print a new line after the loop finishes
    print()

    # Create dir and save matrix
    np.save(target_filename, return_per_dyn)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps',      default=1000, type=int, help='Budget. Global training timesteps (will be spread out across all parallel envs)')
    parser.add_argument('--env',            default="RandomLaserEnv", type=str, help='Train gym env')
    parser.add_argument('--test_env',       default=None, type=str, help='Test gym env')
    parser.add_argument('--group',          default=None, type=str, help='Wandb run group')
    parser.add_argument('--algo',           default='sac', type=str, help='RL Algo (ppo, lstmppo, sac)')
    parser.add_argument('--lr',             default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--gamma',          default=0.99, type=float, help='gamma discount factor')
    parser.add_argument('--now',            default=2, type=int, help='Number of cpus for parallelization')
    parser.add_argument('--multiprocessing', default=False, action='store_true', help='Use multiprocessing for env parallelization')
    parser.add_argument('--stop_at_reward_threshold', default=False, action='store_true', help='Stop at reward threshold')
    parser.add_argument('--reward_threshold_perc_margin', default=0., type=float, help='Percentage of (threshold - reward_for_random_policy) to use as margin.')
    parser.add_argument('--eval_freq',      default=10000, type=int, help='Global timesteps frequency for training evaluations')
    parser.add_argument('--eval_episodes',  default=10, type=int, help='# episodes for training evaluations')
    parser.add_argument('--test_episodes',  default=10, type=int, help='# episodes for test evaluations')
    parser.add_argument('--seed',           default=0, type=int, help='Random seed')
    parser.add_argument('--device',         default='cpu', type=str, help='<cpu,cuda,mps>')
    parser.add_argument('--notes',          default=None, type=str, help='Wandb notes')
    parser.add_argument('--wandb',          default='online', type=str, help='Wandb mode. [online, offline, disabled]')
    parser.add_argument('--verbose',        default=1, type=int, help='Verbose integer value')
    parser.add_argument('--stack_history',  default=5, type=int, help='Stack a number of previous (obs, actions) into the current obs vector. If > 1, it allows for implicit online systId, hence adaptive behavior.')
    parser.add_argument('--dr_percentage',  default=0.1, type=float, help='Percentage of ground truth values used to build the Uniform DR distribution. gt +- gt*percentage')
    parser.add_argument('--rand_only',      default=None, type=int, nargs='+', help='Index of dynamics parameter to randomize, instead of randomizing all possible parameters.')
    parser.add_argument('--rand_all_but',   default=None, type=int, help='Helper parameter that sets --rand_only [] to all indexes except for the one specified by --rand_all_but.')
    parser.add_argument('--dyn_in_obs',     default=False, action='store_true', help='If True, concatenate the dynamics of the environment in the observation vector, for task-aware policies.')
    parser.add_argument('--gradient_steps', default=-1, type=int, help='Number of gradient steps when policy is updated in sb3 using SAC. -1 means as many as --args.now')
    parser.add_argument('--debug',          default=False, action='store_true', help='Debug flag. Used to speed up some steps when they are just being tested.')
    parser.add_argument('--compute_final_heatmap', default=False, action='store_true', help='If set, compute 2D heatmap at the end of training and save results to file.')

    # Params for asymmetric information
    parser.add_argument('--actor_state_only',   default=False, action='store_true', help='History or dynamics are filtered out from the actor input')
    parser.add_argument('--actor_history_only', default=False, action='store_true', help='Dynamics are filtered out from the actor input')
    parser.add_argument('--critic_dyn_only',    default=False, action='store_true', help='History is filtered out from the critic input')

    # DORAEMON params
    parser.add_argument('--n_iters',         default=5, type=int, help='Minimum number of DORAEMON opt. iterations (could be more due to early stopping when threshold is reached, if stopAtRewardThreshold is set)')
    parser.add_argument('--performance_lb',  default=0.65, type=float, help='Performance lower bound for DORAEMON opt. problem. env.reward_threshold is used if None')
    parser.add_argument('--kl_ub',           default=1, type=float, help='KL upper bound for DORAEMON opt. problem')
    parser.add_argument('--min_dyn_samples', default=100, type=int, help='Minimum number of dynamics samples for computing performance constraint')
    parser.add_argument('--max_dyn_samples', default=1000, type=int, help='Maximum number of dynamics samples for computing performance constraint')
    parser.add_argument('--train_until_lb',  default=True, action='store_true', help='Train on initial distribution until performance lower bound is reached')
    parser.add_argument('--hard_performance_constraint', default=False, action='store_true', help='Performance constraint may not be violated. Update will be skipped if not performance threshold has been reached.')
    parser.add_argument('--robust_estimate', default=False, action='store_true', help='Use lower_confidence_bound as performance constraint instead of the sample mean.')
    parser.add_argument('--alpha_ci',        default=0.9, type=float, help='Confidence level for the lower_confidence_bound')
    parser.add_argument('--performance_lb_percentile', default=None, type=float, help='Use Percentile as performance constraint, instead of the mean.')
    parser.add_argument('--success_rate_condition',    default=0.5, type=float, help='Desired expected success rate value used as performance constraint')
    parser.add_argument('--start_from_wide_uniform',   default=False, action='store_true', help='start with the wide max entropy uniform distribution, and leverage the inverted opt. problem to find an easy region to train on initially.')
    parser.add_argument('--prior_constraint',          default=False, action='store_true', help='if true, constraint prior parameters density to be equal to uniform')
    parser.add_argument('--force_success_with_returns', default=False, action='store_true', help='If set, force using returns as a success metric condition even if env.success_metric is defined. A proper corresponding performance_lb needs to be defined.')
    parser.add_argument('--init_beta_param', default=100., type=float, help='Beta distribution initial value for parameters a and b.')

    return parser.parse_args()

args = parse_args()

# Get environment kwargs
env_kwargs = {}
args.env_kwargs = env_kwargs

if __name__ == '__main__':
    main()

