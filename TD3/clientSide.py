import torch
import os
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

class clientSide: 
    def __init__(self, PIDSIM, agent):
        self.PIDSIM = PIDSIM
        self.agent = agent
        self.training_history = []
        self.best_reward = float('-inf')
        self.best_model_path = None
        self.performance_data = []

    def train_agent(self, num_sessions=5, episodes_per_session=500, batch_size=100, save_path='./SavedModels', 
                   auto_save_interval=50, save_best_only=False, plot_performance=True, overwrite_saves=True):

        if overwrite_saves:
            session_save_path = os.path.join(save_path, "current_training_session")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_save_path = os.path.join(save_path, f"training_session_{timestamp}")
        
        os.makedirs(session_save_path, exist_ok=True)
        
        config = {
            'num_sessions': num_sessions,
            'episodes_per_session': episodes_per_session,
            'batch_size': batch_size,
            'auto_save_interval': auto_save_interval,
            'save_best_only': save_best_only,
            'overwrite_saves': overwrite_saves,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(session_save_path, 'training_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        total_episodes = 0
        session_rewards = []
        
        for session in range(num_sessions):
            print(f"Starting Training Session {session+1}/{num_sessions}")
            
            session_reward = 0
            episode_rewards = []
            
            for episode in range(episodes_per_session):
                env = self.PIDSIM()
                state = env.reset()
                episode_reward = 0
                
                episode_performance = {
                    'session': session + 1,
                    'episode': episode + 1,
                    'total_episode': total_episodes + episode + 1,
                    'times': [],
                    'motor_speeds': [],
                    'desired_speeds': [],
                    'pid_gains': [],
                    'rewards': []
                }
                
                for step in range(200):
                    action = self.agent.select_action(state)
                    noise = np.random.normal(0, 0.1, size=action.shape)
                    action = np.clip(action + noise, self.agent.action_low.cpu().numpy(), self.agent.action_high.cpu().numpy())
                    
                    next_state, reward, done, info = env.step(action)
                    
                    self.agent.replay_buffer.append(state, action, reward, next_state, float(done))
                    self.agent.train(batch_size)
                    
                    episode_performance['times'].append(step * 0.1)
                    episode_performance['motor_speeds'].append(state[0])
                    episode_performance['desired_speeds'].append(env.desired_state)
                    episode_performance['pid_gains'].append(action.copy())
                    episode_performance['rewards'].append(reward)
                    
                    state = next_state
                    episode_reward += reward
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
                session_reward += episode_reward
                
                self.performance_data.append(episode_performance)
                
                if (episode + 1) % auto_save_interval == 0:
                    avg_reward = np.mean(episode_rewards[-auto_save_interval:])
                    print(f"Auto-saved model at Session {session+1}, Episode {episode+1} (Avg Reward: {avg_reward:.2f})")
                    
                    checkpoint_dir = os.path.join(session_save_path, f"episode_{episode+1}")
                    self.save_model(checkpoint_dir, f"checkpoint_episode_{episode+1}", overwrite=overwrite_saves)
                
                if (episode + 1) % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    print(f"Session {session+1}, Episode {episode+1}, Reward: {episode_reward:.2f}, Noise: {noise[0]:.4f}")
            
            total_episodes += episodes_per_session
            session_rewards.append(session_reward)
            
            session_dir = os.path.join(session_save_path, f"session_{session+1}_complete")
            self.save_model(session_dir, f"session_{session+1}_complete", overwrite=overwrite_saves)
            
            print(f"Model saved after session {session+1} (Avg Reward: {np.mean(episode_rewards):.2f})")
            
            if plot_performance:
                self.plot_session_performance(session + 1, session_save_path)
        
        if plot_performance:
            self.plot_training_summary(session_save_path)
        
        return session_rewards

    def plot_session_performance(self, session_num, save_path):
        
        session_data = [data for data in self.performance_data if data['session'] == session_num]
        
        if not session_data:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Session {session_num} Performance', fontsize=16)
        
        last_episode = session_data[-1]
        times = last_episode['times']
        motor_speeds = last_episode['motor_speeds']
        desired_speeds = last_episode['desired_speeds']
        
        axes[0, 0].plot(times, motor_speeds, 'b-', label='Motor Speed', linewidth=2)
        axes[0, 0].plot(times, desired_speeds, 'r--', label='Desired Speed', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Speed')
        axes[0, 0].set_title(f'Motor Response (Episode {last_episode["episode"]})')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        pid_gains = np.array(last_episode['pid_gains'])
        axes[0, 1].plot(times, pid_gains[:, 0], 'g-', label='Kp', linewidth=2)
        axes[0, 1].plot(times, pid_gains[:, 1], 'b-', label='Ki', linewidth=2)
        axes[0, 1].plot(times, pid_gains[:, 2], 'r-', label='Kd', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('PID Gains')
        axes[0, 1].set_title('PID Gains Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        episode_rewards = [data['rewards'][-1] if data['rewards'] else 0 for data in session_data]
        episode_numbers = [data['episode'] for data in session_data]
        
        axes[1, 0].plot(episode_numbers, episode_rewards, 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Episode Reward')
        axes[1, 0].set_title('Episode Rewards')
        axes[1, 0].grid(True)
        
        avg_motor_speeds = []
        avg_desired_speeds = []
        episode_numbers_avg = []
        
        for data in session_data:
            if data['motor_speeds'] and data['desired_speeds']:
                avg_motor_speeds.append(np.mean(data['motor_speeds']))
                avg_desired_speeds.append(np.mean(data['desired_speeds']))
                episode_numbers_avg.append(data['episode'])
        
        if avg_motor_speeds:
            axes[1, 1].plot(episode_numbers_avg, avg_motor_speeds, 'b-', label='Avg Motor Speed', linewidth=2)
            axes[1, 1].plot(episode_numbers_avg, avg_desired_speeds, 'r--', label='Avg Desired Speed', linewidth=2)
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Average Speed')
            axes[1, 1].set_title('Average Speed vs Episode')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'session_{session_num}_performance.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        avg_session_reward = np.mean(episode_rewards)
        print(f"Session {session_num} Summary:")
        print(f"   Average Episode Reward: {avg_session_reward:.2f}")
        print(f"   Best Episode Reward: {max(episode_rewards):.2f}")
        print(f"   Worst Episode Reward: {min(episode_rewards):.2f}")
        
        if avg_motor_speeds:
            avg_tracking_error = np.mean(np.abs(np.array(avg_motor_speeds) - np.array(avg_desired_speeds)))
            print(f"   Average Tracking Error: {avg_tracking_error:.4f}")

    def plot_training_summary(self, save_path):
        
        if not self.performance_data:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Summary', fontsize=16)
        
        all_rewards = [data['rewards'][-1] if data['rewards'] else 0 for data in self.performance_data]
        episode_numbers = [data['total_episode'] for data in self.performance_data]
        
        axes[0, 0].plot(episode_numbers, all_rewards, 'purple', alpha=0.7, linewidth=1)
        
        window_size = min(50, len(all_rewards) // 10)
        if window_size > 1:
            moving_avg = np.convolve(all_rewards, np.ones(window_size)/window_size, mode='valid')
            moving_avg_episodes = episode_numbers[window_size-1:]
            axes[0, 0].plot(moving_avg_episodes, moving_avg, 'red', linewidth=2, label=f'{window_size}-episode moving average')
            axes[0, 0].legend()
        
        axes[0, 0].set_xlabel('Total Episode')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].grid(True)
        
        tracking_errors = []
        for data in self.performance_data:
            if data['motor_speeds'] and data['desired_speeds']:
                error = np.mean(np.abs(np.array(data['motor_speeds']) - np.array(data['desired_speeds'])))
                tracking_errors.append(error)
            else:
                tracking_errors.append(0)
        
        if tracking_errors:
            axes[0, 1].plot(episode_numbers, tracking_errors, 'orange', linewidth=2)
            axes[0, 1].set_xlabel('Total Episode')
            axes[0, 1].set_ylabel('Average Tracking Error')
            axes[0, 1].set_title('Tracking Error Over Time')
            axes[0, 1].grid(True)
        
        best_episode_idx = np.argmax(all_rewards)
        best_episode = self.performance_data[best_episode_idx]
        
        if best_episode['times']:
            axes[1, 0].plot(best_episode['times'], best_episode['motor_speeds'], 'b-', label='Motor Speed', linewidth=2)
            axes[1, 0].plot(best_episode['times'], best_episode['desired_speeds'], 'r--', label='Desired Speed', linewidth=2)
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Speed')
            axes[1, 0].set_title(f'Best Performance (Episode {best_episode["total_episode"]})')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        avg_kp = []
        avg_ki = []
        avg_kd = []
        
        for data in self.performance_data:
            if data['pid_gains']:
                gains = np.array(data['pid_gains'])
                avg_kp.append(np.mean(gains[:, 0]))
                avg_ki.append(np.mean(gains[:, 1]))
                avg_kd.append(np.mean(gains[:, 2]))
            else:
                avg_kp.append(0)
                avg_ki.append(0)
                avg_kd.append(0)
        
        if avg_kp:
            axes[1, 1].plot(episode_numbers, avg_kp, 'g-', label='Avg Kp', linewidth=2)
            axes[1, 1].plot(episode_numbers, avg_ki, 'b-', label='Avg Ki', linewidth=2)
            axes[1, 1].plot(episode_numbers, avg_kd, 'r-', label='Avg Kd', linewidth=2)
            axes[1, 1].set_xlabel('Total Episode')
            axes[1, 1].set_ylabel('Average PID Gains')
            axes[1, 1].set_title('PID Gains Evolution')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_summary.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Overall Training Summary:")
        print(f"   Total Episodes: {len(self.performance_data)}")
        print(f"   Best Episode Reward: {max(all_rewards):.2f}")
        print(f"   Average Episode Reward: {np.mean(all_rewards):.2f}")
        if tracking_errors:
            print(f"   Final Average Tracking Error: {tracking_errors[-1]:.4f}")
            print(f"   Best Tracking Error: {min(tracking_errors):.4f}")

    def test_agent_performance(self, num_tests=5, plot_results=True):
        
        print(f"Testing agent performance with {num_tests} trials...")
        
        test_results = []
        
        for test_num in range(num_tests):
            env = self.PIDSIM()
            state = env.reset()
            
            times = []
            motor_speeds = []
            desired_speeds = []
            pid_gains = []
            rewards = []
            
            for step in range(200):
                action = self.agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                times.append(step * 0.1)
                motor_speeds.append(state[0])
                desired_speeds.append(env.desired_state)
                pid_gains.append(action.copy())
                rewards.append(reward)
                
                state = next_state
                if done:
                    break
            
            test_results.append({
                'test_num': test_num + 1,
                'times': times,
                'motor_speeds': motor_speeds,
                'desired_speeds': desired_speeds,
                'pid_gains': pid_gains,
                'rewards': rewards,
                'total_reward': sum(rewards),
                'avg_tracking_error': np.mean(np.abs(np.array(motor_speeds) - np.array(desired_speeds)))
            })
        
        if plot_results:
            self.plot_test_results(test_results)
        
        return test_results

    def plot_test_results(self, test_results):
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Agent Performance Test Results', fontsize=16)
        
        for result in test_results:
            axes[0, 0].plot(result['times'], result['motor_speeds'], alpha=0.7, linewidth=1)
        
        if test_results:
            axes[0, 0].plot(test_results[0]['times'], test_results[0]['desired_speeds'], 'r--', linewidth=2, label='Desired Speed')
        
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Motor Speed')
        axes[0, 0].set_title('All Test Runs')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        best_test = max(test_results, key=lambda x: x['total_reward'])
        axes[0, 1].plot(best_test['times'], best_test['motor_speeds'], 'b-', label='Motor Speed', linewidth=2)
        axes[0, 1].plot(best_test['times'], best_test['desired_speeds'], 'r--', label='Desired Speed', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Speed')
        axes[0, 1].set_title(f'Best Test Run (Reward: {best_test["total_reward"]:.2f})')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        pid_gains = np.array(best_test['pid_gains'])
        axes[1, 0].plot(best_test['times'], pid_gains[:, 0], 'g-', label='Kp', linewidth=2)
        axes[1, 0].plot(best_test['times'], pid_gains[:, 1], 'b-', label='Ki', linewidth=2)
        axes[1, 0].plot(best_test['times'], pid_gains[:, 2], 'r-', label='Kd', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('PID Gains')
        axes[1, 0].set_title('PID Gains (Best Test)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        total_rewards = [r['total_reward'] for r in test_results]
        tracking_errors = [r['avg_tracking_error'] for r in test_results]
        
        x_pos = range(len(test_results))
        axes[1, 1].bar([x - 0.2 for x in x_pos], total_rewards, 0.4, label='Total Reward', alpha=0.7)
        axes[1, 1].bar([x + 0.2 for x in x_pos], tracking_errors, 0.4, label='Tracking Error', alpha=0.7)
        axes[1, 1].set_xlabel('Test Number')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels([f'Test {i+1}' for i in range(len(test_results))])
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Test Results Summary:")
        print(f"   Average Total Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"   Best Total Reward: {max(total_rewards):.2f}")
        print(f"   Average Tracking Error: {np.mean(tracking_errors):.4f} ± {np.std(tracking_errors):.4f}")
        print(f"   Best Tracking Error: {min(tracking_errors):.4f}")

    def plot_realtime_performance(self, episode_num=None):
        if not self.performance_data:
            print("No performance data available. Run training first.")
            return
        
        if episode_num is None:
            episode_data = self.performance_data[-1]
            episode_num = episode_data['total_episode']
        else:
            episode_data = None
            for data in self.performance_data:
                if data['total_episode'] == episode_num:
                    episode_data = data
                    break
            
            if episode_data is None:
                print(f"Episode {episode_num} not found in performance data.")
                return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Episode {episode_num} Performance (Session {episode_data["session"]})', fontsize=16)
        
        times = episode_data['times']
        motor_speeds = episode_data['motor_speeds']
        desired_speeds = episode_data['desired_speeds']
        pid_gains = np.array(episode_data['pid_gains'])
        rewards = episode_data['rewards']
        
        axes[0, 0].plot(times, motor_speeds, 'b-', label='Motor Speed', linewidth=2)
        axes[0, 0].plot(times, desired_speeds, 'r--', label='Desired Speed', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Speed')
        axes[0, 0].set_title('Motor Response')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(times, pid_gains[:, 0], 'g-', label='Kp', linewidth=2)
        axes[0, 1].plot(times, pid_gains[:, 1], 'b-', label='Ki', linewidth=2)
        axes[0, 1].plot(times, pid_gains[:, 2], 'r-', label='Kd', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('PID Gains')
        axes[0, 1].set_title('PID Gains Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(times, rewards, 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].set_title('Rewards Over Time')
        axes[1, 0].grid(True)
        
        tracking_error = np.abs(np.array(motor_speeds) - np.array(desired_speeds))
        axes[1, 1].plot(times, tracking_error, 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Tracking Error')
        axes[1, 1].set_title('Tracking Error Over Time')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        total_reward = sum(rewards)
        avg_tracking_error = np.mean(tracking_error)
        max_tracking_error = np.max(tracking_error)
        
        print(f"Episode {episode_num} Summary:")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Average Tracking Error: {avg_tracking_error:.4f}")
        print(f"   Max Tracking Error: {max_tracking_error:.4f}")
        print(f"   Final Motor Speed: {motor_speeds[-1]:.4f}")
        print(f"   Desired Speed: {desired_speeds[-1]:.4f}")
        print(f"   Final PID Gains - Kp: {pid_gains[-1, 0]:.4f}, Ki: {pid_gains[-1, 1]:.4f}, Kd: {pid_gains[-1, 2]:.4f}")

    def plot_training_progress(self):
        if not self.performance_data:
            print("No performance data available. Run training first.")
            return
        
        episode_numbers = [data['total_episode'] for data in self.performance_data]
        episode_rewards = [sum(data['rewards']) for data in self.performance_data]
        tracking_errors = []
        
        for data in self.performance_data:
            if data['motor_speeds'] and data['desired_speeds']:
                error = np.mean(np.abs(np.array(data['motor_speeds']) - np.array(data['desired_speeds'])))
                tracking_errors.append(error)
            else:
                tracking_errors.append(0)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress Overview', fontsize=16)
        
        axes[0, 0].plot(episode_numbers, episode_rewards, 'purple', alpha=0.7, linewidth=1)
        
        window_size = min(20, len(episode_rewards) // 10)
        if window_size > 1:
            moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            moving_avg_episodes = episode_numbers[window_size-1:]
            axes[0, 0].plot(moving_avg_episodes, moving_avg, 'red', linewidth=2, label=f'{window_size}-episode moving average')
            axes[0, 0].legend()
        
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(episode_numbers, tracking_errors, 'orange', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Tracking Error')
        axes[0, 1].set_title('Tracking Error Over Time')
        axes[0, 1].grid(True)
        
        best_episode_idx = np.argmax(episode_rewards)
        worst_episode_idx = np.argmin(episode_rewards)
        
        best_data = self.performance_data[best_episode_idx]
        worst_data = self.performance_data[worst_episode_idx]
        
        if best_data['times'] and worst_data['times']:
            axes[1, 0].plot(best_data['times'], best_data['motor_speeds'], 'g-', label=f'Best (Ep {best_data["total_episode"]})', linewidth=2)
            axes[1, 0].plot(worst_data['times'], worst_data['motor_speeds'], 'r-', label=f'Worst (Ep {worst_data["total_episode"]})', linewidth=2)
            axes[1, 0].plot(best_data['times'], best_data['desired_speeds'], 'k--', label='Desired Speed', linewidth=2)
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Motor Speed')
            axes[1, 0].set_title('Best vs Worst Performance')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        axes[1, 1].hist(episode_rewards, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].axvline(np.mean(episode_rewards), color='red', linestyle='--', label=f'Mean: {np.mean(episode_rewards):.2f}')
        axes[1, 1].set_xlabel('Episode Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Training Progress Summary:")
        print(f"   Total Episodes: {len(self.performance_data)}")
        print(f"   Best Episode Reward: {max(episode_rewards):.2f}")
        print(f"   Worst Episode Reward: {min(episode_rewards):.2f}")
        print(f"   Average Episode Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"   Best Tracking Error: {min(tracking_errors):.4f}")
        print(f"   Average Tracking Error: {np.mean(tracking_errors):.4f} ± {np.std(tracking_errors):.4f}")

    def save_model(self, save_path, model_name, overwrite=True):
        if overwrite:
            model_dir = os.path.join(save_path, model_name)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = os.path.join(save_path, f"{model_name}_{timestamp}")
        
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save(self.agent.actor.state_dict(), os.path.join(model_dir, 'actor.pth'))
        torch.save(self.agent.critic.state_dict(), os.path.join(model_dir, 'critic.pth'))
        torch.save(self.agent.actor_target.state_dict(), os.path.join(model_dir, 'actor_target.pth'))
        torch.save(self.agent.critic_target.state_dict(), os.path.join(model_dir, 'critic_target.pth'))
        
        torch.save(self.agent.actor_optimizer.state_dict(), os.path.join(model_dir, 'actor_optimizer.pth'))
        torch.save(self.agent.critic_optimizer.state_dict(), os.path.join(model_dir, 'critic_optimizer.pth'))
        
        metadata = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'agent_config': {
                'state_dim': self.agent.actor.dense_1.in_features,
                'action_dim': self.agent.actor.dense_3.out_features,
                'action_low': self.agent.action_low.cpu().numpy().tolist(),
                'action_high': self.agent.action_high.cpu().numpy().tolist(),
                'gamma': self.agent.gamma,
                'tau': self.agent.tau,
                'policy_noise': self.agent.policy_noise,
                'noise_clip': self.agent.noise_clip,
                'policy_freq': self.agent.policy_freq
            }
        }
        
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return model_dir

    def load_model(self, model_path):
        if os.path.isdir(model_path):
            self.agent.actor.load_state_dict(torch.load(os.path.join(model_path, 'actor.pth')))
            self.agent.critic.load_state_dict(torch.load(os.path.join(model_path, 'critic.pth')))
            self.agent.actor_target.load_state_dict(torch.load(os.path.join(model_path, 'actor_target.pth')))
            self.agent.critic_target.load_state_dict(torch.load(os.path.join(model_path, 'critic_target.pth')))
            
            if os.path.exists(os.path.join(model_path, 'actor_optimizer.pth')):
                self.agent.actor_optimizer.load_state_dict(torch.load(os.path.join(model_path, 'actor_optimizer.pth')))
            if os.path.exists(os.path.join(model_path, 'critic_optimizer.pth')):
                self.agent.critic_optimizer.load_state_dict(torch.load(os.path.join(model_path, 'critic_optimizer.pth')))
                
            print(f"Model loaded from: {model_path}")
        else:
            self.agent.actor.load_state_dict(torch.load(os.path.join(model_path, 'td3_actor.pth')))
            self.agent.critic.load_state_dict(torch.load(os.path.join(model_path, 'td3_critic.pth')))
            self.agent.actor_target.load_state_dict(torch.load(os.path.join(model_path, 'td3_actor_target.pth')))
            self.agent.critic_target.load_state_dict(torch.load(os.path.join(model_path, 'td3_critic_target.pth')))
            print(f"Legacy model loaded from: {model_path}")

    def cleanup_old_sessions(self, save_path='./SavedModels', keep_recent=2):
        if not os.path.exists(save_path):
            print("No SavedModels directory found!")
            return
        
        training_sessions = []
        for item in os.listdir(save_path):
            item_path = os.path.join(save_path, item)
            if os.path.isdir(item_path) and item.startswith('training_session_'):
                creation_time = os.path.getctime(item_path)
                training_sessions.append((item_path, creation_time))
        
        if len(training_sessions) <= keep_recent:
            print(f"Only {len(training_sessions)} training sessions found. No cleanup needed.")
            return
        
        training_sessions.sort(key=lambda x: x[1])
        sessions_to_remove = training_sessions[:-keep_recent]
        
        print(f"Cleaning up {len(sessions_to_remove)} old training sessions...")
        
        for session_path, creation_time in sessions_to_remove:
            try:
                shutil.rmtree(session_path)
                session_name = os.path.basename(session_path)
                print(f"   Removed: {session_name}")
            except Exception as e:
                print(f"   Failed to remove {session_path}: {e}")
        
        print(f"Cleanup complete! Kept {keep_recent} most recent sessions.")

    def list_saved_models(self, save_path='./SavedModels'):
        if not os.path.exists(save_path):
            print("No SavedModels directory found!")
            return
        
        print("Saved Models Overview:")
        print("="*60)
        
        total_size = 0
        
        for item in os.listdir(save_path):
            item_path = os.path.join(save_path, item)
            
            if os.path.isdir(item_path):
                dir_size = 0
                file_count = 0
                for root, dirs, files in os.walk(item_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        dir_size += os.path.getsize(file_path)
                        file_count += 1
                
                creation_time = datetime.fromtimestamp(os.path.getctime(item_path))
                
                if dir_size > 1024**3:
                    size_str = f"{dir_size / (1024**3):.2f} GB"
                elif dir_size > 1024**2:
                    size_str = f"{dir_size / (1024**2):.2f} MB"
                elif dir_size > 1024:
                    size_str = f"{dir_size / 1024:.2f} KB"
                else:
                    size_str = f"{dir_size} B"
                
                print(f"{item}")
                print(f"   Size: {size_str} ({file_count} files)")
                print(f"   Created: {creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print()
                
                total_size += dir_size
        
        if total_size > 1024**3:
            total_size_str = f"{total_size / (1024**3):.2f} GB"
        elif total_size > 1024**2:
            total_size_str = f"{total_size / (1024**2):.2f} MB"
        elif total_size > 1024:
            total_size_str = f"{total_size / 1024:.2f} KB"
        else:
            total_size_str = f"{total_size} B"
        
        print(f"Total disk usage: {total_size_str}")

    def save_best_model_only(self, save_path='./SavedModels', model_name='best_model'):
        if not self.performance_data:
            print("No performance data available. Run training first.")
            return
        
        episode_rewards = [sum(data['rewards']) for data in self.performance_data]
        best_episode_idx = np.argmax(episode_rewards)
        best_episode = self.performance_data[best_episode_idx]
        
        print(f"Saving best model from episode {best_episode['total_episode']} (reward: {episode_rewards[best_episode_idx]:.2f})")
        
        model_dir = self.save_model(save_path, model_name, overwrite=True)
        
        performance_file = os.path.join(model_dir, 'best_episode_performance.json')
        with open(performance_file, 'w') as f:
            json.dump(best_episode, f, indent=2)
        
        print(f"Best model saved to: {model_dir}")
        return model_dir

    def test_agent_on_new_system(self, num_tests=5, total_time=20.0, time_step=0.1):
        for test_num in range(num_tests):
            env = self.PIDSIM()
            state = env.reset()

            pid_gains = self.agent.select_action(state)
            print(f"Test {test_num+1}: PID gains selected: Kp={pid_gains[0]:.4f}, Ki={pid_gains[1]:.4f}, Kd={pid_gains[2]:.4f}")

            mass = state[2]
            friction = state[3]
            time_constant = state[4]
            print(f"System Parameters: Mass={mass:.2f}, Friction={friction:.2f}, Time Constant={time_constant:.2f}")

            states_over_time = []
            times = []

            steps = int(total_time / time_step)
            for step in range(steps):
                states_over_time.append(state[0])
                times.append(step * time_step)

                next_state, reward, done, info = env.step(pid_gains)
                state = next_state

                if done:
                    break

            plt.figure(figsize=(10, 6))
            plt.plot(times, states_over_time, label='Motor Speed')
            plt.axhline(y=env.desired_state, color='r', linestyle='--', label='Desired Speed')
            plt.xlabel('Time (s)')
            plt.ylabel('Motor Speed')
            plt.title(f'System Response - Test {test_num+1}')
            plt.legend()
            plt.grid(True)
            plt.show()

    def testWithoutNoise(self):
        env = self.PIDSIM()
        total_reward = 0
        state = env.reset()
        for step in range(200):
            action = self.agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break
        print(f"Total evaluation reward: {total_reward}")

    def hyperparameter_tuning(self, param_grid, num_trials=3, save_path='./SavedModels'):

        print(f"Starting hyperparameter tuning with {num_trials} trials per configuration")
        
        import itertools
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tuning_dir = os.path.join(save_path, f"hyperparameter_tuning_{timestamp}")
        os.makedirs(tuning_dir, exist_ok=True)
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        print(f"Testing {len(param_combinations)} parameter combinations")
        
        results = []
        best_avg_reward = float('-inf')
        best_config = None
        
        for i, combination in enumerate(param_combinations):
            config = dict(zip(param_names, combination))
            print(f"Testing configuration {i+1}/{len(param_combinations)}: {config}")
            
            trial_rewards = []
            valid_trials = 0
            
            for trial in range(num_trials):
                try:
                    trial_agent = self._create_agent_with_params(config)
                    trial_client = clientSide(self.PIDSIM, trial_agent)
                    
                    trial_reward = trial_client._quick_train(num_sessions=1, episodes_per_session=25)
                    
                    if trial_reward and not np.isnan(trial_reward):
                        trial_rewards.append(trial_reward)
                        valid_trials += 1
                    else:
                        print(f"   Trial {trial+1}: Invalid reward (NaN)")
                        
                except Exception as e:
                    print(f"   Trial {trial+1}: Error - {e}")
                    continue
            
            if valid_trials > 0:
                avg_reward = np.mean(trial_rewards)
                std_reward = np.std(trial_rewards)
                
                result = {
                    'config': config,
                    'avg_reward': avg_reward,
                    'std_reward': std_reward,
                    'valid_trials': valid_trials,
                    'trial_rewards': trial_rewards
                }
                results.append(result)
                
                print(f"   Average reward: {avg_reward:.2f} ± {std_reward:.2f} ({valid_trials}/{num_trials} valid trials)")
                
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_config = config
                    print(f"   New best configuration!")
            else:
                print(f"   No valid trials for this configuration")
        
        results.sort(key=lambda x: x['avg_reward'], reverse=True)
        
        tuning_results = {
            'best_config': best_config,
            'best_avg_reward': best_avg_reward,
            'all_results': results,
            'param_grid': param_grid,
            'num_trials': num_trials,
            'timestamp': timestamp
        }
        
        with open(os.path.join(tuning_dir, 'tuning_results.json'), 'w') as f:
            json.dump(tuning_results, f, indent=2, default=str)
        
        print(f"Tuning results saved to: {tuning_dir}")
        return best_config, tuning_results

    def _create_agent_with_params(self, config):

        from TD3.TD3Agent import TD3Agent
        import torch
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        action_low = np.array([0, 0, 0])
        action_high = np.array([10, 5, 5])
        state_shape = (5,)
        
        agent = TD3Agent(
            state_dim=5,
            action_dim=3,
            action_low=action_low,
            action_high=action_high,
            device=device,
            state_shape=state_shape
        )
        
        # Set hyperparameters after creation
        if 'gamma' in config:
            agent.gamma = config['gamma']
        if 'tau' in config:
            agent.tau = config['tau']
        if 'policy_noise' in config:
            agent.policy_noise = config['policy_noise']
        if 'noise_clip' in config:
            agent.noise_clip = config['noise_clip']
        if 'actor_lr' in config:
            agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=config['actor_lr'])
        if 'critic_lr' in config:
            agent.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=config['critic_lr'])
        
        return agent

    def _quick_train(self, num_sessions=1, episodes_per_session=25):

        total_reward = 0
        episode_count = 0
        
        for session in range(num_sessions):
            for episode in range(episodes_per_session):
                env = self.PIDSIM()
                state = env.reset()
                episode_reward = 0
                
                for step in range(100):
                    action = self.agent.select_action(state)
                    noise = np.random.normal(0, 0.1, size=action.shape)
                    action = np.clip(action + noise, self.agent.action_low.cpu().numpy(), self.agent.action_high.cpu().numpy())
                    
                    next_state, reward, done, info = env.step(action)
                    
                    self.agent.replay_buffer.append(state, action, reward, next_state, float(done))
                    self.agent.train(64)
                    
                    state = next_state
                    episode_reward += reward
                    
                    if done:
                        break
                
                total_reward += episode_reward
                episode_count += 1
        
        return total_reward / episode_count if episode_count > 0 else 0
