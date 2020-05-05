from Agent import *
import matplotlib
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
matplotlib.rc('axes', labelsize=15)
plt.rc('legend', fontsize=12)


class HyperparameterTuner:

    def __init__(self):
        self.gamma = 0.999
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.0005
        self.target_update = 4
        self.max_iter = 1000
        self.batch_size = 64
        self.tau = 0.001
        self.dropout_ratio = 0
        self.colors = ["b", "r", "g", "y"]

    def moving_average(self, a, n=30):
        a = np.array(a)
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def gamma_analysis(self):
        gammas = [0.99, 0.999, 0.9]
        filename = "results/gamma.png"
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        for idx, gamma in enumerate(gammas):
            agent = Agent(gamma, self.epsilon_start, self.epsilon_end, self.epsilon_decay, self.alpha, self.target_update, self.max_iter, self.tau, self.batch_size, self.dropout_ratio)
            if agent.reward_exist():
                rewards = agent.load_rewards()
                print("load value for {}".format(agent.tag))
            else:
                rewards = agent.train()
            rewards = self.moving_average(rewards)
            epochs = [(i + 1) for i in range(len(rewards))]
            plt.plot(epochs, rewards, color=self.colors[idx], linestyle='-')
        plt.xlabel("Epochs")
        plt.ylabel("Rewards")
        plt.xlim(0, self.max_iter)
        plt.ylim(-300, 300)
        plt.legend(gammas, loc='best')
        fig.savefig(filename, dpi=fig.dpi)
        return

    def alpha_analysis(self):
        alphas = [0.0005, 0.001, 0.01]
        filename = "results/alpha.png"
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        for idx, alpha in enumerate(alphas):
            agent = Agent(self.gamma, self.epsilon_start, self.epsilon_end, self.epsilon_decay, alpha,
                          self.target_update, self.max_iter, self.tau, self.batch_size, self.dropout_ratio)
            if agent.reward_exist():
                rewards = agent.load_rewards()
                print("load value for {}".format(agent.tag))
            else:
                rewards = agent.train()
            rewards = self.moving_average(rewards)
            epochs = [(i + 1) for i in range(len(rewards))]
            plt.plot(epochs, rewards, color=self.colors[idx], linestyle='-')
        plt.xlabel("Epochs")
        plt.ylabel("Rewards")
        plt.xlim(0, self.max_iter)
        plt.ylim(-400, 300)
        plt.legend(alphas, loc='best')
        fig.savefig(filename, dpi=fig.dpi)
        return

    def epsilon_decay_analysis(self):
        epsilon_decays = [0.999, 0.995, 0.99, 0.9]
        filename = "results/epsilon_decay.png"
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        for idx, epsilon_decay in enumerate(epsilon_decays):
            agent = Agent(self.gamma, self.epsilon_start, self.epsilon_end, epsilon_decay, self.alpha,
                          self.target_update, self.max_iter, self.tau, self.batch_size, self.dropout_ratio)
            if agent.reward_exist():
                rewards = agent.load_rewards()
                print("load value for {}".format(agent.tag))
            else:
                rewards = agent.train()
            rewards = self.moving_average(rewards)
            epochs = [(i + 1) for i in range(len(rewards))]
            plt.plot(epochs, rewards, color=self.colors[idx], linestyle='-')
        plt.xlabel("Epochs")
        plt.ylabel("Rewards")
        plt.xlim(0, self.max_iter)
        plt.ylim(-400, 300)
        plt.legend(epsilon_decays, loc='best')
        fig.savefig(filename, dpi=fig.dpi)
        return

    def target_update_analysis(self):
        target_updates = [(4, 0.001), (30, 1), (8, 0.01), (15, 0.1)]
        target_updates_str = ["t" + str(a[0]) + "_tau" + str(a[1]) for a in target_updates]
        filename = "results/target_update_tau.png"
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        for idx, target_update_tau in enumerate(target_updates):
            target_update, tau = target_update_tau
            agent = Agent(self.gamma, self.epsilon_start, self.epsilon_end, self.epsilon_decay, self.alpha,
                          target_update, self.max_iter, tau, self.batch_size, self.dropout_ratio)
            if agent.reward_exist():
                rewards = agent.load_rewards()
                print("load value for {}".format(agent.tag))
            else:
                rewards = agent.train()
            rewards = self.moving_average(rewards)
            epochs = [(i + 1) for i in range(len(rewards))]
            plt.plot(epochs, rewards, color=self.colors[idx], linestyle='-')
        plt.xlabel("Epochs")
        plt.ylabel("Rewards")
        plt.xlim(0, self.max_iter)
        plt.ylim(-1200, 300)
        plt.legend(target_updates_str, loc='best')
        fig.savefig(filename, dpi=fig.dpi)
        return


if __name__ == "__main__":

    # Base model train

    # gamma = 0.99
    # epsilon_start = 1.0
    # epsilon_end = 0.01
    # epsilon_decay = 0.995
    # alpha = 0.0005
    # target_update = 20
    # max_iter = 2000
    # batch_size = 64
    # tau = 0.001
    # dropout_ratio = 0
    #
    # agent = Agent(gamma, epsilon_start, epsilon_end, epsilon_decay, alpha, target_update, max_iter, tau, batch_size, dropout_ratio)
    # agent.train()
    # agent.load_model()
    # agent.test()

    # Hyperparameter
    tuner = HyperparameterTuner()
    # tuner.gamma_analysis()
    # tuner.alpha_analysis()
    # tuner.epsilon_decay_analysis()
    tuner.target_update_analysis()

