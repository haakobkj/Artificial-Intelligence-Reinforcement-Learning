    def __init__(self, environment: Environment):
        self.environment = environment
        #
        # TODO: (optional) Define any class instance variables you require (e.g. Q-value tables) here.
        #
        self.n_episode = 2
        self.max_timesteps_per_episode = 100

        self.observation_space = [self.environment.get_init_state()]
        self.action_space = ROBOT_ACTIONS
        
        self.Q = np.random.rand(len(self.observation_space), len(self.action_space))
        self.Q[:, :] = np.zeros(len(self.action_space)) 

    # === Q-learning ===================================================================================================

    def q_learn_train(self):
        """
        Train this RL agent via Q-Learning.
        """
        self.action_space.seed(12345) # To get a determenistic random number generator
        #self.reset(seed=12345)

        n_episode = 2 #
        max_timesteps_per_episode = 100 # Prevents from going on forever

        for ep_idx in range(n_episode): #Loop for every episode (similar to Q-learning)
            s, _ = env.reset()
            for t in range(max_timesteps_per_episode): #Loop for envery step in every episode
                action = env.action_space.sample() #Gives us sample from action space
                next_state, reward, terminated, _, _ = env.step(action)
                
                if terminated:
                    break