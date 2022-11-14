def extract_neural_net_dims(hyperparameters):
    hyperparameters.actor_network = \
        tuple([int(string) for string in hyperparameters.actor_network.split(' ')])
    hyperparameters.critic_network = \
        tuple([int(string) for string in hyperparameters.critic_network.split(' ')])
    return hyperparameters