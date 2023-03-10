def extract_neural_net_dims(hyperparameters):
    hyperparameters.actor_hidden_dims = \
        tuple([int(string) for string in hyperparameters.actor_hidden_dims.split(' ')])
    hyperparameters.critic_hidden_dims = \
        tuple([int(string) for string in hyperparameters.critic_hidden_dims.split(' ')])
    return hyperparameters