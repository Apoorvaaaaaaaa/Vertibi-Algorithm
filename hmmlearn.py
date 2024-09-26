def viterbi(observations, states, start_prob, trans_prob, emit_prob):
    """
    Viterbi algorithm to find the most probable sequence of states given a sequence of observations.

    Parameters:
    observations : tuple
        A sequence of observed events.
    states : tuple
        A sequence of possible states.
    start_prob : dict
        The initial probabilities for each state.
    trans_prob : dict
        The transition probabilities from one state to another.
    emit_prob : dict
        The emission probabilities for each state to each observation.

    Returns:
    max_prob : float
        The highest probability of the best state sequence.
    best_path : list
        The most probable sequence of states.
    """
    
    # Validate input parameters
    validate_input(observations, states, start_prob, trans_prob, emit_prob)

    # Initialize the Viterbi matrix and the path tracker
    V = [{}]  # List of dictionaries to store the probabilities of the states at each time step
    path = {}  # Dictionary to store the best path leading to each state

    # Initialize the base cases (t == 0)
    print("Initialization:")
    for state in states:
        # Probability of starting in state and emitting the first observation
        V[0][state] = start_prob[state] * emit_prob[state][observations[0]]
        
        # Initialize path for each state as starting with itself
        path[state] = [state]
        print(f"State: {state}, Probability: {V[0][state]}, Path: {path[state]}")

    # Run Viterbi for t > 0 (for each observation after the first)
    for t in range(1, len(observations)):
        V.append({})  # Add a new layer for time step t
        new_path = {}  # Temporary dictionary to store new paths

        print(f"\nTime step {t}: Observation = {observations[t]}")
        # Loop through each state for the current time step
        for current_state in states:
            # Compute the maximum probability and the previous state that gives that probability
            max_prob, prev_st = max(
                (V[t - 1][prev_state] * trans_prob[prev_state][current_state] * emit_prob[current_state][observations[t]], prev_state)
                for prev_state in states
            )

            # Set the max probability for the current state at time t
            V[t][current_state] = max_prob

            # Update the new path by adding the current state to the best path from the previous state
            new_path[current_state] = path[prev_st] + [current_state]
            print(f"Current State: {current_state}, Max Probability: {max_prob}, Prev State: {prev_st}, New Path: {new_path[current_state]}")

        # Replace the old paths with the new ones for the next iteration
        path = new_path

    # Find the final most probable state and its probability
    max_prob, final_state = max((V[-1][state], state) for state in states)

    print("\nFinal probabilities at last time step:")
    for state in states:
        print(f"State: {state}, Probability: {V[-1][state]}")
    
    print(f"\nMost probable final state: {final_state} with probability {max_prob}")
    
    # Return the highest probability and the best path (most probable sequence of states)
    return max_prob, path[final_state]


def validate_input(observations, states, start_prob, trans_prob, emit_prob):
    """
    Validate the inputs to the Viterbi algorithm to ensure correctness.

    Parameters:
    observations : tuple
        The sequence of observations.
    states : tuple
        The sequence of possible states.
    start_prob : dict
        The starting probability for each state.
    trans_prob : dict
        The state transition probabilities.
    emit_prob : dict
        The emission probabilities for each observation from each state.
    """

    assert isinstance(observations, (list, tuple)), "Observations must be a list or tuple."
    assert isinstance(states, (list, tuple)), "States must be a list or tuple."
    assert set(start_prob.keys()) == set(states), "Start probabilities must match states."
    assert set(trans_prob.keys()) == set(states), "Transition probabilities must match states."
    assert all(set(trans_prob[s].keys()) == set(states) for s in states), "Transition probabilities must be defined for all states."
    assert set(emit_prob.keys()) == set(states), "Emission probabilities must match states."
    assert all(set(emit_prob[s].keys()) == set(observations) for s in states), "Emission probabilities must be defined for all observations."

    print("Input validation passed.\n")


# Example usage
observations = ('walk', 'shop', 'clean')
states = ('Rainy', 'Sunny')
start_prob = {'Rainy': 0.6, 'Sunny': 0.4}  # Probability of starting in each state
trans_prob = {
    'Rainy': {'Rainy': 0.7, 'Sunny': 0.3},  # Transition probabilities from Rainy to other states
    'Sunny': {'Rainy': 0.4, 'Sunny': 0.6},  # Transition probabilities from Sunny to other states
}
emit_prob = {
    'Rainy': {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},  # Emission probabilities for Rainy state
    'Sunny': {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},  # Emission probabilities for Sunny state
}

# Run the Viterbi algorithm
print("Running Viterbi algorithm...\n")
max_prob, best_path = viterbi(observations, states, start_prob, trans_prob, emit_prob)

# Output the results
print("\nFinal Results:")
print(f"Maximum Probability: {max_prob}")
print(f"Best Path: {best_path}")




