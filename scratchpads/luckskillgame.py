import numpy as np

# Constants for the simulation
NUM_APPLICANTS = 10000
NUM_SELECTED = 11
SKILL_FACTOR = 0.95
LUCK_FACTOR = 0.05
NUM_EXPERIMENTS = 1000


def run_selection_simulation():
    """
    Simulates an astronaut selection process to determine the impact of luck.
    """
    # Generate fixed skill scores for all applicants (0 to 1)
    # These remain constant across all experiments
    # skill_scores = np.random.rand(NUM_APPLICANTS)
    luck_scores = np.random.rand(NUM_APPLICANTS)

    # Applicant IDs (0 to NUM_APPLICANTS-1)
    applicant_ids = np.arange(NUM_APPLICANTS)

    # List to store sets of selected applicant IDs for each experiment
    all_selections_ids = []

    for experiment_num in range(NUM_EXPERIMENTS):
        # Generate random luck scores for this experiment (0 to 1)
        # luck_scores = np.random.rand(NUM_APPLICANTS)
        skill_scores = np.random.rand(NUM_APPLICANTS)

        # Calculate combined scores
        combined_scores = (
            skill_scores * SKILL_FACTOR) + (luck_scores * LUCK_FACTOR)

        # Select top NUM_SELECTED applicants
        # np.argpartition is efficient for finding k largest/smallest elements
        # We get the indices of the NUM_SELECTED applicants with the highest
        # combined scores
        selected_indices = np.argpartition(
            combined_scores, -NUM_SELECTED)[-NUM_SELECTED:]

        # Store the set of selected applicant IDs for this experiment
        selected_ids_this_experiment = set(applicant_ids[selected_indices])
        all_selections_ids.append(selected_ids_this_experiment)

    if NUM_EXPERIMENTS == 0:
        print("No experiments were run.")
        return

    if not all_selections_ids:
        print("No selections were made (e.g., NUM_EXPERIMENTS might be 0 or NUM_SELECTED is 0).")
        return

    # Use the first experiment's selection as the baseline
    baseline_selection = all_selections_ids[0]
    total_changed_astronauts = 0

    # Compare subsequent experiments to the baseline
    # (NUM_EXPERIMENTS - 1) comparisons will be made
    if NUM_EXPERIMENTS > 1:
        for i in range(1, NUM_EXPERIMENTS):
            current_selection = all_selections_ids[i]
            # Count how many astronauts in the current selection were NOT in the baseline selection
            # This also equals how many from baseline are no longer in current selection,
            # as the total number selected is constant.
            differences = len(current_selection - baseline_selection)
            total_changed_astronauts += differences

        average_changed_astronauts = total_changed_astronauts / \
            (NUM_EXPERIMENTS - 1)
    else:
        # If only one experiment, no changes to compare against baseline
        average_changed_astronauts = 0

    print(f"Astronaut Selection Simulation Results:")
    print(f"---------------------------------------")
    print(f"Number of applicants: {NUM_APPLICANTS}")
    print(f"Number selected each round: {NUM_SELECTED}")
    print(f"Skill factor: {SKILL_FACTOR * 100:.1f}%")
    print(f"Luck factor: {LUCK_FACTOR * 100:.1f}%")
    print(f"Total experiments run: {NUM_EXPERIMENTS}")
    print(f"---------------------------------------")
    if NUM_EXPERIMENTS > 1:
        print(
            f"On average, {
                average_changed_astronauts:.2f} of the {NUM_SELECTED} selected astronauts were different")
        print(f"in subsequent experiments due to luck, when compared to the outcome of the first experiment.")
    elif NUM_EXPERIMENTS == 1:
        print(
            f"Only one experiment was run. The selected group is: {baseline_selection}")
        print(
            f"Cannot calculate average changes as there are no subsequent experiments to compare.")
    else:  # NUM_EXPERIMENTS == 0, already handled but as a fallback
        print(f"No experiments run to determine changes.")


if __name__ == "__main__":
    # For reproducibility during testing, one might uncomment the next line
    # np.random.seed(42)
    run_selection_simulation()
