import matplotlib.pyplot as plt

# Possible outcomes for each die
dice_outcomes = list(range(1, 7))

# Calculate the sum of two dice
sum_outcomes = [a + b for a in dice_outcomes for b in dice_outcomes]

# Calculate the probability for each sum
probabilities = [sum_outcomes.count(i) / 36 for i in range(2, 13)]

# Calculate E[S] (expected value)
E_S = sum([(i) * probabilities[i - 2] for i in range(2, 13)])

# Calculate Var(S) (variance)
Var_S = sum([((i + 2) - E_S) ** 2 * probabilities[i - 2] for i in range(2, 13)])

# Visualize the probability distribution
plt.bar(range(2, 13), probabilities, tick_label=range(2, 13))
plt.xlabel('Sum of Two Dice')
plt.ylabel('Probability')
plt.title('Probability Distribution of the Sum of Two Dice')
plt.grid(axis='y', linestyle='--', alpha=0.7)


print(f'E[S] = {E_S}')
print(f'Var(S) = {Var_S}')
plt.show()

