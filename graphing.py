import matplotlib.pyplot as plt

# Generation data
generations = list(range(11))
mean_values = [-109497, -65300.1, -49633.2, -40887.7, -29688.3, -28870.1, -25901.3, -25391.8, -23898.3, -22491.3, -19584.2]
max_values = [-23414.2, -20821.6, -17441, -17441, -13237.8, -13237.8, -13237.8, -13237.8, -13237.8, -13237.8, -13046]
min_values = [-297379, -224199, -85655, -195391, -107024, -99301, -85655, -85655, -85655, -78710.2, -85655]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(generations, mean_values, label='Mean', marker='o')
plt.plot(generations, max_values, label='Max (Best Fitness)', marker='o')
plt.plot(generations, min_values, label='Min (Worst Fitness)', marker='o')

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness Evolution Over 10 Generations")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("fitness_evolution_10.png", dpi=300)
plt.show()