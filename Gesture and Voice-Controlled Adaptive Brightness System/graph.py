# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # # Set Seaborn style
# # sns.set_style("darkgrid")
# # sns.set_context("talk")  # Makes labels bigger

# # # Sample Data: Distance between thumb & index finger (in cm)
# # distance = np.linspace(0, 10, 10)  # Distance from 0 cm to 10 cm
# # expected_brightness = np.linspace(0, 100, 10)  # Expected Brightness (0% to 100%)
# # actual_brightness = expected_brightness + np.random.normal(0, 5, 10)  # Adding some noise

# # # Create figure
# # plt.figure(figsize=(10, 6))

# # # Plot expected and actual brightness
# # sns.lineplot(x=distance, y=expected_brightness, marker='o', label="Expected Brightness", linewidth=2.5, color="blue")
# # sns.lineplot(x=distance, y=actual_brightness, marker='s', label="Actual Brightness", linewidth=2.5, color="red")

# # # Titles and Labels
# # plt.title("Brightness vs. Distance Between Thumb and Index Finger", fontsize=16, fontweight='bold')
# # plt.xlabel("Distance Between Thumb & Index (cm)", fontsize=14)
# # plt.ylabel("Brightness Percentage (%)", fontsize=14)
# # plt.xticks(fontsize=12)
# # plt.yticks(fontsize=12)

# # # Add legend
# # plt.legend(loc="lower right", fontsize=12)

# # # Show grid
# # plt.grid(True, linestyle="--", alpha=0.6)

# # # Show the plot
# # plt.show()
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # # Set Seaborn style
# # sns.set_style("darkgrid")
# # sns.set_context("talk")

# # # Updated Hand Gestures and Brightness Levels
# # gestures = ["Fist", "Thumbs Up", "Palm (5 Fingers)"]
# # expected_brightness = [0, 50, 90]  # Expected brightness for each gesture
# # actual_brightness = [3, 48, 87]  # Actual brightness with slight variation

# # # X-axis positions
# # x_pos = np.arange(len(gestures))

# # # Create figure
# # plt.figure(figsize=(8, 5))

# # # Plot expected and actual brightness
# # sns.lineplot(x=x_pos, y=expected_brightness, marker='o', label="Expected Brightness", linewidth=2.5, color="blue")
# # sns.lineplot(x=x_pos, y=actual_brightness, marker='s', label="Actual Brightness", linewidth=2.5, color="red")

# # # Titles and Labels
# # plt.title("Brightness vs. Hand Gesture", fontsize=16, fontweight='bold')
# # plt.xlabel("Hand Gesture", fontsize=14)
# # plt.ylabel("Brightness Percentage (%)", fontsize=14)
# # plt.xticks(ticks=x_pos, labels=gestures, fontsize=12)
# # plt.yticks(fontsize=12)

# # # Add legend
# # plt.legend(loc="upper left", fontsize=12)

# # # Show grid
# # plt.grid(True, linestyle="--", alpha=0.6)

# # # Show the plot
# # plt.show()
# import numpy as np
# import matplotlib.pyplot as plt

# ambient_light = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# adjusted_brightness = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95, 100])

# plt.figure(figsize=(7, 5))
# plt.scatter(ambient_light, adjusted_brightness, color='green', label="Adjusted Brightness")
# plt.plot(ambient_light, ambient_light, linestyle="--", color="red", label="Ideal Adjustment")

# plt.xlabel("Ambient Light Intensity (%)")
# plt.ylabel("Adjusted Brightness (%)")
# plt.title("Ambient Light vs. Adjusted Brightness")
# plt.legend()
# plt.grid(True)
# plt.show()
import matplotlib.pyplot as plt
import seaborn as sns



ambient_light = [0, 50, 100, 150, 200, 255]
expected = [0, 20, 40, 60, 80, 100]
actual = [0, 21, 39, 58, 78, 97]

plt.figure(figsize=(8,5))
sns.lineplot(x=ambient_light, y=expected, label="Expected", linewidth=2)
sns.lineplot(x=ambient_light, y=actual, label="Actual", linewidth=2, linestyle="--")
plt.xlabel("Ambient Light Intensity (Grayscale Value)")
plt.ylabel("Brightness (%)")
plt.title("Ambient Light vs. Adaptive Brightness")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



