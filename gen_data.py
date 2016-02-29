import random
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Set up the data generation parameters.
NUM_POINTS = 25
MIN_X = -50
MAX_X = 50
X_SIGMA = 25
MIN_Y = -50
MAX_Y = 50
Y_SIGMA = 25

# Set up the 3 classes we can generate data for.
COLORS = [
    "red",
    "blue",
    "yellow"
]

# Here is where the magic happens!
def main():
    # Create data points for the red class.
    red_points = random.randrange(NUM_POINTS)

    x_r = np.random.normal(np.random.randint(MIN_X, high = MAX_X),
                           np.random.randint(X_SIGMA) + 1,
                           red_points)

    y_r = np.random.normal(np.random.randint(MIN_Y, high = MAX_Y),
                           np.random.randint(Y_SIGMA) + 1,
                           red_points)

    c_r = ["red" for c in xrange(red_points)]

    # Ditto for the blue class.
    blue_points = random.randrange(NUM_POINTS - red_points)

    x_b = np.random.normal(np.random.randint(MIN_X, high = MAX_X),
                           np.random.randint(X_SIGMA) + 1,
                           blue_points)

    y_b = np.random.normal(np.random.randint(MIN_Y, high = MAX_Y),
                           np.random.randint(Y_SIGMA) + 1,
                           blue_points)

    c_b = ["blue" for c in xrange(blue_points)]

    # And for the yellow class.
    yellow_points = (NUM_POINTS - red_points) - blue_points

    x_y = np.random.normal(np.random.randint(MIN_X, high = MAX_X),
                           np.random.randint(X_SIGMA) + 1,
                           yellow_points)

    y_y = np.random.normal(np.random.randint(MIN_Y, high = MAX_Y),
                           np.random.randint(Y_SIGMA) + 1,
                           yellow_points)

    c_y = ["yellow" for c in xrange(yellow_points)]

    # Append all the data into numpy arrays.
    x = np.concatenate((x_r, x_b, x_y))
    y = np.concatenate((y_r, y_b, y_y))
    c = np.concatenate((c_r, c_b, c_y))

    # Plot the data if able.
    if plt is not None:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.scatter(x, y, c = c)
        ax1.set_aspect(1.0 / ax1.get_data_ratio())
        plt.show()

    # Save the data to file.
    data = np.transpose(np.array([x, y, c]))
    np.savetxt("data.csv", data, fmt = "%s,%s,%s,")

# Entry point
if __name__ == "__main__":
    main()
