"""
Electrostatic potential and electric field in two different systems:

1. Point-charge configurations inside a square region
2. Parallel-plate capacitor with adjustable plate separation

The potential is obtained on a 2D grid and the electric field is
computed from the numerical gradient of the potential.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from time import time


# =============================================================================
# PHYSICAL CONSTANTS AND DEFAULT PARAMETERS
# =============================================================================

m = 100             # Number of grid intervals per side
k = 1               # Coulomb constant
target = 1e-4       # Precision threshold
omega = 0.9         # Relaxation parameter
q = 10              # Charge magnitude

x_grid = np.linspace(0, 100, m + 1)
y_grid = np.linspace(0, 100, m + 1)

x_cm = np.linspace(0, 10, m + 1)
y_cm = np.linspace(0, 10, m + 1)

deltax = 0.1
deltay = 0.1


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def point_potential(r, rq, charge):
    """
    Compute the electrostatic potential of a point charge.
    """
    return k * charge / norm(r - rq)


def electric_field(phi, fixed_points=None):
    """
    Compute the electric field and its magnitude from the potential.
    """
    ex = np.zeros((m + 1, m + 1), float)
    ey = np.zeros((m + 1, m + 1), float)
    e_mod = np.zeros((m + 1, m + 1), float)

    if fixed_points is None:
        fixed_points = []

    for i in range(m + 1):
        for j in range(m + 1):
            if i == 0 or i == m or j == 0 or j == m:
                e_mod[i, j] = e_mod[i, j]

            elif [i, j] in fixed_points:
                e_mod[i, j] = e_mod[i, j]

            else:
                ex[i, j] = -(phi[i + 1, j] - phi[i - 1, j]) / (2 * deltax)
                ey[i, j] = -(phi[i, j + 1] - phi[i, j - 1]) / (2 * deltay)
                e_mod[i, j] = np.sqrt(ex[i, j]**2 + ey[i, j]**2)

    return ex, ey, e_mod


# =============================================================================
# PLOTTING
# =============================================================================

def plot_potential(phi, title_text):
    """
    Plot the electrostatic potential.
    """
    plt.figure()
    plt.imshow(phi, extent=[x_cm.min(), x_cm.max(), y_cm.min(), y_cm.max()])
    plt.colorbar(label="Potential")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.title(title_text)


def plot_field_magnitude(e_mod, title_text):
    """
    Plot the magnitude of the electric field.
    """
    plt.figure()
    plt.imshow(e_mod, extent=[x_cm.min(), x_cm.max(), y_cm.min(), y_cm.max()])
    plt.colorbar(label="E")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.title(title_text)


def plot_field_lines(ex, ey, e_mod, title_text, cmap_name="jet"):
    """
    Plot the electric field lines using a streamplot.
    """
    r = np.meshgrid(x_cm, y_cm)

    fig, ax = plt.subplots()
    strm = ax.streamplot(r[0], r[1], ey, ex, color=e_mod, cmap=getattr(plt.cm, cmap_name), density=1.5)
    fig.colorbar(strm.lines)
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.title(title_text)



# =============================================================================
# POINT-CHARGE CONFIGURATIONS
# =============================================================================

def charge_configuration(case):
    """
    Return the list of charges for the selected configuration.
    Each element has the form: (position, charge).
    """
    if case == 1:
        return [
            ([30, 30],  q),
            ([30, 70],  q),
            ([70, 30],  q),
            ([70, 70],  q),
            ([50, 20], -q),
            ([50, 80], -q),
            ([20, 50], -q),
            ([80, 50], -q),
        ]

    elif case == 2:
        return [
            ([30, 30], q),
            ([30, 70], q),
            ([70, 30], q),
            ([70, 70], q),
        ]

    else:
        raise ValueError("Invalid charge configuration.")


def charge_potential(case):
    """
    Compute the initial potential produced by the selected set of point charges.
    """
    phi = np.zeros((m + 1, m + 1), float)
    charges = charge_configuration(case)

    for rq, charge in charges:
        rq = np.array(rq, dtype=float)

        for i in range(m + 1):
            for j in range(m + 1):
                if i != rq[0] and j != rq[1]:
                    r = np.array([x_grid[i], y_grid[j]])
                    phi[i, j] += point_potential(r, rq, charge)

    return phi, charges


def solve_potential_charges(phi, charge_positions):
    """
    Relax the potential on the grid keeping the charge points fixed.
    """
    delta = 1.0

    while delta > target:
        delta = 0.0

        for i in range(m + 1):
            for j in range(m + 1):
                if i == 0 or i == m or j == 0 or j == m:
                    phi[i, j] = 0

                elif [i, j] in charge_positions:
                    phi[i, j] = phi[i, j]

                else:
                    phi_old = phi[i, j]
                    phi[i, j] = (
                        (1 + omega) * (
                            phi[i + 1, j] +
                            phi[i - 1, j] +
                            phi[i, j + 1] +
                            phi[i, j - 1]
                        ) / 4
                        - omega * phi[i, j]
                    )
                    delta += abs(phi_old - phi[i, j])

        delta = delta / (m + 1)**2
        print("delta:", delta)

    return phi


def run_charges(case):
    """
    Solve the electrostatic problem for two different point-charge configurations.
    """
    start = time()

    phi, charges = charge_potential(case)
    charge_positions = [list(map(int, rq)) for rq, _ in charges]

    phi = solve_potential_charges(phi, charge_positions)
    ex, ey, e_mod = electric_field(phi, fixed_points=charge_positions)

    elapsed = time() - start
    print("Elapsed time:", elapsed, "seconds")
    print("Maximum electric field:", np.max(e_mod))

    if case == 1:
        plot_potential(phi, "Potential for 8 point charges")
        plot_field_lines(ex, ey, e_mod, "Electric field lines for 8 point charges", cmap_name="jet")
    elif case == 2:
        plot_potential(phi, "Potential for 4 positive charges")
        plot_field_lines(ex, ey, e_mod, "Electric field lines for 4 positive charges", cmap_name="jet")
    else:
        raise ValueError("Invalid charge configuration.")

    plt.show()


# =============================================================================
# PARALLEL-PLATE CAPACITOR
# =============================================================================

def capacitor_potential(separation):
    """
    Create the initial potential for a parallel-plate capacitor.
    """
    phi = np.zeros((m + 1, m + 1), float)

    sep_grid = separation * 10
    a = int((100 - sep_grid) / 2)

    phi[20:80, a] = 1
    phi[20:80, 100 - a] = -1

    return phi, a


def solve_potential_capacitor(phi, a):
    """
    Relax the potential for the capacitor geometry keeping the plates fixed.
    """
    delta = 1.0

    while delta > target:
        delta = 0.0

        for i in range(m + 1):
            for j in range(m + 1):
                if i == 0 or i == m or j == 0 or j == m:
                    phi[i, j] = phi[i, j]

                elif (j == a or j == 100 - a) and (20 < i < 80):
                    phi[i, j] = phi[i, j]

                else:
                    phi_old = phi[i, j]
                    phi[i, j] = (
                        (1 + omega) * (
                            phi[i + 1, j] +
                            phi[i - 1, j] +
                            phi[i, j + 1] +
                            phi[i, j - 1]
                        ) / 4
                        - omega * phi[i, j]
                    )
                    delta += abs(phi_old - phi[i, j])

        delta = delta / (m + 1)**2
        print("delta:", delta)

    return phi


def run_capacitor(sep):
    """
    Solve the electrostatic problem for a parallel-plate capacitor.
    """
    if sep < 1 or sep > 9:
        raise ValueError("Invalid separation. Choose an integer between 1 and 9 cm.")

    start = time()

    phi, a = capacitor_potential(sep)
    phi = solve_potential_capacitor(phi, a)

    fixed_points = []
    for i in range(20, 81):
        fixed_points.append([i, a])
        fixed_points.append([i, 100 - a])

    ex, ey, e_mod = electric_field(phi, fixed_points=fixed_points)

    elapsed = time() - start
    print("Elapsed time:", elapsed, "seconds")

    plot_potential(phi, "Potential inside the capacitor")
    plot_field_magnitude(e_mod, "Electric field magnitude")
    plot_field_lines(ex, ey, e_mod, "Electric field lines inside the capacitor", cmap_name="winter")

    plt.show()



# =============================================================================
# MAIN
# =============================================================================

def main(mode="default"):
    cases = {
        1: "Point-charge configurations",
        2: "Parallel-plate capacitor"
    }

    if mode == "interactive":
        print("Available electrostatic problems:")
        print("1: Point-charge configurations")
        print("2: Parallel-plate capacitor")

        case = int(input("Enter the case number (1 or 2): "))

        if case == 1:
            print("\nRunning case 1: Point-charge configurations\n")
            subcase = int(input("Enter 1 for 8 charges or 2 for 4 positive charges: "))
            run_charges(subcase)

        elif case == 2:
            print("\nRunning case 2: Parallel-plate capacitor\n")
            sep = int(input("Enter the plate separation between 1 and 9 cm: "))
            run_capacitor(sep)

        else:
            raise ValueError("Invalid case selection.")

    else:
        case = 1
        description = cases[case]
        print(f"\nRunning case {case}: {description}\n")
        print("Use interactive mode to explore other cases.\n")

        subcase = 1
        run_charges(subcase)


if __name__ == "__main__":
    main()