import sympy as sp
import math
import matplotlib.pyplot as plt


def solve_galerkin():
    """Выполняет аппроксимацию методом Галеркина (аналитически)."""
    x = sp.Symbol('x')
    C1, C2, C3, C4, C5 = sp.symbols('C1 C2 C3 C4 C5')

    phi0 = x
    phi1 = x**2 - 2 * x
    phi2 = x**3 - 3 * x
    phi3 = x**4 - 4 * x
    phi4 = x**5 - 5 * x
    phi5 = x**6 - 6 * x

    # phi0 = x
    # phi1 = sin(pi*x/2)
    # phi2 = sin(3*pi*x/2)
    # phi3 = sin(5*pi*x/2)

    print(f"  Используем базис:")
    print(f"  phi0 = {phi0}")
    print(f"  phi1 = {phi1}")
    print(f"  phi2 = {phi2}")
    print(f"  phi3 = {phi3}")
    print(f"  phi4 = {phi4}")
    print(f"  phi5 = {phi5}")

    u_approx = phi0 + C1 * phi1 + C2 * phi2 + C3 * phi3 + C4 * phi4 + C5 * phi5

    u_p = sp.diff(u_approx, x)
    u_pp = sp.diff(u_p, x)

    R = u_pp - 2 * x * u_p + 2 * u_approx - x

    eq1 = sp.integrate(R * phi1, (x, 0, 1))
    eq2 = sp.integrate(R * phi2, (x, 0, 1))
    eq3 = sp.integrate(R * phi3, (x, 0, 1))
    eq4 = sp.integrate(R * phi4, (x, 0, 1))
    eq5 = sp.integrate(R * phi5, (x, 0, 1))

    solution = sp.solve((eq1, eq2, eq3, eq4, eq5), (C1, C2, C3, C4, C5))

    u_final_expr = u_approx.subs(solution)

    u_func = sp.lambdify(x, u_final_expr, 'math')

    return u_func, u_final_expr


def solve_fdm(N=10000):
    """Решает задачу методом прогонки (конечные разности)."""
    h = 1.0 / N
    x_nodes = [i * h for i in range(N + 1)]

    alpha = [0.0] * (N + 1)
    beta = [0.0] * (N + 1)

    alpha[1] = 0.0
    beta[1] = 0.0

    for i in range(1, N):
        xi = x_nodes[i]

        A_i = 1.0 + xi * h
        C_i = 2.0 - 2.0 * (h**2)
        B_i = 1.0 - xi * h
        F_i = -xi * (h**2)

        denom = C_i - A_i * alpha[i]
        alpha[i + 1] = B_i / denom
        beta[i + 1] = (F_i + A_i * beta[i]) / denom

    u_fdm = [0.0] * (N + 1)

    u_fdm[N] = (h + beta[N]) / (1.0 - alpha[N])

    for i in range(N - 1, -1, -1):
        u_fdm[i] = alpha[i + 1] * u_fdm[i + 1] + beta[i + 1]

    return x_nodes, u_fdm


def print_res(x_nodes, u_gal, u_fdm, num_rows=10):
    """Выводит таблицу для сравнения значений."""
    print(
        f"\n{'x':<5} | {'Галеркин':<22} | {'Прогонка':<20} | {'Разница':<10}")
    print("-" * 70)

    step = len(x_nodes) // num_rows
    for i in range(0, len(x_nodes), step):
        x = x_nodes[i]
        g = u_gal[i]
        f = u_fdm[i]
        print(f"{x:<5.1f} | {g:<22.6f} | {f:<20.6f} | {(abs(g - f)):.2e}")


def plot(x_nodes, u_gal, u_fdm):
    """Отрисовка графиков."""
    plt.figure(figsize=(10, 6))

    plt.plot(x_nodes,
             u_gal,
             label='Метод Галеркина',
             color='blue',
             linewidth=2)

    plt.plot(x_nodes[::20],
             u_fdm[::20],
             'ro',
             label='Метод прогонки (узлы)',
             markersize=4)

    plt.title('Сравнение методов: Галеркин vs Конечные разности')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.show()


def main():
    print("Решение краевой задачи: u'' - 2x*u' + 2*u = x, u(0)=0, u'(1)=1")

    u_gal_func, u_expr = solve_galerkin()
    print(f"\nАналитическое приближение:")
    print(f"u(x) = {sp.expand(u_expr)}")

    N = 1000
    x_nodes, u_fdm_vals = solve_fdm(N)

    u_gal_vals = [u_gal_func(x) for x in x_nodes]

    print(
        f"Погрешность в правой точке: {(abs(u_fdm_vals[-1] - u_gal_vals[-1])):.3e}"
    )

    print_res(x_nodes, u_gal_vals, u_fdm_vals)
    plot(x_nodes, u_gal_vals, u_fdm_vals)


if __name__ == '__main__':
    main()
