import math
import matplotlib.pyplot as plt

VARIANT = 2

R_CYL = 0.35
T_W, T_0 = 2000.0, 10000.0
P_VAL = 4.0
C_LIGHT = 3e10

TABLE_T = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
TABLE_K = [
    8.200e-3, 2.768e-2, 6.560e-2, 1.281e-1, 2.214e-1, 3.516e-1, 5.248e-1,
    7.472e-1, 1.025e+0
]

if VARIANT == 2:
    TABLE_K = [
        1.600e+00, 5.400e+00, 1.280e+01, 2.500e+01, 4.320e+01, 6.860e+01,
        1.024e+02, 1.458e+02, 2.000e+02
    ]

TABLE_XI = [math.log(t) for t in TABLE_T]
TABLE_ETA = [math.log(k) for k in TABLE_K]


def temp(r):
    return (T_W - T_0) * ((r / R_CYL)**P_VAL) + T_0


def plank(T):
    return (3.084e-4) / (math.exp(4.799e4 / T) - 1.0)


def get_k(T):
    xi = math.log(T)
    if xi <= TABLE_XI[0]: eta = TABLE_ETA[0]
    elif xi >= TABLE_XI[-1]: eta = TABLE_ETA[-1]
    else:
        for i in range(len(TABLE_XI) - 1):
            if TABLE_XI[i] <= xi <= TABLE_XI[i + 1]:
                x1, x2, y1, y2 = TABLE_XI[i], TABLE_XI[
                    i + 1], TABLE_ETA[i], TABLE_ETA[i + 1]
                eta = y1 + (y2 - y1) * (xi - x1) / (x2 - x1)
                break
    return math.exp(eta)


def solve_by_balance_method(N=100):
    """Решает задачу методом баланса (разностная прогонка)."""
    h = R_CYL / N
    r = [i * h for i in range(N + 1)]
    k_vals = [get_k(temp(ri)) for ri in r]
    up_vals = [plank(temp(ri)) for ri in r]

    alpha = [0.0] * (N + 1)
    beta = [0.0] * (N + 1)

    r_05 = 0.5 * h
    k_05 = get_k(temp(r_05))

    B0 = 1.0 / (2.0 * k_05)
    C0 = B0 + (3.0 / 8.0) * (h**2) * k_vals[0]
    D0 = (3.0 / 8.0) * (h**2) * k_vals[0] * up_vals[0]

    alpha[1] = B0 / C0
    beta[1] = D0 / C0

    for i in range(1, N):
        r_ph, r_mh = r[i] + 0.5 * h, r[i] - 0.5 * h
        k_ph, k_mh = get_k(temp(r_ph)), get_k(temp(r_mh))

        Ai = r_mh / (h * k_mh)
        Bi = r_ph / (h * k_ph)
        Ci = Ai + Bi + 3.0 * h * r[i] * k_vals[i]
        Di = 3.0 * h * r[i] * k_vals[i] * up_vals[i]

        denom = Ci - Ai * alpha[i]
        alpha[i + 1] = Bi / denom
        beta[i + 1] = (Di + Ai * beta[i]) / denom

    u = [0.0] * (N + 1)
    r_Nh = R_CYL - 0.5 * h
    k_Nh = get_k(temp(r_Nh))

    AN = r_Nh / (h * k_Nh)
    CN = 1.17 * R_CYL + AN + 1.5 * h * R_CYL * k_vals[N]
    DN = 1.5 * h * R_CYL * k_vals[N] * up_vals[N]

    u[N] = (DN + AN * beta[N]) / (CN - AN * alpha[N])
    for i in range(N - 1, -1, -1):
        u[i] = alpha[i + 1] * u[i + 1] + beta[i + 1]

    return r, u, k_vals, up_vals


def calculate_fluxes(r, u, k_vals, up_vals):
    """Вычисляет поток F, его дивергенцию и сравнивает F(R) разными способами."""
    N = len(r) - 1
    h = R_CYL / N
    F = [0.0] * (N + 1)
    divF = [C_LIGHT * k_vals[i] * (up_vals[i] - u[i]) for i in range(N + 1)]

    for i in range(1, N):
        du_dr = (u[i + 1] - u[i - 1]) / (2.0 * h)
        F[i] = -(C_LIGHT / (3.0 * k_vals[i])) * du_dr

    du_dr_R = (3.0 * u[N] - 4.0 * u[N - 1] + u[N - 2]) / (2.0 * h)
    F[N] = -(C_LIGHT / (3.0 * k_vals[N])) * du_dr_R

    f_boundary = C_LIGHT * 0.39 * u[N]

    integral = sum((0.5 if (i == 0 or i == N) else 1.0) * r[i] * divF[i]
                   for i in range(N + 1)) * h
    f_integral = integral / R_CYL

    return F, divF, f_boundary, f_integral


def plot_all(r, u, up_vals, F, divF, k_vals):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].plot(r, u, 'b-', label='u(r)', linewidth=2)
    axs[0, 0].plot(r, up_vals, 'r--', label='u_p(r)')
    axs[0, 0].set_title('Плотность энергии')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(r, F, 'g-', label='F(r)', linewidth=2)
    axs[0, 1].set_title('Профиль потока')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(r, divF, 'm-', label='div F(r)', linewidth=2)
    axs[1, 0].set_title('Дивергенция потока')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(r, k_vals, 'k-', label='k(r)')
    axs[1, 1].set_title('Коэффициент поглощения')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    print("ЗАДАЧА 3: МЕТОД КОНЕЧНЫХ РАЗНОСТЕЙ (БАЛАНС)")

    r, u, k_vals, up_vals = solve_by_balance_method(N=200)

    F, divF, f_target, f_int = calculate_fluxes(r, u, k_vals, up_vals)

    print("-" * 45)

    print(f"F(R) из краевого условия: {f_target:.5e}")
    print(f"F(R) из интеграла divF:  {f_int:.5e}")
    err = abs(f_target - f_int) / f_target * 100
    print(f"Разница методов: {err:.5f}%")
    print("-" * 45)

    plot_all(r, u, up_vals, F, divF, k_vals)


if __name__ == '__main__':
    main()
