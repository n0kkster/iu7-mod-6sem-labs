import math
import matplotlib.pyplot as plt

VARIANT = 1

R_CYL = 0.35
T_W = 2000.0
T_0 = 10000.0
P_VAL = 4.0

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
    """Температурное поле в цилиндре."""
    z = r / R_CYL
    return (T_W - T_0) * (z**P_VAL) + T_0


def plank(T):
    """Функция Планка (равновесная плотность энергии)."""
    return (3.084e-4) / (math.exp(4.799e4 / T) - 1.0)


def get_k(T):
    """Коэффициент поглощения k(T) через логарифмическую интерполяцию."""
    xi_val = math.log(T)

    if xi_val <= TABLE_XI[0]:
        eta = TABLE_ETA[0]
    elif xi_val >= TABLE_XI[-1]:
        eta = TABLE_ETA[-1]
    else:
        for i in range(len(TABLE_XI) - 1):
            if TABLE_XI[i] <= xi_val <= TABLE_XI[i + 1]:
                x1, x2 = TABLE_XI[i], TABLE_XI[i + 1]
                y1, y2 = TABLE_ETA[i], TABLE_ETA[i + 1]
                eta = y1 + (y2 - y1) * (xi_val - x1) / (x2 - x1)
                break
    return math.exp(eta)


def get_derivatives(r, u, F):
    """Правые части системы ОДУ (u', F')."""
    T = temp(r)
    k_val = get_k(T)
    up_val = plank(T)

    du_dr = -3.0 * k_val * F

    if r < 1e-12:
        dF_dr = -0.5 * k_val * (u - up_val)
    else:
        dF_dr = -k_val * (u - up_val) - (F / r)

    return du_dr, dF_dr


def shoot_rk4(xi, N_steps=1000):
    """Один расчет системы методом Рунге-Кутты 4-го порядка."""
    h = R_CYL / N_steps
    r, F = 0.0, 0.0
    u = xi * plank(temp(0))

    history = {'r': [r], 'u': [u], 'F': [F]}

    for _ in range(N_steps):
        k1_u, k1_F = get_derivatives(r, u, F)
        k2_u, k2_F = get_derivatives(r + h / 2, u + h / 2 * k1_u,
                                     F + h / 2 * k1_F)
        k3_u, k3_F = get_derivatives(r + h / 2, u + h / 2 * k2_u,
                                     F + h / 2 * k2_F)
        k4_u, k4_F = get_derivatives(r + h, u + h * k3_u, F + h * k3_F)

        u += (h / 6) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
        F += (h / 6) * (k1_F + 2 * k2_F + 2 * k3_F + k4_F)
        r += h

        history['r'].append(r)
        history['u'].append(u)
        history['F'].append(F)

    residual = history['F'][-1] - 0.39 * history['u'][-1]
    return residual, history


def find_optimal_xi(xi_start0, xi_start1, tol=1e-8, max_iter=20):
    """Ищет оптимальный параметр xi методом секущих."""
    res0, _ = shoot_rk4(xi_start0)
    res1, history = shoot_rk4(xi_start1)

    print(f"Итерация 1: xi={xi_start0:.4f}, ошибка={res0:.2e}")
    print(f"Итерация 2: xi={xi_start1:.4f}, ошибка={res1:.2e}")

    xi0, xi1 = xi_start0, xi_start1

    for i in range(3, max_iter + 1):
        if abs(res1 - res0) < 1e-20: break

        xi_new = xi1 - res1 * (xi1 - xi0) / (res1 - res0)

        xi0, res0 = xi1, res1
        xi1 = xi_new
        res1, history = shoot_rk4(xi1)

        print(f"Итерация {i}: xi={xi1:.6f}, ошибка={res1:.2e}")

        if abs(res1) < tol:
            break

    return xi1, history


def plot_results(hist_shoot, xi_final):
    r_vals = hist_shoot['r']
    up_vals = [plank(temp(r)) for r in r_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Задача 2')

    ax1.plot(r_vals,
             up_vals,
             'r--',
             label='Функция Планка $u_p(r)$',
             linewidth=2)
    ax1.plot(hist_shoot['r'],
             hist_shoot['u'],
             'g-.',
             label=f'Стрельба u(r) (xi={xi_final:.6f})',
             linewidth=2)

    ax1.set_xlabel('Радиус r, см')
    ax1.set_ylabel('u(r), Дж/см³')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(hist_shoot['r'],
             hist_shoot['F'],
             'g-.',
             label='Стрельба F(r)',
             linewidth=2)

    ax2.set_xlabel('Радиус r, см')
    ax2.set_ylabel('Поток F(r)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    print("ВАРИАНТ 2: ОПТИЧЕСКИ ПЛОТНАЯ СРЕДА (k до 200)")

    xi_start_0 = 0.5
    xi_start_1 = 0.9

    if VARIANT == 2:
        xi_start_0 = 0.9999
        xi_start_1 = 1.0

    xi_final, history_shoot = find_optimal_xi(xi_start_0, xi_start_1)
    plot_results(history_shoot, xi_final)


if __name__ == '__main__':
    main()
