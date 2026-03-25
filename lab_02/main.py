import math
import matplotlib.pyplot as plt

R_radius = 0.35
l_p = 12.0
L_k = 187e-6
C_k = 268e-6
R_k = 0.25
U_0 = 1400.0
I_0 = 0.5
T_w = 2000.0

table1_I = [0.5, 1, 5, 10, 50, 200, 400, 800, 1200]
table1_T0 = [6730, 6790, 7150, 7270, 8010, 9185, 10010, 11140, 12010]
table1_m = [0.50, 0.55, 1.7, 3, 11, 32, 40, 41, 39]

table2_T = [
    4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000
]
table2_sigma = [
    0.031, 0.27, 2.05, 6.06, 12.0, 19.9, 29.6, 41.1, 54.1, 67.7, 81.5
]


def interpolate(x, x_data, y_data):
    if x <= x_data[0]:
        return y_data[0]

    if x >= x_data[-1]:
        return y_data[-1]

    for i in range(len(x_data) - 1):
        if x_data[i] <= x <= x_data[i + 1]:
            x1, x2 = x_data[i], x_data[i + 1]
            y1, y2 = y_data[i], y_data[i + 1]

            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

    return y_data[-1]


def integrate_sigma(T0, m, steps=40):
    h = 1.0 / steps
    integral = 0.0

    for i in range(steps + 1):
        z = i * h
        T_z = T0 + (T_w - T0) * (z**m)
        sigma_z = interpolate(T_z, table2_T, table2_sigma)

        f_z = sigma_z * z

        if i == 0 or i == steps:
            weight = 1
        elif i % 2 == 1:
            weight = 4
        else:
            weight = 2

        integral += weight * f_z

    return integral * (h / 3.0)


def get_Rp_and_T0(I):
    abs_I = abs(I)

    T0 = interpolate(abs_I, table1_I, table1_T0)
    m = interpolate(abs_I, table1_I, table1_m)

    integral_val = integrate_sigma(T0, m)

    if integral_val < 1e-12:
        integral_val = 1e-12

    Rp = l_p / (2 * math.pi * (R_radius**2) * integral_val)

    return Rp, T0


def solve_circuit(t_max, h, mode="standard"):
    t = 0.0
    I = I_0
    U = U_0

    t_vals, I_vals, U_vals = [t], [I], [U]
    Rp_vals, T0_vals = [], []

    if mode == "standard":
        Rp, T0 = get_Rp_and_T0(I)
    elif mode == "zero_R":
        Rp, T0 = 0.0, 0.0
    elif mode == "const_200":
        Rp, T0 = 200.0, 0.0

    Rp_vals.append(Rp)
    T0_vals.append(T0)

    def dI_dt(curr_I, curr_U):
        if mode == "standard":
            curr_Rp, _ = get_Rp_and_T0(curr_I)
            R_total = R_k + curr_Rp
        elif mode == "zero_R":
            R_total = 0.0
        elif mode == "const_200":
            R_total = 200.0

        return (curr_U - R_total * curr_I) / L_k

    def dU_dt(curr_I):
        return -curr_I / C_k

    steps = int(t_max / h)

    print(f"Запуск симуляции '{mode}' (шаг: {h:.1e} с, шагов: {steps})...")

    for _ in range(steps):
        k1_I = dI_dt(I, U)
        k1_U = dU_dt(I)

        k2_I = dI_dt(I + h / 2 * k1_I, U + h / 2 * k1_U)
        k2_U = dU_dt(I + h / 2 * k1_I)

        k3_I = dI_dt(I + h / 2 * k2_I, U + h / 2 * k2_U)
        k3_U = dU_dt(I + h / 2 * k2_I)

        k4_I = dI_dt(I + h * k3_I, U + h * k3_U)
        k4_U = dU_dt(I + h * k3_I)

        I = I + (h / 6) * (k1_I + 2 * k2_I + 2 * k3_I + k4_I)
        U = U + (h / 6) * (k1_U + 2 * k2_U + 2 * k3_U + k4_U)
        t += h

        t_vals.append(t)
        I_vals.append(I)
        U_vals.append(U)

        if mode == "standard":
            Rp, T0 = get_Rp_and_T0(I)
        elif mode == "const_200":
            Rp, T0 = (200.0, 0.0)
        else:
            Rp, T0 = (0.0, 0.0)

        Rp_vals.append(Rp)
        T0_vals.append(T0)

        if mode == "standard" and I < 1:
            break

    return t_vals, I_vals, U_vals, Rp_vals, T0_vals


def run_all_tasks():
    h_std = 1e-7
    t_max_std = 1000e-6
    t_std, I_std, U_std, Rp_std, T0_std = solve_circuit(t_max_std,
                                                        h_std,
                                                        mode="standard")

    print(f"Левый край:\n\tНапряжение: {U_std[0]:.6f}, Ток: {I_std[0]:.6f}")
    print(
        f"Правый край:\n\tНапряжение: {U_std[-1]:.6f}, Ток: {I_std[-1]:.6f}, время: {t_std[-1]:.6f}"
    )

    # Обратный ход Рунге-Кутты. Доказательство неустойчивости системы
    # =========================================================
    t_back = t_std[-1]
    I_back = int(I_std[-1]) + 1
    U_back = int(U_std[-1]) + 1

    h_back = -h_std

    t_rev_vals = [t_back]
    I_rev_vals = [I_back]
    U_rev_vals = [U_back]

    def dI_dt_back(curr_I, curr_U):
        curr_Rp, _ = get_Rp_and_T0(curr_I)
        return (curr_U - (R_k + curr_Rp) * curr_I) / L_k

    def dU_dt_back(curr_I):
        return -curr_I / C_k

    steps_back = len(t_std) - 1
    for _ in range(steps_back):
        k1_I = dI_dt_back(I_back, U_back)
        k1_U = dU_dt_back(I_back)

        k2_I = dI_dt_back(I_back + h_back / 2 * k1_I,
                          U_back + h_back / 2 * k1_U)
        k2_U = dU_dt_back(I_back + h_back / 2 * k1_I)

        k3_I = dI_dt_back(I_back + h_back / 2 * k2_I,
                          U_back + h_back / 2 * k2_U)
        k3_U = dU_dt_back(I_back + h_back / 2 * k2_I)

        k4_I = dI_dt_back(I_back + h_back * k3_I, U_back + h_back * k3_U)
        k4_U = dU_dt_back(I_back + h_back * k3_I)

        I_back += (h_back / 6) * (k1_I + 2 * k2_I + 2 * k3_I + k4_I)
        U_back += (h_back / 6) * (k1_U + 2 * k2_U + 2 * k3_U + k4_U)
        t_back += h_back

        t_rev_vals.append(t_back)
        I_rev_vals.append(I_back)
        U_rev_vals.append(U_back)

    print(f"Стартовый ток (I0)       = {I_0:.6f} А")
    print(f"Ток после обратного хода = {I_rev_vals[-1]:.6f} А")
    print(f"Абсолютная ошибка по току = {abs(I_0 - I_rev_vals[-1]):.3f} А\n")

    print(f"Стартовое напряжение (U0)       = {U_0:.6f} В")
    print(f"Напр. после обратного хода      = {U_rev_vals[-1]:.6f} В")
    print(
        f"Абсолютная ошибка по напряжению = {abs(U_0 - U_rev_vals[-1]):.3f} В")

    plt.figure(figsize=(10, 5))
    plt.plot([t * 1e6 for t in t_std],
             I_std,
             'b-',
             label='Прямой ход (от 0 до конца)',
             linewidth=4,
             alpha=0.5)

    plt.plot([t * 1e6 for t in t_rev_vals],
             I_rev_vals,
             'r--',
             label='Обратный ход (от конца к 0)',
             linewidth=2)

    plt.title('Тест обратимости времени: Прямой и Обратный ход Рунге-Кутты')
    plt.xlabel('Время, мкс')
    plt.ylabel('Ток, А')
    plt.legend()
    plt.grid(True)
    plt.show()
    # =========================================================

    IRp_std = [I_std[i] * Rp_std[i] for i in range(len(I_std))]

    t_std_mks = [t * 1e6 for t in t_std]

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Задача 1: Графики зависимости параметров от времени',
                 fontsize=14)

    axs[0, 0].plot(t_std_mks, I_std, 'b')
    axs[0, 0].set_title('Ток I(t), A')
    axs[0, 0].grid()

    axs[0, 1].plot(t_std_mks, U_std, 'r')
    axs[0, 1].set_title('Напряжение U(t), В')
    axs[0, 1].grid()

    axs[1, 0].plot(t_std_mks, Rp_std, 'g')
    axs[1, 0].set_title('Сопротивление плазмы Rp(t), Ом')
    axs[1, 0].grid()

    axs[1, 1].plot(t_std_mks, IRp_std, 'm')
    axs[1, 1].set_title('Произведение I(t) * Rp(t), В')
    axs[1, 1].grid()

    axs[2, 0].plot(t_std_mks, T0_std, 'k')
    axs[2, 0].set_title('Температура в центре T0(t), K')
    axs[2, 0].set_xlabel('Время, мкс')
    axs[2, 0].grid()

    axs[2, 1].axis('off')

    plt.tight_layout()
    plt.show()

    I_max = max(I_std)
    threshold = 0.35 * I_max

    t1, t2 = None, None
    for i in range(len(I_std) - 1):

        if I_std[i] <= threshold and I_std[i + 1] >= threshold:
            slope = (I_std[i + 1] - I_std[i]) / h_std
            t1 = t_std[i] + (threshold - I_std[i]) / slope

        if I_std[i] >= threshold and I_std[i + 1] <= threshold:
            slope = (I_std[i + 1] - I_std[i]) / h_std
            t2 = t_std[i] + (threshold - I_std[i]) / slope

    if t1 and t2:
        t_imp = (t2 - t1) * 1e6
        print(f"\n--- РЕЗУЛЬТАТЫ ЗАДАЧИ 4 ---")
        print(f"Максимальный ток (I_max) = {I_max:.1f} А")
        print(f"Порог 0.35 * I_max = {threshold:.1f} А")
        print(f"Длительность импульса t_imp = {t_imp:.1f} мкс")

    t_max_zero = 3e-3
    t_zero, I_zero, _, _, _ = solve_circuit(t_max_zero, h_std, mode="zero_R")

    peaks_t = []
    for i in range(1, len(I_zero) - 1):
        if I_zero[i] > I_zero[i - 1] and I_zero[i] > I_zero[i + 1]:
            peaks_t.append(t_zero[i])

    peaks_t = []
    peaks_I = []

    for i in range(1, len(I_zero) - 1):
        if I_zero[i] > I_zero[i - 1] and I_zero[i] > I_zero[i + 1]:
            peaks_t.append(t_zero[i])
            peaks_I.append(I_zero[i])

    if len(peaks_t) >= 3:
        T_first = peaks_t[1] - peaks_t[0]
        T_last = peaks_t[-1] - peaks_t[-2]

        I_first = peaks_I[0]
        I_last = peaks_I[-1]

        print(f"Период 1-й волны:        {T_first * 1e6:.4f} мкс")
        print(f"Период последней волны:  {T_last * 1e6:.4f} мкс")
        print(
            f"Разница периодов:        {abs(T_first - T_last) * 1e6:.6f} мкс")

        print(f"\nАмплитуда 1-го пика:     {I_first:.6f} А")
        print(f"Амплитуда {len(peaks_I)}-го пика:    {I_last:.6f} А")
        print(
            f"Потеря амплитуды за {len(peaks_I)} циклов: {abs(I_first - I_last):.2e} А"
        )
    # =========================================================

    plt.figure(figsize=(10, 4))
    plt.plot([t * 1e6 for t in t_zero], I_zero, color='orange')
    plt.title('Задача 2: I(t) при Rk + Rp = 0 (Незатухающие колебания)')
    plt.xlabel('Время, мкс')
    plt.ylabel('Ток, А')
    plt.grid()
    plt.show()

    h_fast = 1e-7
    t_max_const = 20e-6
    t_const, I_const, _, _, _ = solve_circuit(t_max_const,
                                              h_fast,
                                              mode="const_200")

    plt.figure(figsize=(10, 4))
    plt.plot([t * 1e6 for t in t_const], I_const, color='red')
    plt.title('Задача 3: I(t) при R_total = 200 Ом')
    plt.xlabel('Время, мс')
    plt.ylabel('Ток, А')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    run_all_tasks()
