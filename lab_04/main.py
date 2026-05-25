import sys
import math
import matplotlib.pyplot as plt

# =====================================================================
# 1. НАЧАЛЬНЫЕ УСЛОВИЯ И ПАРАМЕТРЫ ЗАДАЧИ
# =====================================================================
VARIANT = 1     # 1 или 2
SOLUTION = 1    # 1 - неявная схема, 2 - итерации

VARIANTS_DATA = {
    1: {
        "N": 50,
        "tau": 5e-6,
        "EPSILON": 1e-4,
        "MAX_ITER": 50,
        "XI_REL": 0.5,
        "TABLE_k": [8.2e-3, 2.768e-2, 6.56e-2, 1.281e-1, 2.214e-1, 3.516e-1, 5.248e-1, 7.472e-1, 1.025]
    },
    2: {
        "N": 150,
        "tau": 1e-7,
        "EPSILON": 1e-4,
        "MAX_ITER": 150,
        "XI_REL": 0.05,
        "TABLE_k": [1.600, 5.400, 1.280e+1, 2.500e+1, 4.320e+1, 6.860e+1, 1.024e+2, 1.458e+2, 2.000e+2]
    }
}

# Валидация выбора
if VARIANT not in VARIANTS_DATA:
    print(f"Ошибка: Вариант {VARIANT} не существует. Выберите 1 или 2.")
    sys.exit(1)

# Извлекаем параметры текущего варианта
cfg = VARIANTS_DATA[VARIANT]

T0 = 8000.0         # Температура в центре при t=0, К
T_w = 1800.0        # Температура на стенке, К
p_param = 12.0       # Параметр p для начального профиля
R = 0.35            # Радиус трубки, см
I_max = 1000.0      # Амплитуда тока, А
t_max = 80e-6       # Время максимума тока, сек (80 мкс)
t_end = 200e-6      # Конечное время расчета, сек (200 мкс)
C_LIGHT = 3e10      # Скорость света, см/с


# Параметры сетки
N = cfg["N"]        # Количество шагов по радиусу
tau = cfg["tau"]    # Шаг по времени, сек
h = R / N           # Шаг по пространству, см


# Моменты времени для графиков: 4 на переднем фронте, 4 на заднем
PRINT_TIMES =[
    10e-6, 30e-6, 50e-6, 80e-6,   # Передний фронт (ток растет)
    100e-6, 120e-6, 150e-6, 200e-6, # Задний фронт (ток падает)
    # 100e-5, 120e-5, 150e-5, 200e-5,
]

if SOLUTION == 2:
    # --- Параметры для МНОГОКРАТНОЙ ПРОГОНКИ (Итераций) ---
    if VARIANT == 1:
        EPSILON = cfg["EPSILON"]        # Точность сходимости итераций (в Кельвинах)
        MAX_ITER = cfg["MAX_ITER"]      # Максимальное число итераций на одном шаге по времени
        XI_REL = cfg["XI_REL"]          # Коэффициент релаксации (кси). От 0.05 до 1.0. 
                                        # 0.5 означает: "верим новой расчетной температуре на 50%, а на 50% старой догадке"

# =====================================================================
# 2. ТАБЛИЦЫ СВОЙСТВ ПЛАЗМЫ
# =====================================================================
TABLE_T = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
TABLE_SIGMA = [0.309e-3, 0.309e-2, 0.309e-1, 0.270, 2.05, 6.06, 12.0, 19.9, 29.6, 41.1, 54.1]
TABLE_LAMBDA = [0.381e-3, 0.381e-3, 0.381e-3, 0.448e-3, 0.577e-3, 0.733e-3, 0.131e-2, 0.218e-2, 0.358e-2, 0.562e-2, 0.832e-2]
TABLE_C = [1.90e-3, 1.90e-3, 0.95e-3, 0.75e-3, 0.64e-3, 0.61e-3, 0.66e-3, 0.66e-3, 1.15e-3, 1.79e-3, 2.02e-3]

TABLE_T_k = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
TABLE_k = cfg["TABLE_k"] 
    
# =====================================================================
# 3. ФУНКЦИИ ФИЗИКИ И МАТЕМАТИК
# =====================================================================

def interpolate(val, x_array, y_array):
    if val <= x_array[0]: return y_array[0]
    if val >= x_array[-1]: return y_array[-1]
    for i in range(len(x_array) - 1):
        if x_array[i] <= val <= x_array[i+1]:
            x0, x1 = x_array[i], x_array[i+1]
            y0, y1 = y_array[i], y_array[i+1]
            return y0 + (y1 - y0) * (val - x0) / (x1 - x0)

def get_sigma(T): return interpolate(T, TABLE_T, TABLE_SIGMA)
def get_lambda(T): return interpolate(T, TABLE_T, TABLE_LAMBDA)
def get_c(T): return interpolate(T, TABLE_T, TABLE_C)
def get_k(T): return interpolate(T, TABLE_T_k, TABLE_k)

def I(t):
    # return I_max
    return I_max * (t / t_max) * math.exp(-(t / t_max - 1))

def u_p(T):
    T_safe = max(T, 1.0)
    power = 4.799e4 / T_safe
    if power > 700:
        return 0.0
    return (3.084e-4) / math.expm1(power)

def progonka(A, B, C, F):
    """Метод прогонки"""
    n = len(B)
    alpha, beta, x = [0.0] * n, [0.0] * n, [0.0] * n

    alpha[1] = C[0] / B[0]
    beta[1] = F[0] / B[0]
    
    for i in range(1, n - 1):
        denom = B[i] - A[i] * alpha[i]
        alpha[i+1] = C[i] / denom
        beta[i+1] = (A[i] * beta[i] + F[i]) / denom
        
    denom = B[n-1] - A[n-1] * alpha[n-1]
    x[n-1] = (A[n-1] * beta[n-1] + F[n-1]) / denom
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i+1] * x[i+1] + beta[i+1]
    return x

# =====================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ РАСЧЕТА
# =====================================================================

def compute_coeffs(T_dist):
    """Вычисляет массивы физических свойств для текущего распределения температур."""
    sigma = [get_sigma(T_dist[i]) for i in range(N + 1)]
    lam = [get_lambda(T_dist[i]) for i in range(N + 1)]
    c_T = [get_c(T_dist[i]) for i in range(N + 1)]
    k_coef = [get_k(T_dist[i]) for i in range(N + 1)]
    up = [u_p(T_dist[i]) for i in range(N + 1)]
    return sigma, lam, c_T, k_coef, up

def solve_radiation(r, h, k_coef, up):
    """Решает уравнение переноса излучения в диффузионном приближении."""
    Au, Bu, Cu, Fu = [0.0] * (N + 1), [0.0] * (N + 1), [0.0] * (N + 1), [0.0] * (N + 1)
    for i in range(1, N):
        D_minus = 1.0 / (3.0 * 0.5 * (k_coef[i-1] + k_coef[i]))
        D_plus  = 1.0 / (3.0 * 0.5 * (k_coef[i] + k_coef[i+1]))
        Au[i] = (r[i] - 0.5 * h) * D_minus / (h**2)
        Cu[i] = (r[i] + 0.5 * h) * D_plus / (h**2)
        Bu[i] = Au[i] + Cu[i] + k_coef[i] * r[i]
        Fu[i] = k_coef[i] * up[i] * r[i]
        
    # Граничные условия
    Bu[0], Cu[0], Au[0], Fu[0] = 1.0, 1.0, 0.0, 0.0
    D_N = 1.0 / (3.0 * k_coef[N])
    Au[N], Bu[N], Cu[N], Fu[N] = D_N, D_N + 0.39 * h, 0.0, 0.0
    
    u_rad = progonka(Au, Bu, Cu, Fu)
    # Расчет радиационного потока (объемные потери)
    q = [C_LIGHT * k_coef[i] * (up[i] - u_rad[i]) for i in range(N + 1)]
    return q

def solve_temperature_step(T_prev, T_guess, r, h, tau, sigma, lam, c_T, E, q):
    """Формирует и решает систему для определения нового профиля температур."""
    At, Bt, Ct, Ft = [0.0] * (N + 1), [0.0] * (N + 1), [0.0] * (N + 1), [0.0] * (N + 1)
    for i in range(1, N):
        lam_minus = 0.5 * (lam[i-1] + lam[i])
        lam_plus  = 0.5 * (lam[i] + lam[i+1])
        At[i] = (r[i] - 0.5 * h) * lam_minus / (h**2)
        Ct[i] = (r[i] + 0.5 * h) * lam_plus / (h**2)
        Bt[i] = At[i] + Ct[i] + (c_T[i] * r[i]) / tau
        # T_prev берется с прошлого временного слоя, остальное — с итерационного (T_guess)
        Ft[i] = (c_T[i] * r[i] / tau) * T_prev[i] + (sigma[i] * E**2 - q[i]) * r[i]
        
    # Граничные условия
    Bt[0], Ct[0], At[0], Ft[0] = 1.0, 1.0, 0.0, 0.0
    At[N], Bt[N], Ct[N], Ft[N] = 0.0, 1.0, 0.0, T_w
    
    return progonka(At, Bt, Ct, Ft)

# =====================================================================
# 4. ГЛАВНАЯ ПРОГРАММА (РАСЧЕТ)
# =====================================================================
r = [i * h for i in range(N + 1)]
T = [T0 + (T_w - T0) * (ri / R)**p_param for ri in r]

t = 0.0
results = {0.0: list(T)}
history_t, history_I = [], []

# Настройка параметров итераций в зависимости от метода
max_iters = 1 if SOLUTION == 1 else MAX_ITER
relaxation = 1.0 if SOLUTION == 1 else XI_REL
convergence_eps = -1.0 if SOLUTION == 1 else EPSILON # Отключаем проверку для SOLUTION 1

print(f"Начат расчет..")

while t <= t_end + 1e-9:
    I_t = I(t)
    history_t.append(t * 1e6)
    history_I.append(I_t)
    
    T_guess = list(T)
    iter_count = 0
    
    for iteration in range(max_iters):
        iter_count = iteration + 1
        
        # 1. Свойства и Электрическое поле
        sigma, lam, c_T, k_coef, up = compute_coeffs(T_guess)
        
        integral_sigma = sum(0.5 * (sigma[i] * r[i] + sigma[i + 1] * r[i + 1]) * h for i in range(N))
        E = I_t / (2.0 * math.pi * integral_sigma) if integral_sigma > 0 else 0.0
        
        # 2. Излучение
        q = solve_radiation(r, h, k_coef, up)
        
        # 3. Теплопроводность
        T_new_calc = solve_temperature_step(T, T_guess, r, h, tau, sigma, lam, c_T, E, q)
        
        # 4. Релаксация и Сходимость
        max_diff = 0.0
        T_next_guess = [0.0] * (N + 1)
        for i in range(N + 1):
            T_next_guess[i] = relaxation * T_new_calc[i] + (1.0 - relaxation) * T_guess[i]
            max_diff = max(max_diff, abs(T_next_guess[i] - T_guess[i]))
        
        T_guess = T_next_guess
        
        # Условие выхода для итерационного метода
        if max_diff < convergence_eps:
            break
            
    # Обновление состояния
    T = list(T_guess)
    t += tau
    
    # Логирование
    for pt in PRINT_TIMES:
        if abs(t - pt) < (tau * 0.5):
            results[pt] = list(T)
            if SOLUTION != 1:
                print(f"t = {int(t*1e6)} мкс. Итераций: {iter_count}")

print("Расчет завершен.")

# =====================================================================
# 5. ВИЗУАЛИЗАЦИЯ (MATPLOTLIB)
# =====================================================================

# Настраиваем размер окна и создаем два подграфика
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- График 1: Распределение температур ---
# Выбираем красивые стили линий (сплошные для переднего фронта, пунктирные для заднего)
for pt in [0.0] + PRINT_TIMES:
    if pt in results:
        t_mks = int(pt * 1e6)
        # Если время <= 80 мкс (нагрев) рисуем сплошной линией, иначе (остывание) пунктирной
        linestyle = '-' if t_mks <= 80 else '--' 
        linewidth = 2 if t_mks == 80 else 1.5 # Выделим максимум жирнее
        
        ax1.plot(r, results[pt], 
                 label=f't = {t_mks} мкс', 
                 linestyle=linestyle, 
                 linewidth=linewidth)

ax1.set_title("Распределение температуры T(r) в разные моменты времени", fontsize=12)
ax1.set_xlabel("Радиус r, см", fontsize=11)
ax1.set_ylabel("Температура T, К", fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(loc='upper right', fontsize=9)

# --- График 2: Форма импульса тока ---
ax2.plot(history_t, history_I, color='red', linewidth=2)
# Добавим точки на график тока, чтобы было видно, в какие моменты мы "фотографировали" температуру
for pt in PRINT_TIMES:
    t_mks = pt * 1e6
    i_val = I(pt)
    ax2.plot(t_mks, i_val, marker='o', color='black')
    
ax2.set_title("Форма импульса тока I(t)", fontsize=12)
ax2.set_xlabel("Время t, мкс", fontsize=11)
ax2.set_ylabel("Ток I, А", fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.7)

# Показываем всю красоту
plt.tight_layout()
plt.show()
