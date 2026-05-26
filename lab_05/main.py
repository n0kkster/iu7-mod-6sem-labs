import sys
import math
import matplotlib.pyplot as plt

# =====================================================================
# 1. КОНФИГУРАЦИЯ И НАЧАЛЬНЫЕ ДАННЫЕ
# =====================================================================
VARIANT = 2 # 1 - Вариант 1 (левый край нагрев II рода, остальные III)
            # 2 - Вариант 2 (все границы III рода)

# Геометрические размеры пластины
a = 10.0            # Длина пластины по X, см
b = 10.0            # Ширина пластины по Z, см
delta = 0.5         # Толщина пластины, см

# Параметры теплоотдачи и окружения
u0 = 300.0          # Температура окружающей среды (воздуха), К
alpha0 = 0.05       # Коэф. теплоотдачи плоских граней пластины, Вт/(см² К)

# Коэффициенты теплоотдачи на торцах
alpha1 = 0.08       # Для границы 4 (левая, x = 0) в Варианте 2
alpha2 = 0.08       # Для границы 2 (правая, x = a)
alpha3 = 0.08       # Для границы 3 (нижняя, z = 0)
alpha4 = 0.08       # Для границы 1 (верхняя, z = b)

# Тепловой поток на левой границе (только для Варианта 1)
F0 = 30.0           # Вт/см²

# Центры лазерного нагрева f(x,z)
LASER_SOURCES = [
    {"f0": 150.0, "beta": 0.15, "x0": 5.0, "z0": 5.0},
    # {"f0": 150.0, "beta": 0.15, "x0": 1.0, "z0": 9.0},
    # {"f0": 150.0, "beta": 0.15, "x0": 9.0, "z0": 1.0}, 
 
]

# Параметры разностной сетки и времени
Nx = 50             # Шагов по оси X
Nz = 50             # Шагов по оси Z
tau = 0.05          # Шаг виртуального времени, сек
EPSILON = 1e-4      # Критерий выхода на стационарный режим

hx = a / Nx         # Шаг по пространству X, см
hz = b / Nz         # Шаг по пространству Z, см

# =====================================================================
# 2. ФИЗИЧЕСКИЕ ИСТОЧНИКИ И СВОЙСТВА МАТЕРИАЛА
# =====================================================================

def get_lambda(u):
    """Вычисляет коэффициент теплопроводности lambda(u) по формуле (1)"""
    a1 = 0.0134
    b1 = 1.0
    c1 = 4.35e-4
    m1 = 1
    return a1 * (b1 + c1 * (u ** m1))

def get_f_source(x, z):
    """Вычисляет плотность нагрева от лазерных лучей в точке (x, z)"""
    val = 0.0
    for src in LASER_SOURCES:
        power = -src["beta"] * ((x - src["x0"])**2 * (z - src["z0"])**2)
        val += src["f0"] * math.exp(power)
    return val

# =====================================================================
# 3. МАТЕМАТИЧЕСКИЙ ОДНОМЕРНЫЙ РЕШАТЕЛЬ
# =====================================================================

def progonka(A, B, C, F):
    """Решает трехдиагональную систему уравнений: A_i*X_i-1 - B_i*X_i + C_i*X_i+1 = -F_i"""
    n = len(B)
    alpha = [0.0] * n
    beta = [0.0] * n
    x = [0.0] * n
    
    alpha[1] = C[0] / B[0]
    beta[1] = F[0] / B[0]
    
    for i in range(1, n - 1):
        denom = B[i] - A[i] * alpha[i]
        if denom == 0: denom = 1e-15
        alpha[i+1] = C[i] / denom
        beta[i+1] = (A[i] * beta[i] + F[i]) / denom
        
    denom = B[n-1] - A[n-1] * alpha[n-1]
    if denom == 0: denom = 1e-15
    x[n-1] = (A[n-1] * beta[n-1] + F[n-1]) / denom
    
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i+1] * x[i+1] + beta[i+1]
    return x

# =====================================================================
# 4. ЛОКАЛЬНО-ОДНОМЕРНЫЙ МЕТОД
# =====================================================================

def x_sweep(u, u_star):
    """Прогонка по строкам"""
    for k in range(Nz + 1):
        A = [0.0] * (Nx + 1)
        B = [0.0] * (Nx + 1)
        C = [0.0] * (Nx + 1)
        F = [0.0] * (Nx + 1)
        
        # Внутренние узлы строки
        for i in range(1, Nx):
            lam_minus = 0.5 * (get_lambda(u[i-1][k]) + get_lambda(u[i][k]))
            lam_plus  = 0.5 * (get_lambda(u[i][k]) + get_lambda(u[i+1][k]))
            
            A[i] = (tau / hx**2) * lam_minus
            C[i] = (tau / hx**2) * lam_plus
            B[i] = A[i] + C[i] + 1.0 + (tau * alpha0) / delta
            F[i] = u[i][k] + (tau * alpha0 / delta) * u0 + (tau / 2.0) * get_f_source(i * hx, k * hz)
            
        # Левая граница
        if VARIANT == 1:
            # II род: 
            lam_0 = get_lambda(u[0][k])
            A[0] = 0.0
            B[0] = lam_0
            C[0] = lam_0
            F[0] = F0 * hx
        else:
            # III род: 
            lam_0 = get_lambda(u[0][k])
            A[0] = 0.0
            B[0] = lam_0 + alpha1 * hx
            C[0] = lam_0
            F[0] = alpha1 * hx * u0
            
        # Правая граница
        lam_Nx = get_lambda(u[Nx][k])
        A[Nx] = lam_Nx
        B[Nx] = lam_Nx + alpha2 * hx
        C[Nx] = 0.0
        F[Nx] = alpha2 * hx * u0

        if VARIANT == 3:
            A[0] = A[Nx] = 0.0
            B[0] = B[Nx] = 1.0
            C[0] = C[Nx] = 0.0
            F[0] = F[Nx] = u0
        
        sol = progonka(A, B, C, F)
        for i in range(Nx + 1):
            u_star[i][k] = sol[i]

def z_sweep(u_star, u_new):
    """Прогонка по столбцам"""
    for i in range(Nx + 1):
        A = [0.0] * (Nz + 1)
        B = [0.0] * (Nz + 1)
        C = [0.0] * (Nz + 1)
        F = [0.0] * (Nz + 1)
        
        # Внутренние узлы столбца
        for k in range(1, Nz):
            lam_minus = 0.5 * (get_lambda(u_star[i][k-1]) + get_lambda(u_star[i][k]))
            lam_plus  = 0.5 * (get_lambda(u_star[i][k]) + get_lambda(u_star[i][k+1]))
            
            A[k] = (tau / hz**2) * lam_minus
            C[k] = (tau / hz**2) * lam_plus
            B[k] = A[k] + C[k] + 1.0 + (tau * alpha0) / delta
            F[k] = u_star[i][k] + (tau * alpha0 / delta) * u0 + (tau / 2.0) * get_f_source(i * hx, k * hz)
            
        # Нижняя граница
        lam_0 = get_lambda(u_star[i][0])
        A[0] = 0.0
        B[0] = lam_0 + alpha3 * hz
        C[0] = lam_0
        F[0] = alpha3 * hz * u0
        
        # Верхняя граница
        lam_Nz = get_lambda(u_star[i][Nz])
        A[Nz] = lam_Nz
        B[Nz] = lam_Nz + alpha4 * hz
        C[Nz] = 0.0
        F[Nz] = alpha4 * hz * u0

        if VARIANT == 3:
            A[0] = A[Nx] = 0.0
            B[0] = B[Nx] = 1.0
            C[0] = C[Nx] = 0.0
            F[0] = F[Nx] = u0
        
        sol = progonka(A, B, C, F)
        for k in range(Nz + 1):
            u_new[i][k] = sol[k]

# =====================================================================
# 5. СИМУЛЯЦИЯ СХОДИМОСТИ
# =====================================================================

# Инициализируем поле температур начальным значением u0
u = [[u0] * (Nz + 1) for _ in range(Nx + 1)]
u_star = [[0.0] * (Nz + 1) for _ in range(Nx + 1)]
u_new = [[0.0] * (Nz + 1) for _ in range(Nx + 1)]

t = 0.0
step = 0
max_steps = 10000

print(f"Запуск расчета установления. Выбран Вариант КУ: {VARIANT}")
print(f"Сетка: {Nx}x{Nz}, Шаг времени tau: {tau} сек")
print("-" * 50)

while step < max_steps:
    # 1. Попеременный проход по двум направлениям
    x_sweep(u, u_star)
    z_sweep(u_star, u_new)
    
    # 2. Вычисление максимального изменения температуры за шаг
    max_diff = 0.0
    for i in range(Nx + 1):
        for k in range(Nz + 1):
            diff = (abs(u_new[i][k] - u[i][k])) / u_new[i][k]
            if diff > max_diff:
                max_diff = diff
                
    # 3. Перезапись данных для следующего шага
    u = [list(row) for row in u_new]
    t += tau
    step += 1
    
    # Логирование
    if step % 50 == 0 or step == 1:
        print(f"Шаг: {step:<4} | Время: {t:<6.2f} сек | Изменение T_max: {max_diff:.3e} К")
        
    # Условие выхода на стационар
    if max_diff < EPSILON:
        print("-" * 50)
        print(f"СТАЦИОНАРНЫЙ РЕЖИМ ДОСТИГНУТ!")
        print(f"Всего шагов: {step} | Время симуляции: {t:.2f} сек")
        print(f"Максимальная температура в центре пластины: {u[Nx//2][Nz//2]:.2f} К")
        break
else:
    print("Внимание: Превышен лимит шагов по времени!")

# =====================================================================
# 6. ВИЗУАЛИЗАЦИЯ
# =====================================================================
import numpy as np

r_x = [i * hx for i in range(Nx + 1)]
r_z = [k * hz for k in range(Nz + 1)]

X, Z = np.meshgrid(r_x, r_z)
U = np.array(u).T

# Создаем окно визуализации
fig = plt.figure(figsize=(15, 6))

# Левое окно: Трехмерный график
ax3d = fig.add_subplot(121, projection='3d')
surf = ax3d.plot_surface(X, Z, U, cmap='hot', edgecolor='none')
ax3d.set_title(f"3D тепловая карта пластины (Вариант {VARIANT})")
ax3d.set_xlabel("Ось X, см")
ax3d.set_ylabel("Ось Z, см")
ax3d.set_zlabel("Температура u, К")
fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)

# Правое окно: Одномерные срезы
ax2d = fig.add_subplot(122)
# Срез по центру пластины вдоль оси X (при z = b / 2)
for p in range(2, 4):
    ax2d.plot(r_x, [u[i][Nz // p] for i in range(Nx + 1)], linewidth=2, label=f'Срез вдоль X при z={r_z[Nz // p]:.1f} см')

# Срез по центру пластины вдоль оси Z (при x = a / 2)
for p in range(2, 4):
    ax2d.plot(r_z, [u[Nx // p][k] for k in range(Nz + 1)], '--', linewidth=2, label=f'Срез вдоль Z при x={r_x[Nx // p]:.1f} см')

ax2d.set_title("Одномерные температурные срезы пластины")
ax2d.set_xlabel("Координата, см")
ax2d.set_ylabel("Температура u, К")
ax2d.grid(True, linestyle=':', alpha=0.8)
ax2d.legend()

plt.tight_layout()
plt.show()
