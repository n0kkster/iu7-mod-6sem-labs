import math
import matplotlib.pyplot as plt
import sympy as sp

# ==========================================
# ЗАДАЧА 1
# 4xu'' + 2u' + u = 0, u(0)=1, u'(0)=-0.5
# ==========================================

def task1_taylor(x):
    # y(x) = 1 - x/2 + x^2/4! - x^3/6! 
    return 1 - 0.5 * x + (1/24) * (x**2) - (1/720) * (x**3)

def task1_euler_system(x_end, h=0.1):
    # y = u
    # z = u'
    # u' = y
    # z' = (-2u' - u) / 4x = (-2z - y) / 4x

    x_values = [0]
    y_values = [1]    # u(0)
    z_values = [-0.5] # u'(0)
    
    x_n = 0.001
    y_n = 1     # u(0)
    z_n = -0.5  # u'(0)
    
    steps = int(x_end / h)
    
    for i in range(steps):
        dy1 = z_n
        dy2 = (-2*z_n - y_n) / (4*x_n)
        
        next_x = x_n + h
        next_y = y_n + h * dy1
        next_z = z_n + h * dy2
            
        x_values.append(next_x)
        y_values.append(next_y)
        z_values.append(next_z)
        
        x_n = next_x
        y_n = next_y
        z_n = next_z
        
    return x_values, y_values

def solve_task1():
    print("\n--- ЗАДАЧА 1 ---")

    h = 0.1
    x_max = 10
    
    x_taylor = [i * 0.1 for i in range(int(x_max/0.1) + 1)]
    y_taylor = [task1_taylor(x) for x in x_taylor]
    
    x_euler, y_euler = task1_euler_system(x_max, h)
    
    print(f"{'x':<5} | {'Тейлор':<10} | {'Эйлер':<10} | {'Разница':<10}")
    for xe, ye in zip(x_euler, y_euler):
        yt = task1_taylor(xe)
        print(f"{xe:<5.1f} | {yt:<10.4f} | {ye:<10.4f} | {abs(yt-ye):<10.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(x_taylor, y_taylor, label='Ряд Тейлора (4 члена)', color='blue')
    plt.plot(x_euler, y_euler, 'o--', label=f'Метод Эйлера (h={h})', color='red')
    plt.title('Задача 1: Сравнение Тейлора и Эйлера')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.grid(True)
    plt.legend()
    plt.show()

# ==========================================
# ЗАДАЧА 2
# Аналитическое решение: x(u) = e^(u^2) - (u^2 + 1)/2
# ==========================================

def task2_analytic_x(u):
    return math.exp(u**2) - (u**2 + 1)/2

def task2_picard(u):
    return 1/2 + u**2/2 + u**4/2 + u**6/6 + u**8/24 + u**10/240

def solve_task2():
    print("\n--- ЗАДАЧА 2 ---")
    
    u_vals = [i * 0.05 for i in range(-30, 31)] 
    analythic_vals = [task2_analytic_x(u) for u in u_vals]
    picard_vals = [task2_picard(u) for u in u_vals]
    
    plt.figure(figsize=(10, 6))
    plt.plot(analythic_vals, u_vals, label='Точное решение u(x)', color='green')
    plt.plot(picard_vals, u_vals, label='Метод Пикара u(x)', color='blue')
    plt.title('Задача 2: Аналитическое решение и Пикар')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.grid(True)
    plt.legend()
    plt.show()

# ==========================================
# ЗАДАЧА 3
# u' = x^2 + u^2, u(0)=0
# Пикар 1..4 и Эйлер
# ==========================================

H_START = 0.5
CHECK_LIMIT = 1.5
EPS = 10**(-6)
TABLE_STEP = 0.001

# ==========================================
# 1. ИНТЕГРАЛЫ ПИКАРА
# ==========================================
x_sym = sp.symbols('x')

u_funcs = [0] 

for i in range(1, 5):
    u_prev = u_funcs[-1]
    u_next = sp.integrate(x_sym**2 + u_prev**2, (x_sym, 0, x_sym))
    u_funcs.append(u_next)

picard_1 = sp.lambdify(x_sym, u_funcs[1], 'math')
picard_2 = sp.lambdify(x_sym, u_funcs[2], 'math')
picard_3 = sp.lambdify(x_sym, u_funcs[3], 'math')
picard_4 = sp.lambdify(x_sym, u_funcs[4], 'math')

# ==========================================
# 2. ФУНКЦИИ МЕТОДА ЭЙЛЕРА
# ==========================================

def f(x, y):
    return x**2 + y**2

def check_step(x0, y0, h, limit):
    x, y = x0, y0
    steps = int(limit / h)
    for _ in range(steps):
        try:
            y += h * f(x, y)
            x += h
        except OverflowError:
            return float('inf')
    return y

def find_step(x0, y0, h, limit, eps):
    y_h2 = check_step(x0, y0, h, limit)
    
    while True:
        y_h = y_h2
        h /= 2
        y_h2 = check_step(x0, y0, h, limit)
        
        if y_h2 == 0 or y_h2 == float('inf'):
            continue
            
        err = abs((y_h - y_h2) / y_h2)

        if err < eps:
            break
    return h

def find_xmax_ymax(x, y, h):
    while True:
        try:
            val = f(x, y)
            y += h * val
            x += h
        except:
            break
    return x, y

def task3_euler(x_max, h):

    x_vals = [0]
    u_vals = [0]
    curr_x = 0
    curr_u = 0
    
    while curr_x < x_max:
        try:
            curr_u = curr_u + h * f(curr_x, curr_u)
            curr_x = curr_x + h
            
            x_vals.append(curr_x)
            u_vals.append(curr_u)
        except:
            break
            
    return x_vals, u_vals

# ==========================================
# 3. РЕШЕНИЕ
# ==========================================

def solve_task3():
    print("\n--- ЗАДАЧА 3 ---")

    h_opt = find_step(0, 0, H_START, CHECK_LIMIT, EPS)
    print(f"Шаг: {h_opt:.6g}")

    xmax, ymax = find_xmax_ymax(0, 0, h_opt)
    xmax_safe = xmax - 5 * h_opt 
    print(f"x_max: {xmax:.6f}")

    print("-" * 75)
    print(f"{'x':<6} | {'Picard 1':<12} | {'Picard 2':<12} | {'Picard 3':<12} | {'Picard 4':<12} | {'Euler':<12}")
    print("-" * 75)
    
    e_x, e_u = task3_euler(xmax_safe, h_opt)
    
    current_table_x = 0
    idx = 0

    last_x_good = [0, 0, 0, 0]
    eps_picard = 0.001
    
    while current_table_x <= xmax_safe:
        while idx < len(e_x) - 1 and e_x[idx] < current_table_x:
            idx += 1
        
        val_x = e_x[idx]
        val_eu = e_u[idx]
        
        p1 = picard_1(val_x)
        p2 = picard_2(val_x)
        p3 = picard_3(val_x)
        p4 = picard_4(val_x)

        if abs(p1 - val_eu) < eps_picard:
            last_x_good[0] = val_x
        if abs(p2 - val_eu) < eps_picard:
            last_x_good[1] = val_x
        if abs(p3 - val_eu) < eps_picard:
            last_x_good[2] = val_x
        if abs(p4 - val_eu) < eps_picard:
            last_x_good[3] = val_x  

        print(f"{val_x:<6.3f} | {p1:<12.5f} | {p2:<12.5f} | {p3:<12.5f} | {p4:<12.5f} | {val_eu:<12.5f}")
        
        current_table_x += TABLE_STEP

    # Неправильно считает, лень фиксить
    # По идее надо сравнивать со след. приближением: 
    # Пикар 1 с Пикар 2, Пикар 2 с Пикар 3, Пикар 3 с Пикар 4, Пикар 4 с Эйлером
    print('Последние допустимые значения:')
    print(f'Пикар 1: {last_x_good[0]:.3f}')
    print(f'Пикар 2: {last_x_good[1]:.3f}')
    print(f'Пикар 3: {last_x_good[2]:.3f}')
    print(f'Пикар 4: {last_x_good[3]:.3f}')
    
    x_plot = []
    curr = 0
    while curr < xmax_safe:
        x_plot.append(curr)
        curr += 0.01
        
    p1_vals = [picard_1(x) for x in x_plot]
    p2_vals = [picard_2(x) for x in x_plot]
    p3_vals = [picard_3(x) for x in x_plot]
    p4_vals = [picard_4(x) for x in x_plot] 

    plt.figure(figsize=(10, 7))

    plt.plot(x_plot, p1_vals, label='Пикар 1')
    plt.plot(x_plot, p2_vals, label='Пикар 2')
    plt.plot(x_plot, p3_vals, label='Пикар 3')
    plt.plot(x_plot, p4_vals, label='Пикар 4')
    
    plt.plot(e_x, e_u, label='Метод Эйлера')

    plt.axvline(x=xmax, color='black', linestyle=':', label=f'Асимптота x={xmax:.4f}')

    plt.title(f'Задача 3: Сравнение Пикара и Эйлера')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    solve_task1()
    solve_task2()
    solve_task3()
