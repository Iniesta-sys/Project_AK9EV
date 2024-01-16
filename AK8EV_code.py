import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import plotly.offline
import plotly.graph_objects as go
from joblib import Parallel, delayed
import DE

# 1. Alpine n1 -- https://www.infinity77.net/global_optimization/test_functions_nd_A.html
def alpine_n1(input_array):
    input_array = np.array(input_array)
    return np.sum(np.abs(input_array * np.sin(input_array) + 0.1 * input_array))

# 2. Sinusoidal
def sinusoidal(input_array):
    input_array = np.array(input_array)
    return np.sum(np.sin(input_array) + np.sin(3 * input_array) + np.sin(5 * input_array))

# 3. Katsuura
def katsuura(input_array):
    input_array = np.array(input_array)
    product_term = np.prod([(1 + (i + 1) * np.sum([2**j * np.abs(2**j * input_array[i] - round(2**j * input_array[i])) for j in range(1, 33)]) * (10 / len(input_array))) for i in range(len(input_array))])
    return 10 / len(input_array) * 2 * product_term - (10 / len(input_array) * 2)

# 4. Rastrigin
def rastrigin(input_array):
    input_array = np.array(input_array)
    return 10 * len(input_array) + np.sum(input_array**2 - 10 * np.cos(2 * np.pi * input_array))

# 5. Langermann
def langermann(input_array, m=4):
    input_array = np.array(input_array)
    A = np.array([[3, 5, 2, 1, 7], [5, 2, 1, 4, 3], [2, 1, 6, 7, 8], [1, 4, 7, 3, 9]])
    C = np.array([1, 2, 5, 2, 3])
    sum_terms = [C[i] * np.exp(-np.sum((input_array - A[i, :len(input_array)])**2) / np.pi) * np.cos(np.pi * np.sum((input_array - A[i, :len(input_array)])**2)) for i in range(m)]
    return -np.sum(sum_terms)

# 6. Styblinski
def styblinski(input_array):
    input_array = np.array(input_array)
    return 0.5 * np.sum(input_array**4 - 16 * input_array**2 + 5 * input_array)

# 7. Levy
def levy(input_array):
    input_array = np.array(input_array)
    w = 1 + (input_array - 1) / 4
    term1 = (np.sin(np.pi * w[0]))**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

# 8. Quartic
def quartic(input_array):
    input_array = np.array(input_array)
    return np.sum(np.arange(1, len(input_array) + 1) * input_array**4) + np.random.uniform(0, 1)

# 9. Xin She Yang N.2
def xin_she_yang_n2(input_array):
    input_array = np.array(input_array)
    return np.sum(np.abs(input_array)) * np.exp(-np.sum(np.sin(input_array**2)))

# 10. Happy Cat
def happy_cat(input_array, alpha=1.0 / 8):
    input_array = np.array(input_array)
    norm_sq = np.sum(input_array**2)
    return ((norm_sq - len(input_array))**2)**alpha + (0.5 * norm_sq + np.sum(input_array)) / len(input_array) + 0.5

# 11. Periodic Function
def perm_function(input_vector, b=0.5):
    input_vector = np.asarray_chkfinite(input_vector)
    num_elements = len(input_vector)
    indices = np.arange(1., num_elements + 1)
    vector_divided_by_indices = np.abs(input_vector) / indices
    return np.mean([(np.mean((indices**k + b) * (vector_divided_by_indices**k - 1))**2) for k in indices / num_elements])

# 12. Ackley
def ackley(input_array, a=20, b=0.2, c=2 * np.pi):
    input_array = np.array(input_array)
    part1 = -a * np.exp(-b * np.sqrt(np.mean(input_array**2)))
    part2 = -np.exp(np.mean(np.cos(c * input_array)))
    return part1 + part2 + a + np.e

# 13. Xin She Yang N.4
def xin_she_yang_n4(input_array):
    input_array = np.array(input_array)
    sum_sq = np.sum(input_array**2)
    return (np.sum(np.sin(input_array)**2) + np.exp(-sum_sq)) * np.exp(-np.sum(np.sin(np.sqrt(np.abs(input_array)))**2))

# 14. Ackley N.4
def ackley_n4(input_array):
    input_array = np.array(input_array)
    return sum(np.exp(-0.2 * np.sqrt(input_array[:-1]**2 + input_array[1:]**2)) + 3 * (np.cos(2 * input_array[:-1]) + np.sin(2 * input_array[1:])))

# 15. Styblinski Tang
def styblinski_tang(input_array):
    input_array = np.array(input_array)
    return 0.5 * np.sum(input_array**4 - 16 * input_array**2 + 5 * input_array)

# 16. Salomon
def salomon(input_array):
    input_array = np.array(input_array)
    norm = np.sqrt(np.sum(input_array**2))
    return 1 - np.cos(2 * np.pi * norm) + 0.1 * norm

# 17. Xin She Yang
def xin_she_yang(input_array):
    input_array = np.array(input_array)
    return np.sum(np.random.uniform(0, 1) * np.abs(input_array)**np.arange(1, len(input_array) + 1))

# 18. Alpine N.2
def alpine_n2(input_array):
    input_array = np.array(input_array)
    return np.product(np.sqrt(input_array) * np.sin(input_array))

# 19. Michalewicz
def michalewicz(input_array, m=5):
    input_array = np.array(input_array)
    indices = np.arange(1, len(input_array) + 1)
    return -np.sum(np.sin(input_array) * (np.sin(indices * input_array**2 / np.pi)**(2 * m)))

# 20. Schwefel
def schwefel(input_array):
    input_array = np.array(input_array)
    return 418.9829 * len(input_array) - np.sum(input_array * np.sin(np.sqrt(np.abs(input_array))))

# 21. Schubert Cubic
def schubert_cubic(input_array):
    input_array = np.array(input_array)
    sum_terms = [np.sum([np.cos((j + 1) * x + j) for j in range(1, 6)]) for x in input_array]
    return np.sum(sum_terms)

# 22. Griewank
def griewank(input_array):
    input_array = np.array(input_array)
    sum_sq = np.sum(input_array**2)
    prod_cos = np.prod(np.cos(input_array / np.sqrt(np.arange(1, len(input_array) + 1))))
    return 1 + sum_sq / 4000 - prod_cos

# 23. Dixon Price
def dixon_price(input_vector):
    input_vector = np.asarray_chkfinite(input_vector)
    num_elements = len(input_vector)
    indices = np.arange(2, num_elements + 1)
    squared_elements = 2 * input_vector**2
    return np.sum(indices * (squared_elements[1:] - input_vector[:-1])**2) + (input_vector[0] - 1)**2

# 24. Shubert 4
def shubert_4(input_array):
    input_array = np.array(input_array)
    return np.sum([j * np.cos((j + 1) * x + j) for j in range(1, 6) for x in input_array])

# 25. Shubert 3
def shubert_3(input_array):
    input_array = np.array(input_array)
    return np.sum([j * np.sin((j + 1) * x + j) for j in range(1, 6) for x in input_array])






# Funkce pro vizualizaci
def render_plot(eval_func, plot_type='3d', x_range=(-5, 5), y_range=(-5, 5), resolution=100):
    func_name = eval_func.__name__
    plot_directory = f'generated_plots/{func_name}'
    os.makedirs(plot_directory, exist_ok=True)

    x_values = np.linspace(x_range[0], x_range[1], resolution)
    y_values = np.linspace(y_range[0], y_range[1], resolution)

    if plot_type == '3d':
        X, Y = np.meshgrid(x_values, y_values)
        Z = np.array([[eval_func([X[i, j], Y[i, j]]) for j in range(len(y_values))] for i in range(len(x_values))])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='coolwarm')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Function Value')
        plt.title(f'{func_name} - 3D Plot')
    elif plot_type == '2d':
        y_plotted = [eval_func([x_val]) for x_val in x_values]
        plt.plot(x_values, y_plotted, color='green')
        plt.xlabel('X Axis')
        plt.ylabel('Function Value')
        plt.title(f'{func_name} - 2D Plot')

    save_path = os.path.join(plot_directory, f'{func_name}_{plot_type}.png')
    plt.savefig(save_path)
    print(f'Plot for {func_name} ({plot_type}) created and saved.')
    plt.clf()
    
    
    
    
    

# Funkce pro spuštění diferenciální evoluce
def define_boundaries(dim):
    return np.asarray([(-100, 100) for _ in range(dim)])


def calculate_iterations(dim):
    return int(2000 * dim)

def save_results(func, algorithm, dim, results):
    func_name = func.__name__
    file_path = f'results/{dim}d/{algorithm}/{func_name}.txt'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
        file.write(f'values: {results}\nbest: {min(results)}')

def execute_de_rand(dimensions, population, factor, crossover, max_iter=30):
    for func in [alpine_n1, sinusoidal, katsuura, rastrigin, langermann, styblinski, levy, quartic, xin_she_yang_n2, happy_cat, perm_function, ackley, xin_she_yang_n4, ackley_n4, styblinski_tang, salomon, xin_she_yang, alpine_n2, michalewicz, schwefel, schubert_cubic, griewank, dixon_price, shubert_4, shubert_3]:
        try:
            print(f'Executing {func.__name__} in {dimensions} dimensions...')
            results = Parallel(n_jobs=-1)(delayed(DE.de_rand)(
                func,
                define_boundaries(dimensions),
                population,
                factor,
                crossover,
                calculate_iterations(dimensions)) for _ in range(max_iter))
            print(f'Optimal solution for {func.__name__}: {min(results)}.')
            save_results(func, DE.de_rand.__name__, dimensions, results)
        except Exception as e:
            print(f'Error in {func.__name__} with {dimensions} dimensions: {e}')

def execute_de_best(dimensions, population, factor, crossover, max_iter=30):
    for func in [alpine_n1, sinusoidal, katsuura, rastrigin, langermann, styblinski, levy, quartic, xin_she_yang_n2, happy_cat, perm_function, ackley, xin_she_yang_n4, ackley_n4, styblinski_tang, salomon, xin_she_yang, alpine_n2, michalewicz, schwefel, schubert_cubic, griewank, dixon_price, shubert_4, shubert_3]:
        try:
            print(f'Executing {func.__name__} in {dimensions} dimensions...')
            results = Parallel(n_jobs=-1)(delayed(DE.de_best)(
                func,
                define_boundaries(dimensions),
                population,
                factor,
                crossover,
                calculate_iterations(dimensions)) for _ in range(max_iter))
            print(f'Optimal solution for {func.__name__}: {min(results)}.')
            save_results(func, DE.de_best.__name__, dimensions, results)
        except Exception as e:
            print(f'Error in {func.__name__} with {dimensions} dimensions: {e}')
            
# Reflect Bounds
def reflect_bounds(position, bounds):
    print('reflect bounds')
    for i, (val, bound) in enumerate(zip(position, bounds)):
        if val < bound[0]:
            position[i] = bound[0] + (bound[0] - val)
        elif val > bound[1]:
            position[i] = bound[1] - (val - bound[1])
    return position      

# Třída Particle pro PSO
class Particle:
    def __init__(self, bounds):
        self.position = np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        self.velocity = np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')
            
            
            

# Spouštění optimalizačních algoritmů
execute_de_rand(10, 20, 0.8, 0.9)
execute_de_rand(30, 50, 0.8, 0.9)
execute_de_best(2, 10, 0.8, 0.9)
execute_de_best(10, 20, 0.8, 0.9)
execute_de_best(30, 50, 0.8, 0.9)

# Generování grafů
for func in [alpine_n1, sinusoidal, katsuura, rastrigin, langermann, styblinski, levy, quartic, xin_she_yang_n2, happy_cat, perm_function, ackley, xin_she_yang_n4, ackley_n4, styblinski_tang, salomon, xin_she_yang, alpine_n2, michalewicz, schwefel, schubert_cubic, griewank, dixon_price, shubert_4, shubert_3]:
    render_plot(func, type='3d')
    render_plot(func, type='2d')
