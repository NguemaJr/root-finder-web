from flask import Flask, render_template, request
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

app = Flask(__name__)

def f_eval(f_expr, x_val):
    x = sp.symbols('x')
    f = sp.lambdify(x, f_expr, 'numpy')
    return f(x_val)

def bisection(f_expr, a, b, tol, max_iter):
    table = []
    if f_eval(f_expr, a) * f_eval(f_expr, b) >= 0:
        return None, [["Invalid interval. f(a) and f(b) must have opposite signs."]]
    for i in range(max_iter):
        c = (a + b) / 2
        fc = f_eval(f_expr, c)
        table.append([i+1, a, b, c, fc])
        if abs(fc) < tol:
            return c, table
        elif f_eval(f_expr, a) * fc < 0:
            b = c
        else:
            a = c
    return c, table

def false_position(f_expr, a, b, tol, max_iter):
    table = []
    if f_eval(f_expr, a) * f_eval(f_expr, b) >= 0:
        return None, [["Invalid interval. f(a) and f(b) must have opposite signs."]]
    for i in range(max_iter):
        fa = f_eval(f_expr, a)
        fb = f_eval(f_expr, b)
        c = (a * fb - b * fa) / (fb - fa)
        fc = f_eval(f_expr, c)
        table.append([i+1, a, b, c, fc])
        if abs(fc) < tol:
            return c, table
        elif fa * fc < 0:
            b = c
        else:
            a = c
    return c, table

def newton_raphson(f_expr, x0, tol, max_iter):
    x = sp.symbols('x')
    f_prime = sp.diff(f_expr, x)
    table = []
    for i in range(max_iter):
        fx = f_eval(f_expr, x0)
        fpx = f_eval(f_prime, x0)
        if fpx == 0:
            return None, [["Zero derivative. Choose another initial guess."]]
        x1 = x0 - fx / fpx
        table.append([i+1, x0, fx, fpx, x1])
        if abs(x1 - x0) < tol:
            return x1, table
        x0 = x1
    return x1, table

def secant(f_expr, x0, x1, tol, max_iter):
    table = []
    for i in range(max_iter):
        fx0 = f_eval(f_expr, x0)
        fx1 = f_eval(f_expr, x1)
        if fx1 - fx0 == 0:
            return None, [["Division by zero."]]
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        table.append([i+1, x0, x1, fx0, fx1, x2])
        if abs(x2 - x1) < tol:
            return x2, table
        x0, x1 = x1, x2
    return x2, table

def plot_graph(f_expr, root):
    x_vals = np.linspace(-10, 10, 400)
    y_vals = [f_eval(f_expr, x) for x in x_vals]
    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label='f(x)')
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(root, color='red', linestyle='--', label=f'Root at x = {root:.5f}')
    plt.legend()
    plt.title('Function Plot')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.savefig('static/root_plot.png')
    plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    table = None
    method_name = ''
    if request.method == 'POST':
        expr_str = request.form['function']
        method = request.form['method']
        decimal_places = int(request.form['decimal_places'])
        tol = 10 ** -decimal_places
        max_iter = int(request.form['max_iter'])

        x = sp.symbols('x')
        f_expr = sp.sympify(expr_str)

        try:
            if method == 'bisection':
                a = float(request.form['a'])
                b = float(request.form['b'])
                result, table = bisection(f_expr, a, b, tol, max_iter)
                method_name = 'Bisection Method'
            elif method == 'false_position':
                a = float(request.form['a'])
                b = float(request.form['b'])
                result, table = false_position(f_expr, a, b, tol, max_iter)
                method_name = 'False Position Method'
            elif method == 'newton_raphson':
                x0 = float(request.form['x0'])
                result, table = newton_raphson(f_expr, x0, tol, max_iter)
                method_name = 'Newton-Raphson Method'
            elif method == 'secant':
                x0 = float(request.form['x0'])
                x1 = float(request.form['x1'])
                result, table = secant(f_expr, x0, x1, tol, max_iter)
                method_name = 'Secant Method'
            if result is not None:
                result = round(result, decimal_places)
                plot_graph(f_expr, result)
        except Exception as e:
            result = f"Error: {e}"
            table = []

    return render_template('index.html', result=result, table=table, method=method_name)

if __name__ == "__main__":
    import os

port = int(os.environ.get("PORT", 5050))  # fallback to 5050 for local testing
app.run(host="0.0.0.0", port=port)

