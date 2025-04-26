import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from copy import deepcopy

class TransportSolver:
    def __init__(self, master):
        self.master = master
        master.title("Rezolvare Problema Transport")
        master.geometry("900x700")
        icon=tk.PhotoImage(file="icon.png")
        master.iconphoto(True,icon)

        # Date probleme
        self.supply = []
        self.demand = []
        self.costs = []
        self.solution = []
        self.iterations = []
        self.current_iter = 0

        # Setare UI
        self.setup_ui()

    def setup_ui(self):
        """Configurează interfața grafică"""
        main_frame = tk.Frame(self.master, padx=15, pady=15)
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Creăm un frame principal cu două coloane
        dual_frame = tk.Frame(main_frame)
        dual_frame.pack(expand=True, fill=tk.BOTH)

        # Coloana stângă (datele problemei)
        left_frame = tk.Frame(dual_frame, padx=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Coloana dreaptă (rezolvarea)
        right_frame = tk.Frame(dual_frame, padx=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Secțiune introducere date (în stânga)
        input_frame = tk.LabelFrame(left_frame, text="Date Problema", padx=10, pady=10)
        input_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Butoane introducere
        btn_frame = tk.Frame(input_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        tk.Button(btn_frame, text="Introducere Manuală", 
                command=self.setup_manual_input).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Încarcă din Fișier", 
                command=self.load_from_file).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Date Test", 
                command=self.load_test_data).pack(side=tk.LEFT, padx=5)

        self.input_grid = tk.Frame(input_frame)
        self.input_grid.pack(fill=tk.BOTH, expand=True)

        # Buton rezolvare (rămâne în stânga)
        tk.Button(left_frame, text="REZOLVĂ", font=('Arial', 10, 'bold'),
                bg='#4CAF50', fg='white', command=self.solve_problem).pack(pady=10)

        # Secțiune soluție (mutată în dreapta)
        solution_frame = tk.LabelFrame(right_frame, text="Rezolvare", padx=10, pady=10)
        solution_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Navigare
        nav_frame = tk.Frame(solution_frame)
        nav_frame.pack(fill=tk.X, pady=10)

        self.prev_btn = tk.Button(nav_frame, text="← Anterior", 
                                state=tk.DISABLED, command=self.prev_step)
        self.prev_btn.pack(side=tk.LEFT, padx=5)

        self.next_btn = tk.Button(nav_frame, text="Următor →", 
                                state=tk.DISABLED, command=self.next_step)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        self.iter_label = tk.Label(nav_frame, text="Pas: 0/0")
        self.iter_label.pack(side=tk.LEFT, padx=10)

        # Afișare soluție
        self.solution_text = tk.Text(solution_frame, wrap=tk.WORD, height=18,
                                    font=('Consolas', 10))
        scrollbar = tk.Scrollbar(solution_frame, command=self.solution_text.yview)
        self.solution_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.solution_text.pack(fill=tk.BOTH, expand=True)

        # Variabile introducere
        self.rows_var = tk.IntVar(value=3)
        self.cols_var = tk.IntVar(value=4)
        self.cost_entries = []
        self.supply_entries = []
        self.demand_entries = []

    def setup_manual_input(self):
        """Configurează interfața pentru introducere manuală"""
        for widget in self.input_grid.winfo_children():
            widget.destroy()

        tk.Label(self.input_grid, text="Nr. depozite:").grid(row=0, column=0)
        tk.Entry(self.input_grid, textvariable=self.rows_var, width=5).grid(row=0, column=1)

        tk.Label(self.input_grid, text="Nr. magazine:").grid(row=1, column=0)
        tk.Entry(self.input_grid, textvariable=self.cols_var, width=5).grid(row=1, column=1)

        tk.Button(self.input_grid, text="Generează tabel", 
                 command=self.generate_input_grid).grid(row=2, column=0, columnspan=2, pady=5)

    def generate_input_grid(self):
        """Generează gridul pentru introducere date"""
        rows = self.rows_var.get()
        cols = self.cols_var.get()

        for widget in self.input_grid.winfo_children():
            widget.destroy()

        # Antet coloane
        for j in range(cols):
            tk.Label(self.input_grid, text=f"M{j+1}").grid(row=0, column=j+1)

        # Costuri
        self.cost_entries = []
        for i in range(rows):
            tk.Label(self.input_grid, text=f"D{i+1}").grid(row=i+1, column=0)
            row_entries = []
            for j in range(cols):
                entry = tk.Entry(self.input_grid, width=5)
                entry.insert(0, "0")
                entry.grid(row=i+1, column=j+1)
                row_entries.append(entry)
            self.cost_entries.append(row_entries)

        # Disponibil
        tk.Label(self.input_grid, text="Disponibil").grid(row=rows+1, column=0)
        self.supply_entries = []
        for i in range(rows):
            entry = tk.Entry(self.input_grid, width=5)
            entry.insert(0, "0")
            entry.grid(row=rows+1, column=i+1)
            self.supply_entries.append(entry)

        # Necesar
        tk.Label(self.input_grid, text="Necesar").grid(row=rows+2, column=0)
        self.demand_entries = []
        for j in range(cols):
            entry = tk.Entry(self.input_grid, width=5)
            entry.insert(0, "0")
            entry.grid(row=rows+2, column=j+1)
            self.demand_entries.append(entry)

    def load_from_file(self):
        """Încarcă datele dintr-un fișier"""
        filepath = filedialog.askopenfilename(filetypes=[("Fișiere text", "*.txt")])
        if not filepath:
            return

        try:
            with open(filepath, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            # Citire nr. magazine și depozite
            cols = int(lines[0])
            rows = int(lines[1])

            # Citire costuri
            costs = []
            for i in range(2, 2+rows):
                costs.append(list(map(float, lines[i].split())))

            # Citire disponibil
            supply = list(map(float, lines[2+rows].split()))

            # Citire necesar
            demand = list(map(float, lines[3+rows].split()))

            # Actualizare UI
            self.rows_var.set(rows)
            self.cols_var.set(cols)
            self.generate_input_grid()

            # Completează valorile
            for i in range(rows):
                for j in range(cols):
                    self.cost_entries[i][j].delete(0, tk.END)
                    self.cost_entries[i][j].insert(0, str(costs[i][j]))

            for i in range(rows):
                self.supply_entries[i].delete(0, tk.END)
                self.supply_entries[i].insert(0, str(supply[i]))

            for j in range(cols):
                self.demand_entries[j].delete(0, tk.END)
                self.demand_entries[j].insert(0, str(demand[j]))

        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la încărcare fișier:\n{str(e)}")

    def load_test_data(self):
        """Încarcă date de test"""
        self.rows_var.set(3)
        self.cols_var.set(4)
        self.generate_input_grid()

        # Costuri
        test_costs = [
            [2, 3, 5, 1],
            [7, 3, 4, 6],
            [4, 1, 7, 2]
        ]

        # Disponibil
        test_supply = [8, 5, 9]

        # Necesar
        test_demand = [4, 4, 6, 7]

        # Completează valorile
        for i in range(3):
            for j in range(4):
                self.cost_entries[i][j].delete(0, tk.END)
                self.cost_entries[i][j].insert(0, str(test_costs[i][j]))

        for i in range(3):
            self.supply_entries[i].delete(0, tk.END)
            self.supply_entries[i].insert(0, str(test_supply[i]))

        for j in range(4):
            self.demand_entries[j].delete(0, tk.END)
            self.demand_entries[j].insert(0, str(test_demand[j]))

    def get_input_values(self):
        """Obține valorile introduse"""
        rows = self.rows_var.get()
        cols = self.cols_var.get()

        self.costs = []
        for i in range(rows):
            row = []
            for j in range(cols):
                val = self.cost_entries[i][j].get()
                row.append(float(val) if val else 0.0)
            self.costs.append(row)

        self.supply = []
        for i in range(rows):
            val = self.supply_entries[i].get()
            self.supply.append(float(val) if val else 0.0)

        self.demand = []
        for j in range(cols):
            val = self.demand_entries[j].get()
            self.demand.append(float(val) if val else 0.0)

    def check_balanced(self):
        """Verifică dacă problema este echilibrată"""
        total_supply = sum(self.supply)
        total_demand = sum(self.demand)
        return abs(total_supply - total_demand) < 1e-6

    def solve_problem(self):
        """Rezolvă problema de transport"""
        try:
            self.get_input_values()

            if not self.check_balanced():
                messagebox.showerror("Eroare", "Problema nu este echilibrată!")
                return

            self.iterations = []
            self.current_iter = 0
            self.solve_transport_problem()

            self.display_iteration(0)
            self.update_nav_buttons()

        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la rezolvare:\n{str(e)}")

    def solve_transport_problem(self):
        """Implementează algoritmul de rezolvare"""
        # Soluție inițială - colțul nord-vest
        solution = self.northwest_corner()
        self.store_iteration(solution, "Soluție inițială (metoda colțului nord-vest)")

        # Optimizare
        improved = True
        iteration = 1

        while improved and iteration < 100:  # Limită de siguranță
            improved, solution = self.improve_solution(solution, iteration)
            iteration += 1

    def northwest_corner(self):
        """Metoda colțului nord-vest"""
        rows = len(self.supply)
        cols = len(self.demand)
        solution = np.zeros((rows, cols))
        supply = self.supply.copy()
        demand = self.demand.copy()

        i = j = 0
        while i < rows and j < cols:
            amount = min(supply[i], demand[j])
            solution[i][j] = amount
            supply[i] -= amount
            demand[j] -= amount

            if supply[i] == 0:
                i += 1
            else:
                j += 1

        return solution

    def improve_solution(self, solution, iteration):
        """Îmbunătățește soluția folosind metoda stepping-stone"""
        # Calculează variabilele ui si vj
        u, v = self.calculate_dual_variables(solution)

        # Calculează costurile reduse
        delta, entering = self.calculate_reduced_costs(u, v)

        # Salvează iterația curentă
        self.store_iteration(solution, f"Iterația {iteration-1}", u, v, delta, entering)

        # Verifică optimalitate
        if entering is None:
            self.store_iteration(solution, "Soluție optimă găsită!")
            return False, solution

        # Găsește ciclu și ajustează soluția
        try:
            cycle = self.find_cycle(solution, entering)
            if not cycle:
                raise ValueError("Nu s-a găsit ciclu valid")

            # Determină theta și variabila de ieșire
            theta, leaving = self.find_theta(solution, cycle)

            # Actualizează soluția
            new_solution = solution.copy()
            sign = 1
            for (i, j) in cycle:
                new_solution[i][j] += sign * theta
                sign *= -1

           

            return True, new_solution

        except Exception as e:
            self.store_iteration(solution, f"Eroare la iterația {iteration-1}: {str(e)}")
            return False, solution

    def calculate_dual_variables(self, solution):
        """Calculează variabilele ui si vj"""
        rows = len(self.supply)
        cols = len(self.demand)
        u = [None] * rows
        v = [None] * cols
        u[0] = 0  # Alegere arbitrară

        # Listează celulele de bază
        basis = [(i, j) for i in range(rows) for j in range(cols) if solution[i][j] > 0]

        # Calculează u și v
        changed = True
        while changed:
            changed = False
            for i, j in basis:
                if u[i] is not None and v[j] is None:
                    v[j] = self.costs[i][j] - u[i]
                    changed = True
                elif v[j] is not None and u[i] is None:
                    u[i] = self.costs[i][j] - v[j]
                    changed = True
        # Ensure no None values remain
        for i in range(rows):
            if u[i] is None:
                 u[i] = 0  # Default value if not determined
    
        for j in range(cols):
            if v[j] is None:
                v[j] = 0  # Default value if not determined

        return u, v

    def calculate_reduced_costs(self, u, v):
        """Calculează costurile reduse și variabila de intrare"""
        rows = len(self.supply)
        cols = len(self.demand)
        delta = np.zeros((rows, cols))
        entering = None
        min_delta = 0

        for i in range(rows):
            for j in range(cols):
                delta[i][j] = self.costs[i][j] - (u[i] + v[j])
                if delta[i][j] < min_delta:
                    min_delta = delta[i][j]
                    entering = (i, j)

        return delta, entering

    def find_cycle(self, solution, entering):
        """Găsește corect ciclul pentru variabila de intrare"""
        rows = len(self.supply)
        cols = len(self.demand)
        
        # Construim matricea de bază
        basis = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                if solution[i][j] > 0:
                    basis[i][j] = 1
        
        # Adăugăm variabila de intrare
        i, j = entering
        basis[i][j] = 1
        
        # Inițializăm ciclul
        cycle = [entering]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Sus, jos, stânga, dreapta
        
        def search_path(current, prev=None):
            if len(cycle) > 3 and current == entering:
                return True
            
            for di, dj in directions:
                ni, nj = current[0] + di, current[1] + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    if (ni, nj) != prev and basis[ni][nj] == 1:
                        if (ni, nj) not in cycle or (ni, nj) == entering:
                            cycle.append((ni, nj))
                            if search_path((ni, nj), current):
                                return True
                            cycle.pop()
            return False
        
        if not search_path(entering):
            raise ValueError("Nu s-a găsit ciclu valid")
        
        # Eliminăm duplicatele și păstrăm doar ciclul complet
        clean_cycle = []
        seen = set()
        for cell in cycle:
            if cell not in seen:
                clean_cycle.append(cell)
                seen.add(cell)
        
        return clean_cycle

    def find_theta(self, solution, cycle):
        """Calculează corect theta pentru ciclul dat"""
        # Celulele unde scădem theta sunt cele de pe pozițiile impare (index 1, 3, 5...)
        leaving_candidates = cycle[1::2]
        
        # Găsim minimul dintre aceste celule
        theta = min(solution[i][j] for (i,j) in leaving_candidates)
        
        # Selectăm prima celulă cu valoarea minimă
        for (i,j) in leaving_candidates:
            if solution[i][j] == theta:
                return theta, (i,j)
        
        raise ValueError("Nu s-a găsit variabilă de ieșire validă")

    def store_iteration(self, solution, description, u=None, v=None, delta=None,
                      entering=None, adjustment_info=None):
        """Salvează o iterație pentru afișare ulterioară"""
        self.iterations.append({
            'solution': solution.copy(),
            'description': description,
            'u': u.copy() if u is not None else None,
            'v': v.copy() if v is not None else None,
            'delta': delta.copy() if delta is not None else None,
            'entering': entering,
            'adjustment_info': adjustment_info,
            'total_cost': self.calculate_total_cost(solution)
        })

    def calculate_total_cost(self, solution):
        """Calculează costul total al soluției"""
        total = 0
        for i in range(len(solution)):
            for j in range(len(solution[0])):
                total += solution[i][j] * self.costs[i][j]
        return total

    def display_iteration(self, index):
        """Afișează o iterație cu formatare îmbunătățită"""
        if index < 0 or index >= len(self.iterations):
            return
        
        iteration = self.iterations[index]
        self.solution_text.delete(1.0, tk.END)
        
        # Afișare descriere
        self.solution_text.insert(tk.END, f"=== {iteration['description']} ===\n\n", 'header')
        
        # Afișare soluție
        self.solution_text.insert(tk.END, "Soluție curentă:\n", 'subheader')
        self.display_solution(iteration['solution'])
        
        # Afișare cost total
        self.solution_text.insert(tk.END, f"\nCost total: {iteration['total_cost']:.2f}\n\n", 'bold')
        
        # Dacă e iterație de optimizare
        if iteration['u'] is not None:
            self.solution_text.insert(tk.END, "Variabile ui și vj:\n", 'subheader')
            
            # Afișare sistem de ecuații pentru variabilele ui și vj
            self.display_dual_system(iteration['solution'], iteration['u'], iteration['v'])
            
            # Afișare tabel c' = (u + v)
            self.solution_text.insert(tk.END, "\nTabel c' = u + v:\n", 'subheader')
            self.display_modified_costs(iteration['u'], iteration['v'])
            
            # Formatare mai frumoasă pentru u și v
            u_str = ", ".join([f"u{i+1}={x:.2f}" for i, x in enumerate(iteration['u'])])
            v_str = ", ".join([f"v{j+1}={x:.2f}" for j, x in enumerate(iteration['v'])])
            
            self.solution_text.insert(tk.END, f"\nRezumat variabile ui și vj:\n", 'bold')
            self.solution_text.insert(tk.END, f"u: [{u_str}]\n")
            self.solution_text.insert(tk.END, f"v: [{v_str}]\n\n")
            
            self.solution_text.insert(tk.END, "Tabel Δ:\n", 'subheader')
            self.display_delta(iteration['delta'])
            
           
        # Adăugăm explicația pentru trecerea la următoarea iterație
        if iteration['entering']:
            i, j = iteration['entering']
            delta_value = iteration['delta'][i][j]
            
            # Creăm un frame pentru explicație
            explanation_frame = tk.Frame(self.solution_text, bg='#fff3e0', padx=10, pady=10)
            
            # Titlu explicație
            tk.Label(explanation_frame, text="Concluzie și următorul pas:", 
                    font=('Arial', 10, 'bold'), bg='#fff3e0').pack(anchor='w')
            
            # Text explicație
            if delta_value < 0:
                explanation_text = (
                    f"Soluția nu este optimă deoarece există cel puțin o valoare in tabelul Δ negativă.\n"
                    f"Cea mai negativă valoare este Δ{i+1}{j+1} = {delta_value:.2f} < 0, "
                    f"deci variabila θ se va afla in celula ({i+1},{j+1}).\n"
                    f"Trecem la următoarea iterație pentru a îmbunătăți soluția."
                )
            else:
                explanation_text = (
                    "Soluția este optimă deoarece toate valorile din tabelul Δ sunt ≥ 0.\n"
                    "Nu mai sunt necesare alte iterații."
                )
            
            tk.Label(explanation_frame, text=explanation_text, 
                    font=('Arial', 9), bg='#fff3e0', justify=tk.LEFT,
                    wraplength=500).pack(anchor='w', pady=5)
            
            # Adăugăm frame-ul în text widget
            self.solution_text.window_create(tk.END, window=explanation_frame)
            self.solution_text.insert(tk.END, "\n\n")
            
            # Afișăm și variabila de intrare
            self.solution_text.insert(tk.END,
                f"Variabila θ în celula ({i+1},{j+1}) cu Δ = {delta_value:.2f}\n", 'highlight')
        else:
            # Creăm un frame pentru explicație de optimalitate
            explanation_frame = tk.Frame(self.solution_text, bg='#e8f5e9', padx=10, pady=10)
            
            # Titlu explicație
            tk.Label(explanation_frame, text="Concluzie:", 
                    font=('Arial', 10, 'bold'), bg='#e8f5e9').pack(anchor='w')
            
            # Text explicație
            explanation_text = (
                "Soluția este optimă deoarece toate valorile Δ sunt ≥ 0.\n"
                "Nu mai sunt necesare alte iterații."
            )
            
            tk.Label(explanation_frame, text=explanation_text, 
                    font=('Arial', 9), bg='#e8f5e9', justify=tk.LEFT,
                    wraplength=500).pack(anchor='w', pady=5)
            
            # Adăugăm frame-ul în text widget
            self.solution_text.window_create(tk.END, window=explanation_frame)
            self.solution_text.insert(tk.END, "\n\n")
        
       
    def display_dual_system(self, solution, u, v):
        """Afișează sistemul de ecuații pentru variabilele ui și vj"""
        rows = len(solution)
        cols = len(solution[0]) if rows > 0 else 0
        
        # Creăm un frame pentru sistemul de ecuații
        system_frame = tk.Frame(self.solution_text, padx=10, pady=5, bg='#f8f9fa')
        
        # Titlu pentru sistem
        tk.Label(system_frame, text="Sistem de ecuații pentru variabilele ui și vj:", 
                font=('Arial', 10, 'bold'), bg='#f8f9fa').pack(anchor='w', pady=(0, 5))
        
        # Listăm ecuațiile pentru celulele de bază
        equations_text = tk.Text(system_frame, height=min(10, rows*cols), width=50, 
                            font=('Consolas', 9), bg='#f8f9fa', relief=tk.FLAT)
        
        # Adăugăm ecuațiile
        eq_count = 0
        for i in range(rows):
            for j in range(cols):
                if solution[i][j] > 0:  # Celulă de bază
                    eq_count += 1
                    eq_text = f"Pentru x{i+1}{j+1} > 0: u{i+1} + v{j+1} = c{i+1}{j+1} = {self.costs[i][j]:.2f}"
                    check_text = f"   ({u[i]:.2f} + {v[j]:.2f} = {u[i] + v[j]:.2f} ≈ {self.costs[i][j]:.2f})"
                    
                    equations_text.insert(tk.END, f"{eq_text}\n")
                    equations_text.insert(tk.END, f"{check_text}\n")
        
        equations_text.config(state=tk.DISABLED)  # Facem textul read-only
        equations_text.pack(fill=tk.X)
        
        # Adăugăm o notă explicativă
        note_text = ("Notă: Pentru a determina variabilele ui și vj, se rezolvă sistemul de ecuații "
                    "format din celulele de bază (unde x > 0). Se fixează u₁ = 0 și se determină "
                    "restul variabilelor.")
        
        tk.Label(system_frame, text=note_text, font=('Arial', 8), fg='#555', 
                bg='#f8f9fa', wraplength=450, justify=tk.LEFT).pack(anchor='w', pady=5)
        
        # Adăugăm frame-ul în text widget
        self.solution_text.window_create(tk.END, window=system_frame)
        self.solution_text.insert(tk.END, "\n")

    def display_modified_costs(self, u, v):
        """Afișează tabelul  c' = u + v"""
        rows = len(self.costs)
        cols = len(self.costs[0]) if rows > 0 else 0
        
        # Creăm un frame pentru tabel
        table_frame = tk.Frame(self.solution_text)
        
        # Culori pentru tabel
        header_bg = '#4527a0'  # Violet închis
        header_fg = 'white'
        row_header_bg = '#b39ddb'  # Violet deschis
        
        # Stiluri pentru borduri
        border_style = {'relief': tk.RIDGE, 'borderwidth': 1, 'padx': 5, 'pady': 3}
        
        # Antet
        headers = ["c'ᵢⱼ = uᵢ+vⱼ"] + [f"M{j+1} (vⱼ={v[j]:.2f})" for j in range(cols)]
        for col, header in enumerate(headers):
            label = tk.Label(table_frame, text=header, 
                        font=('Arial', 9, 'bold'), bg=header_bg, fg=header_fg,
                        width=15, **border_style)
            label.grid(row=0, column=col, sticky='nsew')
        
        # Rânduri
        for i in range(rows):
            # Etichetă rând
            tk.Label(table_frame, text=f"D{i+1} (uᵢ={u[i]:.2f})", 
                    font=('Arial', 9, 'bold'), bg=row_header_bg, fg='black',
                    width=15, **border_style).grid(row=i+1, column=0, sticky='nsew')
            
            # Celule cu valori
            for j in range(cols):
                # Calculăm c'ᵢⱼ = cᵢⱼ - (uᵢ + vⱼ)
                original_cost = self.costs[i][j]
                modified_cost =  u[i] + v[j]
                
                # Alegem culoarea 
                bg_color = 'white'  
                fg_color = 'black'
               
                
                # Creăm un frame pentru celulă pentru a afișa mai multe informații
                cell_frame = tk.Frame(table_frame, bg=bg_color, **border_style)
                cell_frame.grid(row=i+1, column=j+1, sticky='nsew')
                
                # Afișăm costul modificat
                tk.Label(cell_frame, text=f"{modified_cost:.2f}", 
                    font=('Arial', 9, 'bold'), bg=bg_color, fg=fg_color).pack(pady=(2, 0))
                
                # Afișăm calculul
                tk.Label(cell_frame, text=f"({u[i]:.2f}+{v[j]:.2f})", 
                    font=('Arial', 7), bg=bg_color, fg='gray').pack(pady=(0, 2))
        
        # Adăugăm spațiu pentru tabel
        for i in range(rows+1):
            table_frame.grid_rowconfigure(i, minsize=40)
        for j in range(cols+1):
            table_frame.grid_columnconfigure(j, minsize=120)
        
        # Adăugăm tabelul în text widget
        self.solution_text.window_create(tk.END, window=table_frame)
        self.solution_text.insert(tk.END, "\n")

 
 
 
    def display_solution(self, solution):
        """Afișare îmbunătățită a soluției"""
        rows = len(solution)
        cols = len(solution[0]) if rows > 0 else 0
        
        # Creăm un frame pentru tabel
        table_frame = tk.Frame(self.solution_text)
        
        # Definim culori pentru tabel
        header_bg = '#3f51b5'  # Albastru închis
        header_fg = 'white'
        row_header_bg = '#bbdefb'  # Albastru deschis
        cell_bg = 'white'
        active_cell_bg = '#e3f2fd'  # Albastru foarte deschis
        border_color = '#90caf9'  # Albastru mediu
        
        # Antet cu stil
        headers = ["Depozit \\ Magazin"] + [f"M{j+1}" for j in range(cols)] + ["Disponibil"]
        for col, header in enumerate(headers):
            label = tk.Label(table_frame, text=header, relief=tk.RIDGE, width=10,
                            font=('Arial', 9, 'bold'), bg=header_bg, fg=header_fg,
                            borderwidth=1, padx=5, pady=3)
            label.grid(row=0, column=col, sticky='nsew')
        
        # Rânduri cu date
        for i in range(rows):
            # Etichetă rând
            tk.Label(table_frame, text=f"D{i+1}", relief=tk.RIDGE, width=10,
                    font=('Arial', 9, 'bold'), bg=row_header_bg, fg='black',
                    borderwidth=1, padx=5, pady=3).grid(row=i+1, column=0, sticky='nsew')
            
            # Celule cu valori
            for j in range(cols):
                val = solution[i][j]
                bg_color = active_cell_bg if val > 0 else cell_bg
                cost_text = f"{self.costs[i][j]:.1f}" if hasattr(self, 'costs') and i < len(self.costs) and j < len(self.costs[i]) else ""
                
                # Afișăm atât valoarea cât și costul
                cell_frame = tk.Frame(table_frame, relief=tk.RIDGE, borderwidth=1, bg=bg_color)
                cell_frame.grid(row=i+1, column=j+1, sticky='nsew')
                
                tk.Label(cell_frame, text=f"{val:.1f}", font=('Arial', 9, 'bold'), 
                        bg=bg_color).pack(pady=(3,0))
                tk.Label(cell_frame, text=f"(c={cost_text})", font=('Arial', 7), 
                        bg=bg_color, fg='gray').pack(pady=(0,3))
            
            # Disponibil
            tk.Label(table_frame, text=f"{self.supply[i]:.1f}", relief=tk.RIDGE, width=10,
                    font=('Arial', 9, 'bold'), bg=row_header_bg, 
                    borderwidth=1, padx=5, pady=3).grid(row=i+1, column=cols+1, sticky='nsew')
        
        # Rând necesar - cu text bold
        necesar_label = tk.Label(table_frame, text="Necesar", relief=tk.RIDGE, width=10,
                font=('Arial', 9, 'bold'), bg=row_header_bg, fg='black',
                borderwidth=1, padx=5, pady=3)
        necesar_label.grid(row=rows+1, column=0, sticky='nsew')
        
        for j in range(cols):
            tk.Label(table_frame, text=f"{self.demand[j]:.1f}", relief=tk.RIDGE, width=10,
                    font=('Arial', 9, 'bold'), bg=row_header_bg,
                    borderwidth=1, padx=5, pady=3).grid(row=rows+1, column=j+1, sticky='nsew')
        
        # Adăugăm tabelul în text widget
        self.solution_text.window_create(tk.END, window=table_frame)


    def display_delta(self, delta):
        """Afișează  tabel colorat și atractiv"""
        rows = len(delta)
        cols = len(delta[0]) if rows > 0 else 0
        
        # Creăm un frame pentru tabel
        table_frame = tk.Frame(self.solution_text)
        
        # Culori pentru tabel
        header_bg = '#673ab7'  # Violet
        header_fg = 'white'
        row_header_bg = '#d1c4e9'  # Violet deschis
        negative_bg = '#ffcdd2'  # Roșu deschis pentru valori negative
        positive_bg = '#dcedc8'  # Verde deschis pentru valori pozitive
        zero_bg = '#f5f5f5'  # Gri deschis pentru zero
        
        # Stiluri pentru borduri
        border_style = {'relief': tk.RIDGE, 'borderwidth': 1, 'padx': 5, 'pady': 3}
        
        # Antet
        headers = ["Depozit \\ Magazin"] + [f"M{j+1}" for j in range(cols)]
        for col, header in enumerate(headers):
            label = tk.Label(table_frame, text=header, 
                            font=('Arial', 9, 'bold'), bg=header_bg, fg=header_fg,
                            width=10, **border_style)
            label.grid(row=0, column=col, sticky='nsew')
        
        # Rânduri
        for i in range(rows):
            # Etichetă rând
            tk.Label(table_frame, text=f"D{i+1}", 
                    font=('Arial', 9, 'bold'), bg=row_header_bg, fg='black',
                    width=10, **border_style).grid(row=i+1, column=0, sticky='nsew')
            
            # Celule cu valori
            for j in range(cols):
                val = delta[i][j]
                # Alegem culoarea în funcție de valoare
                if val < -0.001:  # Folosim o mică toleranță pentru compararea cu zero
                    bg_color = negative_bg
                    fg_color = 'darkred'
                    font_style = ('Arial', 9, 'bold')  # Bold pentru valori negative (importante)
                elif val > 0.001:
                    bg_color = positive_bg
                    fg_color = 'darkgreen'
                    font_style = ('Arial', 9)
                else:
                    bg_color = zero_bg
                    fg_color = 'black'
                    font_style = ('Arial', 9)
                
                # Creăm un frame pentru celulă pentru a adăuga efecte vizuale
                cell_frame = tk.Frame(table_frame, bg=bg_color, **border_style)
                cell_frame.grid(row=i+1, column=j+1, sticky='nsew')
                
                # Adăugăm valoarea în celulă
                label = tk.Label(cell_frame, text=f"{val:.2f}", 
                            font=font_style, bg=bg_color, fg=fg_color)
                label.pack(expand=True, fill=tk.BOTH)
                
                # Adăugăm un indicator vizual pentru valori negative (potențiale îmbunătățiri)
                if val < -0.001:
                    indicator = tk.Label(cell_frame, text="↓", font=('Arial', 7), 
                                    bg=bg_color, fg='darkred')
                    indicator.place(relx=0.85, rely=0.15)
        
        # Adăugăm spațiu pentru tabel
        for i in range(rows+1):
            table_frame.grid_rowconfigure(i, minsize=30)
        for j in range(cols+1):
            table_frame.grid_columnconfigure(j, minsize=80)
        
        # Adăugăm o legendă
        legend_frame = tk.Frame(table_frame, bg='white', pady=5)
        legend_frame.grid(row=rows+1, column=0, columnspan=cols+1, sticky='w', pady=5)
        
        # Legendă pentru valori negative
        neg_sample = tk.Label(legend_frame, text="  ", bg=negative_bg, width=2, **border_style)
        neg_sample.pack(side=tk.LEFT, padx=2)
        tk.Label(legend_frame, text="Valori negative (potențiale îmbunătățiri)", 
                font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
        
        # Legendă pentru valori pozitive
        pos_sample = tk.Label(legend_frame, text="  ", bg=positive_bg, width=2, **border_style)
        pos_sample.pack(side=tk.LEFT, padx=2)
        tk.Label(legend_frame, text="Valori pozitive", 
                font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
        
        # Legendă pentru valori zero
        zero_sample = tk.Label(legend_frame, text="  ", bg=zero_bg, width=2, **border_style)
        zero_sample.pack(side=tk.LEFT, padx=2)
        tk.Label(legend_frame, text="Valori zero", 
                font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
        
        # Adăugăm tabelul în text widget
        self.solution_text.window_create(tk.END, window=table_frame)
        self.solution_text.insert(tk.END, "\n")


    def prev_step(self):
        """Navigare la pasul anterior"""
        if self.current_iter > 0:
            self.current_iter -= 1
            self.display_iteration(self.current_iter)
            self.update_nav_buttons()

    def next_step(self):
        """Navigare la pasul următor"""
        if self.current_iter < len(self.iterations) - 1:
            self.current_iter += 1
            self.display_iteration(self.current_iter)
            self.update_nav_buttons()

    def update_nav_buttons(self):
        """Actualizează starea butoanelor de navigare"""
        self.prev_btn.config(state=tk.NORMAL if self.current_iter > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_iter < len(self.iterations) - 1 else tk.DISABLED)
        self.iter_label.config(text=f"Pas: {self.current_iter+1}/{len(self.iterations)}")

if __name__ == "__main__":
    root = tk.Tk()
    root.option_add('*Font', 'Arial 10')
    root.option_add('*Label.Font', 'Arial 10 bold')
    root.option_add('*Button.Font', 'Arial 10')
    
    app = TransportSolver(root)
    root.mainloop()