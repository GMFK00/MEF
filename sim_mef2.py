import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import sys

# --- MATERIAIS PARA ENGENHARIA AVANÇADA ---
# Incluímos materiais como Titânio e compósitos pois este solver
# suporta grandes deslocamentos, típicos de estruturas de alta performance.
MATERIAIS = {
    "Aço Estrutural": 200e9,
    "Alumínio": 69e9,
    "Titânio": 116e9,
    "Madeira (Pinho)": 12e9,
    "MDF (Compensado)": 4e9,      
    "Concreto": 30e9,
    "Nylon / Plástico Duro": 3e9, 
    "PVC Rígido": 2.5e9,          
    "Polímero ABS": 2e9,          
    "Borracha Dura": 0.1e9,       
    "Personalizado": 0
}

class MathReportWindow:
    """
    Relatório Técnico.
    Foca nos resultados finais de engenharia e na convergência do método,
    omitindo a dedução matemática básica.
    """
    def __init__(self, master, data):
        self.window = tk.Toplevel(master)
        self.window.title("Memorial de Cálculo")
        self.window.geometry("900x800")
        
        container = ttk.Frame(self.window)
        canvas = tk.Canvas(container, bg="white")
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        container.pack(fill="both", expand=True)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.fig = plt.figure(figsize=(8.5, 12)) 
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.render_report(data)
        
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=scrollable_frame)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(padx=20, pady=20)

    def render_report(self, d):
        y = 0.95; line_h = 0.03; left_margin = 0.1
        def text(s, size=11, weight='normal', color='black', math=False):
            nonlocal y
            self.fig.text(left_margin, y, s, fontsize=size, weight=weight, color=color, va='top', fontname='DejaVu Sans')
            y -= line_h
        def skip(n=1): nonlocal y; y -= (line_h * n)

        text("RELATÓRIO TÉCNICO (MEF)", size=16, weight='bold', color='#2c3e50')
        skip()
        mode = f"Não-Linear ({d.get('steps', 0)} passos)" if d['nonlinear'] else "Linear"
        text(f"Método: {mode}", size=12, color='blue')
        text("-" * 75)
        skip()
        text("1. PARÂMETROS", size=12, weight='bold', color='#c0392b')
        skip(0.5)
        text(f"• Dimensões: {d['L_total']}x{d['W']}x{d['H']} m")
        text(f"• Material E: {d['E']/1e9:.2f} GPa")
        text(f"• Carga Total: {d['Fz']} N")
        skip()
        text("2. RESULTADOS", size=12, weight='bold', color='#c0392b')
        skip(0.5)
        text(f"• Flecha na Ponta: {d['tip_disp']:.2f} mm", size=12, weight='bold')
        text(f"• Tensão Máxima: {d['max_stress']:.2f} MPa", size=12, weight='bold')


class TrampolimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador MEF - v15.0 (Estável)")
        self.root.geometry("1600x900")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Layout da GUI
        self.frame_left = ttk.LabelFrame(root, text="1. Controles")
        self.frame_left.place(relx=0.01, rely=0.01, relwidth=0.20, relheight=0.98)

        self.frame_mid = ttk.LabelFrame(root, text="2. Processamento")
        self.frame_mid.place(relx=0.22, rely=0.01, relwidth=0.20, relheight=0.98)

        self.frame_right = ttk.LabelFrame(root, text="3. Visualização 3D")
        self.frame_right.place(relx=0.43, rely=0.01, relwidth=0.56, relheight=0.98)

        self._init_inputs()
        self._init_log()
        self._init_plot()
        
        self.zoom_scale = 1.0
        self.last_calc_data = None 
        self.generate_trampolin()

    def _init_inputs(self):
        pad = {'padx': 5, 'pady': 5}
        
        ttk.Label(self.frame_left, text="Comprimento (m):").pack(**pad)
        self.ent_len = ttk.Entry(self.frame_left); self.ent_len.insert(0, "4.0")
        self.ent_len.pack(fill="x", **pad)

        ttk.Label(self.frame_left, text="Largura (m):").pack(**pad)
        self.ent_width = ttk.Entry(self.frame_left); self.ent_width.insert(0, "0.5")
        self.ent_width.pack(fill="x", **pad)

        ttk.Label(self.frame_left, text="Espessura (m):").pack(**pad)
        self.ent_height = ttk.Entry(self.frame_left); self.ent_height.insert(0, "0.10")
        self.ent_height.pack(fill="x", **pad)

        ttk.Separator(self.frame_left, orient='horizontal').pack(fill='x', pady=5)

        ttk.Label(self.frame_left, text="Segmentos Totais:").pack(**pad)
        self.ent_seg = ttk.Entry(self.frame_left); self.ent_seg.insert(0, "20")
        self.ent_seg.pack(fill="x", **pad)
        
        ttk.Label(self.frame_left, text="Segmentos Fixos:").pack(**pad)
        self.ent_fix_seg = ttk.Entry(self.frame_left); self.ent_fix_seg.insert(0, "5")
        self.ent_fix_seg.pack(fill="x", **pad)

        ttk.Separator(self.frame_left, orient='horizontal').pack(fill='x', pady=10)

        ttk.Label(self.frame_left, text="Material:", foreground="blue").pack(**pad)
        self.combo_mat = ttk.Combobox(self.frame_left, values=list(MATERIAIS.keys()), state="readonly")
        self.combo_mat.current(3)
        self.combo_mat.pack(fill="x", **pad)
        self.combo_mat.bind("<<ComboboxSelected>>", self.on_material_change)

        ttk.Label(self.frame_left, text="Módulo E (Pa):").pack(**pad)
        self.ent_E = ttk.Entry(self.frame_left)
        self.ent_E.insert(0, "12000000000.0") 
        self.ent_E.pack(fill="x", **pad)

        ttk.Label(self.frame_left, text="Força Ponta (N):").pack(**pad)
        self.ent_fz = ttk.Entry(self.frame_left); self.ent_fz.insert(0, "-15000")
        self.ent_fz.pack(fill="x", **pad)

        ttk.Label(self.frame_left, text="Exagero Visual (Linear):").pack(**pad)
        self.ent_scale = ttk.Entry(self.frame_left); self.ent_scale.insert(0, "1.0")
        self.ent_scale.pack(fill="x", **pad)
        
        # --- CONFIGURAÇÕES DE NÃO-LINEARIDADE ---
        # A principal diferença na interface: Controle de iterações (steps)
        self.nonlinear_var = tk.BooleanVar(value=True)
        self.chk_nonlinear = ttk.Checkbutton(self.frame_left, text="Modo Grandes Deformações", variable=self.nonlinear_var)
        self.chk_nonlinear.pack(fill="x", pady=(10,0), padx=5)

        frm_steps = ttk.Frame(self.frame_left)
        frm_steps.pack(fill="x", padx=5, pady=2)
        ttk.Label(frm_steps, text="Passos:").pack(side="left")
        self.ent_steps = ttk.Entry(frm_steps, width=5)
        self.ent_steps.insert(0, "20")
        self.ent_steps.pack(side="right")
        
        # Barra de progresso para acompanhar o solver incremental
        self.progress = ttk.Progressbar(self.frame_left, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(fill="x", padx=5, pady=5)

        # --- Visualização ---
        self.show_faces_var = tk.BooleanVar(value=True)
        self.chk_faces = ttk.Checkbutton(self.frame_left, text="Mostrar Faces", variable=self.show_faces_var, command=self.solve_and_plot)
        self.chk_faces.pack(fill="x", pady=5, padx=5)

        self.scale_opacity = tk.Scale(self.frame_left, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, label="Opacidade")
        self.scale_opacity.set(0.6) 
        self.scale_opacity.pack(fill="x", **pad)
        self.scale_opacity.bind("<ButtonRelease-1>", lambda event: self.solve_and_plot())

        self.btn_gen = ttk.Button(self.frame_left, text="GERAR MALHA", command=self.generate_trampolin)
        self.btn_gen.pack(fill="x", padx=5, pady=10)

        self.btn_calc = ttk.Button(self.frame_left, text="CALCULAR", command=self.solve_and_plot)
        self.btn_calc.pack(fill="x", padx=5, pady=5)
        
        self.btn_report = ttk.Button(self.frame_left, text="📄 RELATÓRIO", command=self.open_report, state='disabled')
        self.btn_report.pack(fill="x", padx=5, pady=5)

    def _init_log(self):
        self.txt_log = scrolledtext.ScrolledText(self.frame_mid, state='disabled', font=("Consolas", 10))
        self.txt_log.pack(expand=True, fill='both', padx=5, pady=5)

    def _init_plot(self):
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand=True, fill='both')
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

    def on_material_change(self, event):
        nome = self.combo_mat.get()
        valor = MATERIAIS[nome]
        if valor > 0: 
            self.ent_E.delete(0, tk.END)
            self.ent_E.insert(0, str(valor))

    def log(self, text):
        self.txt_log.config(state='normal')
        self.txt_log.insert(tk.END, f"> {text}\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state='disabled')
        # Força atualização da interface (Evita travar durante o loop de cálculo)
        self.root.update_idletasks()

    def on_closing(self):
        self.root.quit()
        self.root.destroy()
        sys.exit()

    def on_scroll(self, event):
        base_scale = 1.1
        if event.button == 'up': self.zoom_scale /= base_scale
        elif event.button == 'down': self.zoom_scale *= base_scale
        self.update_plot_limits(self.nodes) 
        self.canvas.draw_idle()

    def open_report(self):
        if self.last_calc_data:
            MathReportWindow(self.root, self.last_calc_data)

    def generate_trampolin(self):
        """
        Gera a Discretização (Malha) inicial.
        Igual ao método linear: definimos a topologia (quem conecta com quem).
        """
        self.txt_log.config(state='normal'); self.txt_log.delete(1.0, tk.END); self.txt_log.config(state='disabled')
        self.log("Gerando malha geométrica...")
        self.btn_report.config(state='disabled')
        self.progress['value'] = 0

        try:
            L = float(self.ent_len.get()); W = float(self.ent_width.get()); H = float(self.ent_height.get())
            n_seg = int(self.ent_seg.get()); n_fix = int(self.ent_fix_seg.get()) 
        except:
            L = 4.0; W = 0.5; H = 0.10; n_seg = 20; n_fix = 5

        if n_fix >= n_seg: n_fix = n_seg - 1
        if n_fix < 0: n_fix = 0
        dx = L / n_seg
        
        nodes = []; elements = []; faces = [] 
        self.element_types = []; self.face_bar_indices = []

        for i in range(n_seg + 1):
            x = i * dx
            nodes.append([x, 0, 0]); nodes.append([x, W, 0])
            nodes.append([x, 0, H]); nodes.append([x, W, H])
            
            if i > 0:
                base = (i - 1) * 4; curr = i * 4
                s_idx = len(elements)
                
                elements.append((base+0, curr+0)); self.element_types.append('long')
                elements.append((base+1, curr+1)); self.element_types.append('long')
                elements.append((base+2, curr+2)); self.element_types.append('long')
                elements.append((base+3, curr+3)); self.element_types.append('long')
                
                diags = [(base+0, curr+2), (base+2, curr+0), (base+1, curr+3), (base+3, curr+1),
                         (base+2, curr+3), (base+3, curr+2), (base+0, curr+1), (base+1, curr+0)]
                for d in diags: elements.append(d); self.element_types.append('diag')

                faces.append([base+2, base+3, curr+3, curr+2]) 
                self.face_bar_indices.append([s_idx+2, s_idx+3])
                faces.append([base+0, base+1, curr+1, curr+0]) 
                self.face_bar_indices.append([s_idx+0, s_idx+1])
                faces.append([base+0, base+2, curr+2, curr+0]) 
                self.face_bar_indices.append([s_idx+0, s_idx+2])
                faces.append([base+1, base+3, curr+3, curr+1]) 
                self.face_bar_indices.append([s_idx+1, s_idx+3])

            idx = i * 4
            trans = [(idx+0, idx+1), (idx+2, idx+3), (idx+0, idx+2), (idx+1, idx+3)]
            for t in trans: elements.append(t); self.element_types.append('trans')

            if i == 0 or i == n_seg: 
                 faces.append([idx+0, idx+1, idx+3, idx+2]); self.face_bar_indices.append([]) 

        self.nodes = np.array(nodes, dtype=float)
        self.elements = elements
        self.faces = faces 
        
        self.fixed_nodes = []
        for i in range(n_fix + 1):
            base_idx = i * 4
            self.fixed_nodes.extend([base_idx, base_idx+1, base_idx+2, base_idx+3])
        
        last = len(nodes)
        self.load_nodes = [last-1, last-2, last-3, last-4] 

        self.log(f"Malha: {len(nodes)} nós, {len(elements)} elementos.")
        self.ax.clear()
        self.plot_structure(self.nodes, color='gray', linestyle='--', alpha=0.5)
        
        try:
            fix_dist = n_fix * dx
            self.ax.plot([fix_dist, fix_dist], [-0.1, W+0.1], [0, 0], color='black', linewidth=4, zorder=20, label="Apoio")
        except: pass
        
        self.canvas.draw()

    def plot_structure(self, nodes, color, linestyle, alpha=1.0):
        for n1, n2 in self.elements:
            p1 = nodes[n1]; p2 = nodes[n2]
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linestyle=linestyle, alpha=alpha)
        self.update_plot_limits(nodes)

    def assemble_and_solve(self, current_nodes, force_vector, E, A, constraints):
        """
        --- KERNEL DO MEF (Montagem e Solução) ---
        Este método é chamado REPETIDAS VEZES no loop não-linear.
        
        Diferença crucial:
        Recebe 'current_nodes' (geometria ATUAL), não a inicial.
        Isso significa que a matriz [K] muda conforme a estrutura se deforma.
        """
        num_nodes = len(current_nodes)
        dof = 3 * num_nodes
        K = np.zeros((dof, dof))
        ex_elem_data = None
        
        # Montagem da Matriz de Rigidez Tangente
        for idx_elem, (n1, n2) in enumerate(self.elements):
            # Vetor da barra na configuração DEFORMADA
            vec = current_nodes[n2] - current_nodes[n1]
            L = np.linalg.norm(vec)
            if L == 0: continue
            
            # Os cossenos diretores (rotação) mudam a cada passo
            cx, cy, cz = vec / L
            k_axial = (E * A / L)
            
            if ex_elem_data is None and self.element_types[idx_elem] == 'long':
                ex_elem_data = {'id': idx_elem, 'n1': n1, 'n2': n2, 'L': L, 'k_val': k_axial, 'cx': cx, 'cy': cy, 'cz': cz}

            # Matriz de Rigidez em coordenadas globais (Atualizada)
            m = np.outer([cx, cy, cz], [cx, cy, cz])
            k_matrix = np.block([[ m, -m], [-m,  m]]) * k_axial
            indices = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
            
            for i in range(6):
                for j in range(6):
                    K[indices[i], indices[j]] += k_matrix[i, j]

        # Aplica condições de contorno (Zera linhas/colunas dos nós fixos)
        for d in constraints:
            K[d, :] = 0; K[:, d] = 0; K[d, d] = 1; force_vector[d] = 0

        # Resolve Ku = F para este incremento de carga
        U = np.linalg.solve(K, force_vector)
        return U, ex_elem_data

    def solve_and_plot(self):
        """
        --- LOOP DE SOLUÇÃO ---
        """
        try:
            self.log("Iniciando cálculos...")
            Fz = float(self.ent_fz.get())
            E = float(self.ent_E.get())
            vis_scale = float(self.ent_scale.get())
            opacity = float(self.scale_opacity.get())
            nonlinear = self.nonlinear_var.get()
            
            try: steps = int(self.ent_steps.get())
            except: steps = 25

            A = 0.002 
            num_nodes = len(self.nodes)
            dof = 3 * num_nodes
            
            fixed_dofs = []
            for node in self.fixed_nodes:
                fixed_dofs.extend([3*node, 3*node+1, 3*node+2])

            # Vetor de Força Total
            F_total = np.zeros(dof)
            force_per_node = Fz / 4
            for node in self.load_nodes:
                idx = 3 * node + 2 
                F_total[idx] = force_per_node

            # 'deformed_nodes' começa igual à geometria inicial, mas vai mudar (evoluir)
            deformed_nodes = np.copy(self.nodes)
            ex_elem_data = None
            
            if nonlinear:
                # --- MÉTODO INCREMENTAL (Lagrangiano Atualizado) ---
                # Em vez de aplicar F de uma vez, dividimos em 'steps'.
                # A cada passo, atualizamos a geometria. Isso captura o efeito de
                # "Geometric Stiffness" (Endurecimento Geométrico).
                self.log(f"Modo Não-Linear ({steps} passos)...")
                
                F_step = F_total / steps # Incremento de carga
                vis_scale = 1.0 # Em não-linear, deformação é real, não escalada.
                self.progress['maximum'] = steps
                
                for s in range(steps):
                    # Chama o Solver passando a geometria do passo anterior
                    U_step, info = self.assemble_and_solve(deformed_nodes, np.copy(F_step), E, A, fixed_dofs)
                    if s==0: ex_elem_data = info
                    
                    # ATUALIZAÇÃO DA GEOMETRIA (não-linear)
                    # x_novo = x_antigo + du
                    disp_matrix = U_step.reshape((num_nodes, 3))
                    deformed_nodes += disp_matrix
                    
                    self.progress['value'] = s + 1
                    self.root.update() 
            else:
                # --- MÉTODO LINEAR (Padrão) ---
                # Assume rigidez constante K0 baseada na geometria inicial.
                self.log("Modo Linear...")
                self.progress['maximum'] = 100
                self.progress['value'] = 50
                self.root.update()
                U, info = self.assemble_and_solve(self.nodes, np.copy(F_total), E, A, fixed_dofs)
                ex_elem_data = info
                displacements = U.reshape((num_nodes, 3))
                deformed_nodes = self.nodes + displacements * vis_scale
                self.progress['value'] = 100

            # --- CÁLCULO DE TENSÕES (Pós-Processamento) ---
            self.log("Calculando tensões finais...")
            element_stresses = []
            long_stresses = []
            
            for idx, (n1, n2) in enumerate(self.elements):
                # Comprimento na configuração de referência (L0)
                vec_orig = self.nodes[n2] - self.nodes[n1]
                L_orig = np.linalg.norm(vec_orig)
                
                # Comprimento na configuração final deformada (Lf)
                vec_final = deformed_nodes[n2] - deformed_nodes[n1]
                L_final = np.linalg.norm(vec_final)
                
                # CÁLCULO DO "TRUE STRAIN" (Deformação Logarítmica)
                # Em grandes deformações, (L-L0)/L0 é impreciso.
                # Usa-se ln(Lf/L0) ou deformação de Green-Lagrange.
                if L_orig > 0 and L_final > 0:
                    strain = np.log(L_final / L_orig)
                else:
                    strain = 0
                    
                stress = abs(strain * E)
                element_stresses.append(stress)
                if self.element_types[idx] == 'long': long_stresses.append(stress)

            if long_stresses: max_stress = max(long_stresses)
            else: max_stress = max(element_stresses) if element_stresses else 1
            if max_stress == 0: max_stress = 1
            
            # Flecha real calculada pela diferença de Z
            orig_z = np.mean(self.nodes[self.load_nodes, 2])
            final_z = np.mean(deformed_nodes[self.load_nodes, 2])
            tip_disp = (final_z - orig_z) * 1000 
            
            self.last_calc_data = {
                'n_nos': num_nodes, 'n_elem': len(self.elements),
                'E': E, 'Fz': Fz, 'A': A, 'steps': steps,
                'L_total': float(self.ent_len.get()), 'W': float(self.ent_width.get()), 'H': float(self.ent_height.get()),
                'tip_disp': tip_disp, 'max_stress': max_stress/1e6,
                'ex_elem': ex_elem_data, 'nonlinear': nonlinear
            }
            self.btn_report.config(state='normal')
            self.log(f"Pronto! Flecha: {tip_disp:.2f}mm")

            # --- RENDERIZAÇÃO ---
            norm = mcolors.Normalize(vmin=0, vmax=max_stress)
            cmap = plt.get_cmap('jet')
            self.ax.clear()
            
            if self.show_faces_var.get():
                face_coords = []; face_colors = []
                for idx_face, f_indices in enumerate(self.faces):
                    pts = deformed_nodes[f_indices]
                    face_coords.append(pts)
                    bars = self.face_bar_indices[idx_face]
                    if len(bars) > 0:
                        vals = [element_stresses[b] for b in bars]
                        avg_val = sum(vals)/len(vals)
                        if avg_val > max_stress: avg_val = max_stress
                        rgba = cmap(norm(avg_val))
                    else: rgba = cmap(0)
                    face_colors.append((rgba[0], rgba[1], rgba[2], opacity))
                
                poly = Poly3DCollection(face_coords, facecolors=face_colors, edgecolor=None, shade=False, zorder=2)
                self.ax.add_collection3d(poly)

            for idx, (n1, n2) in enumerate(self.elements):
                p1 = deformed_nodes[n1]; p2 = deformed_nodes[n2]
                val = element_stresses[idx]
                if val > max_stress: val = max_stress
                zo = 1 if self.show_faces_var.get() else 10
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=cmap(norm(val)), linewidth=2.0, zorder=zo)

            try:
                n_fix = int(self.ent_fix_seg.get()); L = float(self.ent_len.get())
                dx = L / int(self.ent_seg.get()); fix_dist = n_fix * dx
                W = float(self.ent_width.get())
                self.ax.plot([fix_dist, fix_dist], [-0.1, W+0.1], [0, 0], color='black', linewidth=4, zorder=20)
            except: pass

            self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
            self.update_plot_limits(deformed_nodes)
            self.canvas.draw()

        except Exception as e:
            self.log(f"Erro: {str(e)}")
            print(e)

    def update_plot_limits(self, nodes):
        try: W = float(self.ent_width.get())
        except: W = 0.5
        x_min, x_max = nodes[:,0].min(), nodes[:,0].max()
        y_min, y_max = nodes[:,1].min(), nodes[:,1].max()
        z_min, z_max = nodes[:,2].min(), nodes[:,2].max()
        center_x = (x_max + x_min) / 2; center_y = (y_max + y_min) / 2; center_z = (z_max + z_min) / 2
        range_x = x_max - x_min; range_z = z_max - z_min
        required_y_span = 9 * W 
        max_span = max(range_x, required_y_span, range_z)
        plot_span = max_span * 0.5 * self.zoom_scale
        self.ax.set_xlim(center_x - plot_span, center_x + plot_span)
        self.ax.set_ylim(center_y - plot_span, center_y + plot_span)
        self.ax.set_zlim(center_z - plot_span, center_z + plot_span)

if __name__ == "__main__":
    root = tk.Tk()
    app = TrampolimApp(root)
    root.mainloop()