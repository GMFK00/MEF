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

# --- DADOS DE ENTRADA DO MATERIAL ---
MATERIAIS = {
    "Madeira (Pinho)": 12e9,
    "MDF (Compensado)": 4e9,      
    "Aço Estrutural": 200e9,
    "Alumínio": 69e9,
    "Concreto": 30e9,
    "Nylon / Plástico Duro": 3e9, 
    "PVC Rígido": 2.5e9,          
    "Polímero ABS": 2e9,          
    "Borracha Dura": 0.1e9,       
    "Personalizado": 0
}

class MathReportWindow:
    """
    Janela de Relatório Didático.
    Mostra as equações que "estão por trás" do código (K.u = F).
    """
    def __init__(self, master, data):
        self.window = tk.Toplevel(master)
        self.window.title("Memorial de Cálculo Detalhado")
        self.window.geometry("900x800")

        container = ttk.Frame(self.window)
        canvas = tk.Canvas(container, bg="white")
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

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
        y = 0.95
        line_h = 0.03
        left_margin = 0.1

        def text(s, size=11, weight='normal', color='black', math=False):
            nonlocal y
            self.fig.text(left_margin, y, s, fontsize=size, weight=weight, color=color, va='top', fontname='DejaVu Sans')
            y -= line_h

        def skip(n=1):
            nonlocal y
            y -= (line_h * n)

        text("RELATÓRIO DE ANÁLISE ESTRUTURAL (MEF)", size=16, weight='bold', color='#2c3e50')
        skip()
        text(f"Projeto: Trampolim/Viga em Balanço", size=12, color='#7f8c8d')
        text("-" * 75)
        skip()

        text("1. PARÂMETROS DE ENTRADA", size=12, weight='bold', color='#c0392b')
        skip(0.5)
        text(f"• Dimensões: {d['L_total']}m (comp) x {d['W']}m (larg) x {d['H']}m (alt)")
        text(f"• Malha: {d['n_nos']} Nós e {d['n_elem']} Elementos")
        text(f"• Módulo de Elasticidade (E): {d['E']/1e9:.2f} GPa")
        text(f"• Carga Aplicada (Ponta): {d['Fz']} N")
        skip()

        text("2. FORMULAÇÃO MATEMÁTICA", size=12, weight='bold', color='#c0392b')
        skip(0.5)
        text("O sistema resolve a equação matricial de equilíbrio:", size=11)
        skip(0.5)
        text(r"$[K] \cdot \{u\} = \{F\}$", size=14, color='blue', math=True)
        skip(1.5)
        text("Onde a rigidez de cada barra é dada por:", size=11)
        skip(0.5)
        text(r"$k = \frac{EA}{L}$", size=14, math=True)
        skip(1.5)

        if d['ex_elem']:
            ex = d['ex_elem']
            text(f"3. CÁLCULO AMOSTRAL (Elemento #{ex['id']})", size=12, weight='bold', color='#c0392b')
            skip(0.5)
            text(f"Conectividade: Nó {ex['n1']} para Nó {ex['n2']}")
            text(f"Comprimento Inicial (L): {ex['L']:.4f} m")
            text(f"Rigidez Axial Calculada: {ex['k_val']:.2e} N/m")
            text(f"Cossenos Diretores (Vetor Unitário):")
            text(f"   cx={ex['cx']:.2f}, cy={ex['cy']:.2f}, cz={ex['cz']:.2f}")
        skip()

        text("4. RESULTADOS DA SIMULAÇÃO", size=12, weight='bold', color='#c0392b')
        skip(0.5)
        text(f"• Deslocamento Máximo (Flecha): {d['tip_disp']:.2f} mm", size=12, weight='bold')
        text(f"• Tensão Máxima Estimada: {d['max_stress']:.2f} MPa", size=12, weight='bold')
        skip(0.5)
        text("Interpretação:", size=11, color='#2c3e50')
        text("A tensão máxima ocorre próxima ao engaste (apoio), conforme")
        text("esperado para uma viga em balanço. A ponta apresenta deslocamento")
        text("máximo, mas tensão interna mínima (livre de momento fletor).")
        skip(2)
        text("Gerado automaticamente pelo Simulador v12.0", size=9, color='gray')


class TrampolimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador MEF - v12.0 (Corrigido)")
        self.root.geometry("1600x900")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Layout da Interface Gráfica
        self.frame_left = ttk.LabelFrame(root, text="1. Controles")
        self.frame_left.place(relx=0.01, rely=0.01, relwidth=0.20, relheight=0.98)

        self.frame_mid = ttk.LabelFrame(root, text="2. Status do Processamento")
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
        self.combo_mat.current(0)
        self.combo_mat.pack(fill="x", **pad)
        self.combo_mat.bind("<<ComboboxSelected>>", self.on_material_change)

        ttk.Label(self.frame_left, text="Módulo E (Pa):").pack(**pad)
        self.ent_E = ttk.Entry(self.frame_left)
        self.ent_E.insert(0, "12000000000.0") 
        self.ent_E.pack(fill="x", **pad)

        ttk.Label(self.frame_left, text="Força Ponta (N):").pack(**pad)
        self.ent_fz = ttk.Entry(self.frame_left); self.ent_fz.insert(0, "-2000")
        self.ent_fz.pack(fill="x", **pad)

        ttk.Label(self.frame_left, text="Exagero Visual:").pack(**pad)
        self.ent_scale = ttk.Entry(self.frame_left); self.ent_scale.insert(0, "1.0")
        self.ent_scale.pack(fill="x", **pad)
        
        self.show_faces_var = tk.BooleanVar(value=True)
        self.chk_faces = ttk.Checkbutton(self.frame_left, text="Mostrar Faces", variable=self.show_faces_var, command=self.solve_and_plot)
        self.chk_faces.pack(fill="x", **pad)

        self.scale_opacity = tk.Scale(self.frame_left, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, label="Opacidade")
        self.scale_opacity.set(0.6) 
        self.scale_opacity.pack(fill="x", **pad)
        self.scale_opacity.bind("<ButtonRelease-1>", lambda event: self.solve_and_plot())

        self.btn_gen = ttk.Button(self.frame_left, text="GERAR MALHA (VISUALIZAR)", command=self.generate_trampolin)
        self.btn_gen.pack(fill="x", padx=5, pady=10)

        self.btn_calc = ttk.Button(self.frame_left, text="CALCULAR DEFORMAÇÃO", command=self.solve_and_plot)
        self.btn_calc.pack(fill="x", padx=5, pady=5)
        
        self.btn_report = ttk.Button(self.frame_left, text="📄 RELATÓRIO MATEMÁTICO", command=self.open_report, state='disabled')
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
        --- ETAPA 1: PRÉ-PROCESSAMENTO (Discretização / Meshing) ---
        No MEF, não resolvemos equações diferenciais na estrutura inteira.
        Dividimos ela em pequenos pedaços chamados ELEMENTOS conectados por NÓS.
        """
        self.txt_log.config(state='normal'); self.txt_log.delete(1.0, tk.END); self.txt_log.config(state='disabled')
        self.log("Gerando malha geométrica...")
        self.btn_report.config(state='disabled')

        try:
            L = float(self.ent_len.get()); W = float(self.ent_width.get()); H = float(self.ent_height.get())
            n_seg = int(self.ent_seg.get()); n_fix = int(self.ent_fix_seg.get()) 
        except:
            L = 4.0; W = 0.5; H = 0.10; n_seg = 20; n_fix = 5

        if n_fix >= n_seg: n_fix = n_seg - 1
        if n_fix < 0: n_fix = 0
        dx = L / n_seg
        
        # Vetores que guardarão a geometria (Lista de coordenadas) e a topologia (Quem conecta com quem)
        nodes = []; elements = []; faces = [] 
        self.element_types = []; self.face_bar_indices = []

        for i in range(n_seg + 1):
            x = i * dx
            # Definindo 4 nós por seção para criar uma treliça espacial 3D
            nodes.append([x, 0, 0]); nodes.append([x, W, 0])
            nodes.append([x, 0, H]); nodes.append([x, W, H])
            
            if i > 0:
                base = (i - 1) * 4; curr = i * 4
                s_idx = len(elements)
                
                # Criando os Elementos (As barras da treliça)
                # Conectamos o nó 'base' (anterior) ao nó 'curr' (atual)
                elements.append((base+0, curr+0)); self.element_types.append('long')
                elements.append((base+1, curr+1)); self.element_types.append('long')
                elements.append((base+2, curr+2)); self.element_types.append('long')
                elements.append((base+3, curr+3)); self.element_types.append('long')
                
                # Barras diagonais para travamento (Evita mecanismo instável)
                diags = [(base+0, curr+2), (base+2, curr+0), (base+1, curr+3), (base+3, curr+1),
                         (base+2, curr+3), (base+3, curr+2), (base+0, curr+1), (base+1, curr+0)]
                for d in diags: elements.append(d); self.element_types.append('diag')

                # Apenas para visualização (desenhar faces coloridas depois)
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
        
        # Identificando os Nós de Contorno (Onde U = 0)
        self.fixed_nodes = []
        for i in range(n_fix + 1):
            base_idx = i * 4
            self.fixed_nodes.extend([base_idx, base_idx+1, base_idx+2, base_idx+3])
        
        # Identificando os Nós de Carga (Onde aplicaremos F)
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

    def solve_and_plot(self):
        """
        --- ETAPA 2: O SOLVER (Processamento) ---
        MEF Linear. Resolvemos [K]{u} = {F}.
        """
        try:
            self.log("Iniciando cálculos...")
            Fz = float(self.ent_fz.get())
            E = float(self.ent_E.get())
            scale = float(self.ent_scale.get())
            opacity = float(self.scale_opacity.get())
            A = 0.002 

            # GL = Graus de Liberdade. Em 3D, temos 3 por nó (X, Y, Z).
            num_nodes = len(self.nodes)
            dof = 3 * num_nodes
            
            # --- INICIALIZAÇÃO DA MATRIZ DE RIGIDEZ GLOBAL [K] ---
            # Matriz Quadrada (DOF x DOF), inicialmente vazia.
            K = np.zeros((dof, dof))
            F = np.zeros(dof)

            # --- APLICAÇÃO DO VETOR DE CARGAS {F} ---
            # Distribuímos a carga total entre os 4 nós da ponta.
            force_per_node = Fz / 4
            for node in self.load_nodes:
                # O índice do grau de liberdade Z é: 3*nó + 2
                idx = 3 * node + 2 
                F[idx] = force_per_node

            self.log("Montando Matriz de Rigidez...")
            
            ex_elem_data = None
            
            # --- LOOP DE MONTAGEM DOS ELEMENTOS ---
            # Para cada barra (elemento), calculamos sua rigidez e somamos na matriz global.
            for idx_elem, (n1, n2) in enumerate(self.elements):
                xyz1 = self.nodes[n1]; xyz2 = self.nodes[n2]
                vec = xyz2 - xyz1
                L = np.linalg.norm(vec)
                if L == 0: continue
                
                # 1. Matriz de Transformação (Cossenos Diretores)
                # Em aula: T = [cx, cy, cz ...]
                # Define a direção da barra no espaço global.
                cx, cy, cz = vec / L
                
                # 2. Rigidez Axial Local (k_local)
                # Em aula: k = (E*A)/L
                k_axial = (E * A / L)
                
                # Dados para o relatório
                if ex_elem_data is None and self.element_types[idx_elem] == 'long':
                    ex_elem_data = {
                        'id': idx_elem, 'n1': n1, 'n2': n2,
                        'L': L, 'k_val': k_axial,
                        'cx': cx, 'cy': cy, 'cz': cz
                    }

                # 3. Matriz de Rigidez do Elemento em Coordenadas GLOBAIS
                # Em aula, isso equivale a fazer: K_elem = [T]^T * [k_local] * [T]
                # Aqui usamos produto externo (outer product) para otimizar e gerar a matriz 3x3 de sub-blocos.
                m = np.outer([cx, cy, cz], [cx, cy, cz])
                
                # Monta a matriz 6x6 do elemento (Nó 1 e Nó 2)
                # [[ m, -m],
                #  [-m,  m]]
                k_matrix = np.block([[ m, -m], [-m,  m]]) * k_axial
                
                # 4. Assembly (Montagem na Matriz Global)
                # Mapeamos os índices locais (0 a 5) para os índices globais (DOF dos nós)
                indices = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
                
                for i in range(6):
                    for j in range(6):
                        # Soma a rigidez do elemento na posição correspondente da matriz global
                        K[indices[i], indices[j]] += k_matrix[i, j]

            # --- APLICAÇÃO DAS CONDIÇÕES DE CONTORNO (Boundary Conditions) ---
            # Em aula: "Cortar linhas e colunas" dos apoios.
            # No código: Zeramos a linha e coluna e colocamos 1 na diagonal.
            # Isso força a equação: 1 * u_i = 0  => Deslocamento nulo.
            fixed_dofs = []
            for node in self.fixed_nodes:
                fixed_dofs.extend([3*node, 3*node+1, 3*node+2])
            for d in fixed_dofs:
                K[d, :] = 0; K[:, d] = 0; K[d, d] = 1; F[d] = 0

            self.log("Resolvendo sistema linear...")
            # --- SOLUÇÃO DO SISTEMA ---
            # Resolvemos [K]{u} = {F} para encontrar os deslocamentos {u}
            # É aqui que usamos Álgebra Linear (Eliminação Gaussiana / Fatoração LU implícita)
            U = np.linalg.solve(K, F)

            displacements = U.reshape((num_nodes, 3))
            
            # Somamos o deslocamento à posição original para ter a "Forma Deformada"
            # O 'scale' é apenas um exagero visual para vermos o movimento na tela.
            deformed_nodes = self.nodes + displacements * scale
            
            # --- PÓS-PROCESSAMENTO ---
            # Cálculo das Tensões (Stress Recovery) a partir dos deslocamentos
            self.log("Calculando tensões...")
            element_stresses = []
            long_stresses = []
            
            for idx, (n1, n2) in enumerate(self.elements):
                p1 = self.nodes[n1]; p2 = self.nodes[n2]
                u1 = displacements[n1]; u2 = displacements[n2]
                
                vec_orig = p2 - p1
                L_orig = np.linalg.norm(vec_orig)
                
                # Deslocamento relativo
                delta_u = u2 - u1
                
                # Projeção do deslocamento no eixo da barra (Deformação Axial)
                unit_vec = vec_orig / L_orig
                axial_deformation = np.dot(delta_u, unit_vec)
                
                # Strain (Deformação Específica) = delta_L / L
                strain = axial_deformation / L_orig
                
                # Stress (Tensão) = E * Strain (Lei de Hooke)
                stress = abs(strain * E)
                
                element_stresses.append(stress)
                if self.element_types[idx] == 'long': long_stresses.append(stress)

            if long_stresses:
                max_stress = max(long_stresses)
            else:
                max_stress = max(element_stresses) if element_stresses else 1
            if max_stress == 0: max_stress = 1
            
            tip_disp = np.mean(displacements[self.load_nodes, 2]) * 1000 
            
            self.last_calc_data = {
                'n_nos': num_nodes, 'n_elem': len(self.elements),
                'E': E, 'Fz': Fz, 'A': A,
                'L_total': float(self.ent_len.get()), 'W': float(self.ent_width.get()), 'H': float(self.ent_height.get()),
                'tip_disp': tip_disp, 'max_stress': max_stress/1e6,
                'ex_elem': ex_elem_data
            }
            self.btn_report.config(state='normal')
            self.log(f"Pronto! Flecha: {tip_disp:.2f}mm")

            # --- RENDERIZAÇÃO GRÁFICA ---
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