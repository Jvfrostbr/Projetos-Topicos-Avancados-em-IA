import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import uuid

# Classes da Rede Neural
class Neuronio:
    def __init__(self, vies):
        self.id = str(uuid.uuid4())
        self.entrada = 0.0
        self.vies = vies
        self.saida = 0.0
        self.delta = 0.0

class Conexao:
    def __init__(self, peso=None, neuronio_origem=None, neuronio_destino=None):
        self.id = str(uuid.uuid4())
        self.peso = peso if peso is not None else np.random.uniform(-1, 1)
        self.neuronio_origem = neuronio_origem
        self.neuronio_destino = neuronio_destino

class Camada:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.neuronios = []
    
    def adicionar_neuronio(self, vies):
        neuronio = Neuronio(vies)
        self.neuronios.append(neuronio)
    
    def remover_neuronio(self, neuronio):
        self.neuronios.remove(neuronio)
    
    def get_neuronios(self):
        return self.neuronios

class RedeNeural:
    def __init__(self, func_ativacao="sigmoid"):
        self.camadas = []
        self.conexoes = []
        self.func_ativacao = func_ativacao

    def adicionar_camada(self, num_neuronios):
        camada = Camada()
        for _ in range(num_neuronios):
            vies = np.random.uniform(-1, 1)
            camada.adicionar_neuronio(vies)
        self.camadas.append(camada)

    def conectar_camadas(self, camada_origem, camada_destino):
        for neuronio_origem in camada_origem.get_neuronios():
            for neuronio_destino in camada_destino.get_neuronios():
                conexao = Conexao(neuronio_origem=neuronio_origem, neuronio_destino=neuronio_destino)
                self.conexoes.append(conexao)

    def ativacao(self, x):
        if self.func_ativacao == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.func_ativacao == "tanh":
            return np.tanh(x)
        elif self.func_ativacao == "relu":
            return np.maximum(0, x)
        else:
            raise ValueError("Função de ativação não suportada!")

    def derivada_ativacao(self, x):
        if self.func_ativacao == "sigmoid":
            return x * (1 - x)
        elif self.func_ativacao == "tanh":
            return 1 - x**2
        elif self.func_ativacao == "relu":
            return np.where(x > 0, 1, 0)
        else:
            raise ValueError("Função de ativação não suportada!")

    def feedforward(self, entradas):
        # Entradas na primeira camada
        for i, neuronio in enumerate(self.camadas[0].get_neuronios()):
            neuronio.saida = entradas[i]

        # Propaga valores
        for i in range(1, len(self.camadas)):
            for neuronio in self.camadas[i].get_neuronios():
                soma_ponderada = sum(
                    conexao.peso * conexao.neuronio_origem.saida
                    for conexao in self.conexoes
                    if conexao.neuronio_destino == neuronio
                )
                soma_ponderada += neuronio.vies
                neuronio.entrada = soma_ponderada
                neuronio.saida = self.ativacao(soma_ponderada)

        return [n.saida for n in self.camadas[-1].get_neuronios()]

    def backpropagation(self, entradas, saidas_esperadas, taxa_aprendizado):
        # Passo feedforward
        saidas_obtidas = self.feedforward(entradas)

        # Erro na camada de saída
        for i, neuronio in enumerate(self.camadas[-1].get_neuronios()):
            erro = saidas_esperadas[i] - saidas_obtidas[i]
            neuronio.delta = erro * self.derivada_ativacao(neuronio.saida)

        # Erro nas camadas ocultas
        for i in reversed(range(1, len(self.camadas) - 1)):
            for neuronio in self.camadas[i].get_neuronios():
                erro = sum(
                    conexao.peso * conexao.neuronio_destino.delta
                    for conexao in self.conexoes
                    if conexao.neuronio_origem == neuronio
                )
                neuronio.delta = erro * self.derivada_ativacao(neuronio.saida)

        # Atualização dos pesos
        for conexao in self.conexoes:
            gradiente = conexao.neuronio_destino.delta * conexao.neuronio_origem.saida
            conexao.peso += taxa_aprendizado * gradiente

        # Atualização dos vieses
        for camada in self.camadas[1:]:
            for neuronio in camada.get_neuronios():
                neuronio.vies += taxa_aprendizado * neuronio.delta

# Interface Gráfica
class NeuralGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MLP Interativa com CSV")
        self.root.geometry("1300x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.rede = RedeNeural()
        self.dataset = None
        self.entrada_cols = []
        self.saida_cols = []
        self.erros = []
        self.epoch = 0
        self.is_training = False
        self.stop_training = False

        # Frames
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.graph_frame = tk.Frame(root)
        self.graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.analysis_frame = tk.Frame(root)
        self.analysis_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # Controle CSV
        tk.Button(self.control_frame, text="Carregar CSV", command=self.load_csv).pack(pady=5, fill=tk.X)
        tk.Button(self.control_frame, text="Selecionar Entradas/Saídas", command=self.select_columns).pack(pady=5, fill=tk.X)
        
        # Status label
        self.status_var = tk.StringVar(value="Pronto")
        tk.Label(self.control_frame, textvariable=self.status_var, relief=tk.SUNKEN, bd=1).pack(pady=5, fill=tk.X)

        # Camadas
        tk.Label(self.control_frame, text="Configuração de Camadas:").pack(pady=(10, 5))
        self.camadas_frame = tk.Frame(self.control_frame)
        self.camadas_frame.pack(pady=5, fill=tk.X)

        self.add_hidden_btn = tk.Button(self.control_frame, text="Adicionar camada oculta", command=self.add_hidden_layer)
        self.add_hidden_btn.pack(pady=5, fill=tk.X)

        self.remove_hidden_btn = tk.Button(self.control_frame, text="Remover camada oculta", command=self.remove_hidden_layer)
        self.remove_hidden_btn.pack(pady=5, fill=tk.X)

        # Função de ativação
        tk.Label(self.control_frame, text="Função de Ativação:").pack(pady=(10, 5))
        self.activation_var = tk.StringVar(value="sigmoid")
        activation_menu = tk.OptionMenu(self.control_frame, self.activation_var, "sigmoid", "tanh", "relu")
        activation_menu.pack(pady=5, fill=tk.X)

        # Feedforward
        tk.Button(self.control_frame, text="Rodar Feedforward", command=self.feedforward).pack(pady=10, fill=tk.X)

        # Treinamento passo a passo
        tk.Label(self.control_frame, text="Treinamento:").pack(pady=(10, 5))
        
        lr_frame = tk.Frame(self.control_frame)
        lr_frame.pack(fill=tk.X)
        tk.Label(lr_frame, text="Taxa de aprendizado:").pack(side=tk.LEFT)
        self.lr_var = tk.DoubleVar(value=0.1)
        tk.Entry(lr_frame, textvariable=self.lr_var, width=6).pack(side=tk.RIGHT)
        
        tk.Button(self.control_frame, text="Treinar 1 época", command=lambda: self.train_epochs(1)).pack(pady=2, fill=tk.X)
        tk.Button(self.control_frame, text="Treinar 10 épocas", command=lambda: self.train_epochs(10)).pack(pady=2, fill=tk.X)
        tk.Button(self.control_frame, text="Treinar até convergência", command=self.train_until_converge).pack(pady=2, fill=tk.X)
        self.stop_btn = tk.Button(self.control_frame, text="Parar Treinamento", command=self.stop_training_func, state=tk.DISABLED)
        self.stop_btn.pack(pady=2, fill=tk.X)
        tk.Button(self.control_frame, text="Resetar Treinamento", command=self.reset_training).pack(pady=10, fill=tk.X)

        # Grafo da rede
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Gráfico de erro
        self.fig_err, self.ax_err = plt.subplots(figsize=(5, 2))
        self.canvas_err = FigureCanvasTkAgg(self.fig_err, master=self.graph_frame)
        self.canvas_err.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Painel de análise
        tk.Label(self.analysis_frame, text="Análise da Rede", font=("Arial", 14, "bold")).pack(pady=5)
        self.analysis_text = scrolledtext.ScrolledText(self.analysis_frame, width=40, height=25, font=("Consolas", 10))
        self.analysis_text.pack(pady=5, fill=tk.BOTH, expand=True)

        self.update_layers_ui()
        self.draw_graph()
        self.update_analysis()

    def on_closing(self):
        self.stop_training = True
        self.root.quit()
        self.root.destroy()

    def stop_training_func(self):
        self.stop_training = True
        self.status_var.set("Treinamento interrompido")

    # --- Funções CSV ---
    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.dataset = pd.read_csv(file_path)
                messagebox.showinfo("Sucesso", f"CSV carregado: {file_path}\nColunas: {list(self.dataset.columns)}")
                self.status_var.set(f"CSV carregado: {len(self.dataset)} registros")
            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao carregar CSV: {str(e)}")

    def select_columns(self):
        if self.dataset is None:
            messagebox.showerror("Erro", "Carregue um CSV primeiro!")
            return

        cols = list(self.dataset.columns)
        self.entrada_cols = []
        self.saida_cols = []

        selection = tk.Toplevel(self.root)
        selection.title("Selecionar Entradas e Saídas")
        selection.geometry("400x400")

        tk.Label(selection, text="Entradas:").grid(row=0, column=0, sticky="w")
        entrada_vars = [tk.IntVar() for _ in cols]
        for i, col in enumerate(cols):
            tk.Checkbutton(selection, text=col, variable=entrada_vars[i]).grid(row=i+1, column=0, sticky="w")

        tk.Label(selection, text="Saídas:").grid(row=0, column=1, sticky="w")
        saida_vars = [tk.IntVar() for _ in cols]
        for i, col in enumerate(cols):
            tk.Checkbutton(selection, text=col, variable=saida_vars[i]).grid(row=i+1, column=1, sticky="w")

        def confirmar():
            self.entrada_cols = [col for i, col in enumerate(cols) if entrada_vars[i].get()==1]
            self.saida_cols = [col for i, col in enumerate(cols) if saida_vars[i].get()==1]
            if len(self.entrada_cols)==0 or len(self.saida_cols)==0:
                messagebox.showerror("Erro", "Selecione pelo menos uma coluna de entrada e uma de saída")
                return
            selection.destroy()
            self.setup_input_output_layers()

        tk.Button(selection, text="Confirmar", command=confirmar).grid(row=len(cols)+1, column=0, columnspan=2, pady=10)

    # --- Camadas ---
    def setup_input_output_layers(self):
        self.rede.camadas = []
        self.rede.conexoes = []
        self.erros = []
        self.epoch = 0
        self.rede.func_ativacao = self.activation_var.get()

        self.rede.adicionar_camada(len(self.entrada_cols))
        self.rede.adicionar_camada(len(self.saida_cols))
        self.rede.conectar_camadas(self.rede.camadas[0], self.rede.camadas[-1])

        self.update_layers_ui()
        self.draw_graph()
        self.update_analysis()
        self.draw_error_graph()

    def update_layers_ui(self):
        for widget in self.camadas_frame.winfo_children():
            widget.destroy()
        self.layer_controls = []

        for i, camada in enumerate(self.rede.camadas):
            frame = tk.Frame(self.camadas_frame, relief=tk.RIDGE, bd=2, pady=5)
            frame.pack(pady=3, fill=tk.X)
            label = "Entrada" if i==0 else "Saída" if i==len(self.rede.camadas)-1 else f"Oculta {i}"
            tk.Label(frame, text=label).pack(side=tk.LEFT)

            count_var = tk.IntVar(value=len(camada.get_neuronios()))
            tk.Entry(frame, textvariable=count_var, width=5).pack(side=tk.LEFT, padx=5)

            def update_neurons(var=count_var, layer_idx=i):
                target = var.get()
                current = len(self.rede.camadas[layer_idx].get_neuronios())
                if target > current:
                    for _ in range(target-current):
                        self.rede.camadas[layer_idx].adicionar_neuronio(np.random.uniform(-1,1))
                elif target < current:
                    for _ in range(current-target):
                        neuronio = self.rede.camadas[layer_idx].get_neuronios()[-1]
                        self.rede.camadas[layer_idx].remover_neuronio(neuronio)
                self.rede.conexoes = []
                for idx in range(1, len(self.rede.camadas)):
                    self.rede.conectar_camadas(self.rede.camadas[idx-1], self.rede.camadas[idx])
                self.draw_graph()
                self.update_analysis()

            tk.Button(frame, text="Atualizar", command=update_neurons).pack(side=tk.LEFT, padx=5)
            self.layer_controls.append((frame, count_var))

    def add_hidden_layer(self):
        nova_camada = Camada()
        nova_camada.adicionar_neuronio(np.random.uniform(-1,1))
        self.rede.camadas.insert(-1, nova_camada)
        self.rede.conexoes = []
        for idx in range(1, len(self.rede.camadas)):
            self.rede.conectar_camadas(self.rede.camadas[idx-1], self.rede.camadas[idx])
        self.update_layers_ui()
        self.draw_graph()
        self.update_analysis()

    def remove_hidden_layer(self):
        if len(self.rede.camadas) <= 2:
            messagebox.showerror("Erro", "Não é possível remover camada de entrada/saída")
            return
        self.rede.camadas.pop(-2)
        self.rede.conexoes = []
        for idx in range(1, len(self.rede.camadas)):
            self.rede.conectar_camadas(self.rede.camadas[idx-1], self.rede.camadas[idx])
        self.update_layers_ui()
        self.draw_graph()
        self.update_analysis()

    # --- Feedforward ---
    def feedforward(self):
        if self.dataset is None or not self.entrada_cols or not self.saida_cols:
            messagebox.showerror("Erro", "Carregue CSV e selecione entradas/saídas")
            return
        
        # Atualiza função de ativação
        self.rede.func_ativacao = self.activation_var.get()
        
        entradas = self.dataset[self.entrada_cols].iloc[0].values.tolist()
        saida = self.rede.feedforward(entradas)
        messagebox.showinfo("Feedforward", f"Entrada: {entradas}\nSaída: {[round(s, 4) for s in saida]}")
        self.draw_graph()
        self.update_analysis()

    # --- Treinamento ---
    def train_epochs(self, n):
        if self.is_training:
            messagebox.showwarning("Aviso", "Já existe um treinamento em andamento")
            return
            
        if self.dataset is None or not self.entrada_cols or not self.saida_cols:
            messagebox.showerror("Erro", "Carregue CSV e selecione entradas/saídas")
            return
            
        self.is_training = True
        self.stop_training = False
        self.stop_btn.config(state=tk.NORMAL)
        
        # Atualiza função de ativação
        self.rede.func_ativacao = self.activation_var.get()
        
        taxa = self.lr_var.get()
        for epoch in range(n):
            if self.stop_training:
                break
                
            self.status_var.set(f"Treinando época {self.epoch + epoch + 1}/{self.epoch + n}")
            self.root.update()
            
            erro_total = 0
            for idx, row in self.dataset.iterrows():
                entradas = row[self.entrada_cols].values.tolist()
                saidas_esperadas = row[self.saida_cols].values.tolist()
                self.rede.backpropagation(entradas, saidas_esperadas, taxa)
                saidas_obtidas = self.rede.feedforward(entradas)
                erro_total += np.mean((np.array(saidas_esperadas) - np.array(saidas_obtidas))**2)
                
            erro_medio = erro_total / len(self.dataset)
            self.erros.append(erro_medio)
            
            # Update visualization every 5 epochs for performance
            if epoch % 5 == 0:
                self.draw_graph()
                self.update_analysis()
                self.draw_error_graph()
                
        self.epoch += n if not self.stop_training else 0
        self.is_training = False
        self.stop_training = False
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set(f"Treinamento concluído. Total: {self.epoch} épocas")
        
        self.draw_graph()
        self.update_analysis()
        self.draw_error_graph()

    def train_until_converge(self):
        if self.is_training:
            messagebox.showwarning("Aviso", "Já existe um treinamento em andamento")
            return
            
        if self.dataset is None or not self.entrada_cols or not self.saida_cols:
            messagebox.showerror("Erro", "Carregue CSV e selecione entradas/saídas")
            return
            
        self.is_training = True
        self.stop_training = False
        self.stop_btn.config(state=tk.NORMAL)
        
        # Atualiza função de ativação
        self.rede.func_ativacao = self.activation_var.get()
        
        taxa = self.lr_var.get()
        max_epochs = 1000
        tol = 1e-4
        
        for epoch in range(max_epochs):
            if self.stop_training:
                break
                
            self.status_var.set(f"Treinando época {self.epoch + epoch + 1} (convergência)")
            self.root.update()
            
            erro_total = 0
            for idx, row in self.dataset.iterrows():
                entradas = row[self.entrada_cols].values.tolist()
                saidas_esperadas = row[self.saida_cols].values.tolist()
                self.rede.backpropagation(entradas, saidas_esperadas, taxa)
                saidas_obtidas = self.rede.feedforward(entradas)
                erro_total += np.mean((np.array(saidas_esperadas) - np.array(saidas_obtidas))**2)
                
            erro_medio = erro_total / len(self.dataset)
            self.erros.append(erro_medio)
            
            # Update visualization every 5 epochs for performance
            if epoch % 5 == 0:
                self.draw_graph()
                self.update_analysis()
                self.draw_error_graph()
                
            if len(self.erros) > 2 and abs(self.erros[-1] - self.erros[-2]) < tol:
                break
                
        self.epoch += epoch + 1 if not self.stop_training else 0
        self.is_training = False
        self.stop_training = False
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set(f"Treinamento concluído. Total: {self.epoch} épocas")
        
        self.draw_graph()
        self.update_analysis()
        self.draw_error_graph()

    def reset_training(self):
        self.erros = []
        self.epoch = 0
        self.draw_error_graph()
        self.update_analysis()
        self.status_var.set("Treinamento resetado")

    # --- Grafo ---
    def draw_graph(self):
        self.ax.clear()
        G = nx.DiGraph()
        pos = {}
        labels = {}

        for i, camada in enumerate(self.rede.camadas):
            for j, neuronio in enumerate(camada.get_neuronios()):
                G.add_node(neuronio.id)
                pos[neuronio.id] = (i*3, -j)
                labels[neuronio.id] = f"{round(neuronio.saida,2)}"

        for conexao in self.rede.conexoes:
            G.add_edge(conexao.neuronio_origem.id, conexao.neuronio_destino.id, weight=round(conexao.peso,2))

        nx.draw(G, pos, ax=self.ax, with_labels=True, labels=labels, node_color='lightblue', node_size=1000, font_size=8)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=self.ax)
        self.ax.set_title("Rede Neural")
        self.canvas.draw()

    def draw_error_graph(self):
        self.ax_err.clear()
        if self.erros:
            self.ax_err.plot(self.erros, marker='o')
            self.ax_err.set_title("Erro médio por época")
            self.ax_err.set_xlabel("Época")
            self.ax_err.set_ylabel("Erro")
            self.ax_err.grid(True, linestyle='--', alpha=0.7)
        self.canvas_err.draw()

    # --- Análise ---
    def update_analysis(self):
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, f"Épocas treinadas: {self.epoch}\n")
        if self.erros:
            self.analysis_text.insert(tk.END, f"Erro última época: {self.erros[-1]:.6f}\n")
        self.analysis_text.insert(tk.END, f"Função de ativação: {self.rede.func_ativacao}\n")
        self.analysis_text.insert(tk.END, "\nEstrutura da Rede:\n")
        for idx, camada in enumerate(self.rede.camadas):
            tipo = "Entrada" if idx==0 else "Saída" if idx==len(self.rede.camadas)-1 else f"Oculta {idx}"
            self.analysis_text.insert(tk.END, f"  {tipo}: {len(camada.get_neuronios())} neurônios\n")
        self.analysis_text.insert(tk.END, "\nPesos médios por camada:\n")
        for i in range(1, len(self.rede.camadas)):
            pesos = [c.peso for c in self.rede.conexoes if c.neuronio_destino in self.rede.camadas[i].get_neuronios()]
            if pesos:
                self.analysis_text.insert(tk.END, f"  Camada {i-1}->{i}: média={np.mean(pesos):.4f} | min={np.min(pesos):.4f} | max={np.max(pesos):.4f}\n")
        self.analysis_text.insert(tk.END, "\nVieses médios por camada:\n")
        for idx, camada in enumerate(self.rede.camadas[1:], 1):
            vieses = [n.vies for n in camada.get_neuronios()]
            if vieses:
                self.analysis_text.insert(tk.END, f"  Camada {idx}: média={np.mean(vieses):.4f} | min={np.min(vieses):.4f} | max={np.max(vieses):.4f}\n")
        self.analysis_text.insert(tk.END, "\nDetalhes dos pesos e vieses:\n")
        for i in range(1, len(self.rede.camadas)):
            self.analysis_text.insert(tk.END, f"  Pesos {i-1}->{i}:\n")
            for c in self.rede.conexoes:
                if c.neuronio_destino in self.rede.camadas[i].get_neuronios():
                    self.analysis_text.insert(tk.END, f"    {c.neuronio_origem.id[:4]}->{c.neuronio_destino.id[:4]}: {c.peso:.4f}\n")
        self.analysis_text.insert(tk.END, "\nVieses:\n")
        for idx, camada in enumerate(self.rede.camadas[1:], 1):
            for n in camada.get_neuronios():
                self.analysis_text.insert(tk.END, f"  Camada {idx} Neurônio {n.id[:4]}: {n.vies:.4f}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralGUI(root)
    root.mainloop()