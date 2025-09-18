import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QGroupBox, QGridLayout, QDialog, QCheckBox
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from NeuralNetwork.rede import RedeNeural
from NeuralNetwork.camada import Camada

class ColumnSelector(QDialog):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Selecionar Entradas e Saídas")
        self.resize(400, 400)
        self.entrada_checks = []
        self.saida_checks = []
        layout = QGridLayout()
        layout.addWidget(QLabel("Entradas"), 0, 0)
        layout.addWidget(QLabel("Saídas"), 0, 1)
        for i, col in enumerate(columns):
            ent = QCheckBox(col)
            sai = QCheckBox(col)
            layout.addWidget(ent, i+1, 0)
            layout.addWidget(sai, i+1, 1)
            self.entrada_checks.append(ent)
            self.saida_checks.append(sai)
        btn = QPushButton("Confirmar")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn, len(columns)+1, 0, 1, 2)
        self.setLayout(layout)

    def get_selection(self):
        entradas = [cb.text() for cb in self.entrada_checks if cb.isChecked()]
        saidas = [cb.text() for cb in self.saida_checks if cb.isChecked()]
        return entradas, saidas

class NeuralGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MLP Interativa com CSV (PyQt5)")
        self.setGeometry(100, 100, 1300, 700)

        self.rede = RedeNeural()
        self.dataset = None
        self.dataset_train = None
        self.dataset_test = None
        self.dataset_train_normalizado = None
        self.dataset_test_normalizado = None
        self.entrada_cols = []
        self.saida_cols = []
        self.erros = []
        self.epoch = 0
        self.entrada_treino_stats = {}  # Para normalização baseada no treino
        self.saida_treino_stats = {}    # Para normalização baseada no treino

        # Layouts principais
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # Painel de controle
        control_panel = QVBoxLayout()
        main_layout.addLayout(control_panel, 1)

        # Painel de gráficos
        graph_panel = QVBoxLayout()
        main_layout.addLayout(graph_panel, 2)

        # Painel de análise
        analysis_panel = QVBoxLayout()
        main_layout.addLayout(analysis_panel, 1)

        # --- Controle CSV ---
        btn_load = QPushButton("Carregar CSV")
        btn_load.clicked.connect(self.load_csv)
        control_panel.addWidget(btn_load)

        btn_select = QPushButton("Selecionar Entradas/Saídas")
        btn_select.clicked.connect(self.select_columns)
        control_panel.addWidget(btn_select)

        self.status_label = QLabel("Pronto")
        control_panel.addWidget(self.status_label)

        # --- Camadas ---
        camadas_box = QGroupBox("Configuração de Camadas")
        camadas_layout = QVBoxLayout()
        camadas_box.setLayout(camadas_layout)
        self.camadas_layout = camadas_layout
        control_panel.addWidget(camadas_box)

        btn_add_hidden = QPushButton("Adicionar camada oculta")
        btn_add_hidden.clicked.connect(self.add_hidden_layer)
        control_panel.addWidget(btn_add_hidden)

        btn_remove_hidden = QPushButton("Remover camada oculta")
        btn_remove_hidden.clicked.connect(self.remove_hidden_layer)
        control_panel.addWidget(btn_remove_hidden)

        # --- Função de ativação ---
        control_panel.addWidget(QLabel("Função de Ativação:"))
        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["sigmoid", "tanh", "relu"])
        self.activation_combo.currentTextChanged.connect(self.update_activation)
        control_panel.addWidget(self.activation_combo)

        # --- Feedforward ---
        btn_feedforward = QPushButton("Rodar Feedforward")
        btn_feedforward.clicked.connect(self.feedforward)
        control_panel.addWidget(btn_feedforward)

        # --- Treinamento ---
        control_panel.addWidget(QLabel("Treinamento:"))

        # SpinBox para quantidade de épocas
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Épocas:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setMinimum(1)
        self.epochs_spin.setMaximum(10000)
        self.epochs_spin.setValue(1)
        epochs_layout.addWidget(self.epochs_spin)
        control_panel.addLayout(epochs_layout)

        # Taxa de aprendizado
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Taxa de aprendizado:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(4)
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.01)
        lr_layout.addWidget(self.lr_spin)
        control_panel.addLayout(lr_layout)

        # Botão Treinar
        btn_train_custom = QPushButton("Treinar")
        btn_train_custom.clicked.connect(lambda: self.train_epochs(self.epochs_spin.value()))
        control_panel.addWidget(btn_train_custom)

        # Botão Treinar até convergência
        btn_train_conv = QPushButton("Treinar até convergência")
        btn_train_conv.clicked.connect(self.train_until_converge)
        control_panel.addWidget(btn_train_conv)

        # Botão Resetar
        btn_reset = QPushButton("Resetar Rede")
        btn_reset.clicked.connect(self.reset_rede)
        control_panel.addWidget(btn_reset)

        # Botão Avaliar
        btn_eval = QPushButton("Avaliar Rede (Teste)")
        btn_eval.clicked.connect(self.avaliar_rede)
        control_panel.addWidget(btn_eval)

        # --- Gráficos ---
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvas(self.fig)
        graph_panel.addWidget(self.canvas)

        self.fig_err, self.ax_err = plt.subplots(figsize=(5, 2))
        self.canvas_err = FigureCanvas(self.fig_err)
        graph_panel.addWidget(self.canvas_err)

        # --- Análise ---
        analysis_panel.addWidget(QLabel("Análise da Rede"))
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        analysis_panel.addWidget(self.analysis_text)

        self.update_layers_ui()
        self.draw_graph()
        self.update_analysis()

    # --- CSV ---
    def load_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Abrir CSV", "", "CSV Files (*.csv)")
        if fname:
            try:
                self.dataset = pd.read_csv(fname)
                QMessageBox.information(self, "Sucesso", f"CSV carregado: {fname}\nColunas: {list(self.dataset.columns)}")
                self.status_label.setText(f"CSV carregado: {len(self.dataset)} registros")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Falha ao carregar CSV: {str(e)}")

    def select_columns(self):
        if self.dataset is None:
            QMessageBox.warning(self, "Erro", "Carregue um CSV primeiro!")
            return
        cols = list(self.dataset.columns)
        dlg = ColumnSelector(cols, self)
        if dlg.exec_():
            entradas, saidas = dlg.get_selection()
            if not entradas or not saidas:
                QMessageBox.warning(self, "Erro", "Selecione pelo menos uma coluna de entrada e uma de saída")
                return
            self.entrada_cols = entradas
            self.saida_cols = saidas
            self.split_data()
            self.normalize_data()
            self.setup_input_output_layers()
    
    def split_data(self, test_size=0.3, random_state=42):
        """Divide o dataset em treino e teste"""
        if self.dataset is None:
            QMessageBox.warning(self, "Erro", "Carregue um CSV primeiro!")
            return

        # Embaralha o dataset
        df_shuffled = self.dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Define o tamanho do conjunto de treino
        total_size = len(df_shuffled)
        test_size_int = int(test_size * total_size)
        train_size_int = total_size - test_size_int

        # Divide os DataFrames
        self.dataset_train = df_shuffled.iloc[:train_size_int]
        self.dataset_test = df_shuffled.iloc[train_size_int:]

        self.status_label.setText(f"Dados divididos: {len(self.dataset_train)} para treino, {len(self.dataset_test)} para teste.")
        QMessageBox.information(self, "Divisão Concluída", f"Conjunto de treino: {len(self.dataset_train)} registros\nConjunto de teste: {len(self.dataset_test)} registros")
    
    def normalize_data(self):
        """Normaliza os dados de entrada e saída usando z-score"""
        if self.dataset_train is None or not self.entrada_cols or not self.saida_cols:
            return
        
        # Criar cópia dos datasets para normalização
        self.dataset_train_normalizado = self.dataset_train.copy()
        self.dataset_test_normalizado = self.dataset_test.copy()
        
        # Normalizar entradas e saídas usando as estatísticas do CONJUNTO DE TREINO
        for col in self.entrada_cols:
            mean = self.dataset_train[col].mean()
            std = self.dataset_train[col].std()
            if std > 0:
                self.dataset_train_normalizado[col] = (self.dataset_train[col] - mean) / std
                self.dataset_test_normalizado[col] = (self.dataset_test[col] - mean) / std
                self.entrada_treino_stats[col] = {'mean': mean, 'std': std}
            else:
                self.entrada_treino_stats[col] = {'mean': mean, 'std': 1.0}
        
        for col in self.saida_cols:
            mean = self.dataset_train[col].mean()
            std = self.dataset_train[col].std()
            if std > 0:
                self.dataset_train_normalizado[col] = (self.dataset_train[col] - mean) / std
                self.dataset_test_normalizado[col] = (self.dataset_test[col] - mean) / std
                self.saida_treino_stats[col] = {'mean': mean, 'std': std}
            else:
                self.saida_treino_stats[col] = {'mean': mean, 'std': 1.0}

    # --- Camadas ---
    def setup_input_output_layers(self):
        self.rede.camadas = []
        self.rede.conexoes = []
        self.erros = []
        self.epoch = 0
        self.rede.func_ativacao = self.activation_combo.currentText()
        self.rede.adicionar_camada(len(self.entrada_cols))
        self.rede.adicionar_camada(len(self.saida_cols))
        self.rede.conectar_camadas(self.rede.camadas[0], self.rede.camadas[-1])
        self.update_layers_ui()
        self.draw_graph()
        self.update_analysis()
        self.draw_error_graph()

    def update_layers_ui(self):
        # Limpa layout
        while self.camadas_layout.count():
            item = self.camadas_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        # Adiciona controles para cada camada
        for i, camada in enumerate(self.rede.camadas):
            h = QHBoxLayout()
            label = "Entrada" if i==0 else "Saída" if i==len(self.rede.camadas)-1 else f"Oculta {i}"
            h.addWidget(QLabel(label))
            spin = QSpinBox()
            spin.setMinimum(1)
            spin.setValue(len(camada.get_neuronios()))
            idx = i
            def update_neurons(val, layer_idx=idx):
                current = len(self.rede.camadas[layer_idx].get_neuronios())
                if val > current:
                    for _ in range(val-current):
                        self.rede.camadas[layer_idx].adicionar_neuronio(np.random.uniform(-1,1))
                elif val < current:
                    for _ in range(current-val):
                        neuronio = self.rede.camadas[layer_idx].get_neuronios()[-1]
                        self.rede.camadas[layer_idx].remover_neuronio(neuronio)
                self.rede.conexoes = []
                for idx2 in range(1, len(self.rede.camadas)):
                    self.rede.conectar_camadas(self.rede.camadas[idx2-1], self.rede.camadas[idx2])
                self.draw_graph()
                self.update_analysis()
            spin.valueChanged.connect(update_neurons)
            h.addWidget(spin)
            w = QWidget()
            w.setLayout(h)
            self.camadas_layout.addWidget(w)

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
            QMessageBox.warning(self, "Erro", "Não é possível remover camada de entrada/saída")
            return
        self.rede.camadas.pop(-2)
        self.rede.conexoes = []
        for idx in range(1, len(self.rede.camadas)):
            self.rede.conectar_camadas(self.rede.camadas[idx-1], self.rede.camadas[idx])
        self.update_layers_ui()
        self.draw_graph()
        self.update_analysis()

    def update_activation(self):
        self.rede.func_ativacao = self.activation_combo.currentText()

    # --- Feedforward ---
    def feedforward(self):
        if self.dataset is None or not self.entrada_cols or not self.saida_cols:
            QMessageBox.warning(self, "Erro", "Carregue CSV e selecione entradas/saídas")
            return
        self.rede.func_ativacao = self.activation_combo.currentText()
        entradas = self.dataset_train_normalizado[self.entrada_cols].iloc[0].values.tolist()
        saida = self.rede.feedforward(entradas)
        QMessageBox.information(self, "Feedforward", f"Entrada normalizada: {[round(x, 4) for x in entradas]}\nSaída: {[round(s, 4) for s in saida]}")
        self.draw_graph()
        self.update_analysis()

    # --- Treinamento ---
    def train_epochs(self, n):
        if self.dataset is None or not self.entrada_cols or not self.saida_cols:
            QMessageBox.warning(self, "Erro", "Carregue CSV e selecione entradas/saídas")
            return
            
        self.rede.func_ativacao = self.activation_combo.currentText()
        taxa_aprendizado = self.lr_spin.value()
        for epoch in range(n):
            erro_total = 0
            for idx, row in self.dataset_train_normalizado.iterrows():
                entradas = row[self.entrada_cols].values.tolist()
                saidas_esperadas = row[self.saida_cols].values.tolist()
                self.rede.backpropagation(entradas, saidas_esperadas, taxa_aprendizado)
                saidas_obtidas = self.rede.feedforward(entradas)
                erro_total += np.mean((np.array(saidas_esperadas) - np.array(saidas_obtidas))**2)
            erro_medio = erro_total / len(self.dataset_train_normalizado)
            self.erros.append(erro_medio)
        self.epoch += n
        QMessageBox.information(self, "Treinamento", f"Treinamento concluído. Total: {self.epoch} épocas")
        self.draw_graph()
        self.update_analysis()
        self.draw_error_graph()

    def train_until_converge(self):
        if self.dataset is None or not self.entrada_cols or not self.saida_cols:
            QMessageBox.warning(self, "Erro", "Carregue CSV e selecione entradas/saídas")
            return
        self.rede.func_ativacao = self.activation_combo.currentText()
        taxa = self.lr_spin.value()
        max_epochs = 1000
        tol = 1e-4
        for epoch in range(max_epochs):
            erro_total = 0
            for idx, row in self.dataset_train_normalizado.iterrows():
                entradas = row[self.entrada_cols].values.tolist()
                saidas_esperadas = row[self.saida_cols].values.tolist()
                self.rede.backpropagation(entradas, saidas_esperadas, taxa)
                saidas_obtidas = self.rede.feedforward(entradas)
                erro_total += np.mean((np.array(saidas_esperadas) - np.array(saidas_obtidas))**2)
            erro_medio = erro_total / len(self.dataset_train_normalizado)
            self.erros.append(erro_medio)
            if len(self.erros) > 2 and abs(self.erros[-1] - self.erros[-2]) < tol:
                break
        self.epoch += epoch + 1
        QMessageBox.information(self, "Treinamento", f"Treinamento concluído. Total: {self.epoch} épocas")
        self.draw_graph()
        self.update_analysis()
        self.draw_error_graph()

    def reset_rede(self):
        """Reseta a rede neural para valores iniciais"""
        self.rede.reset_treinamento()
        self.erros = []
        self.epoch = 0
        # Re-normalizar dados se necessário
        if self.dataset is not None and self.entrada_cols and self.saida_cols:
            self.normalize_data()
        self.draw_error_graph()
        self.update_analysis()
        self.status_label.setText("Rede resetada")
        self.draw_graph()

    # --- Visualização ---
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
        text = f"Épocas treinadas: {self.epoch}\n"
        if self.erros:
            text += f"Erro última época: {self.erros[-1]:.6f}\n"
        text += f"Função de ativação: {self.rede.func_ativacao}\n"
        text += f"Taxa de aprendizado: {self.lr_spin.value()}\n"
        if self.entrada_treino_stats:
            text += f"Dados normalizados: Sim\n"
        else:
            text += f"Dados normalizados: Não\n"
        text += f"\nEstrutura da Rede:\n"
        
        # Usa os métodos da rede neural para obter informações
        estrutura_info = self.rede.get_estrutura_info()
        detalhes = self.rede.get_detalhes_pesos_vieses()
        
        for camada_info in estrutura_info["camadas"]:
            text += f"  {camada_info['tipo']}: {camada_info['quantidade_neuronios']} neurônios\n"
        
        text += "\nPesos médios por camada:\n"
        for peso_info in estrutura_info["pesos_por_camada"]:
            text += f"  Camada {peso_info['de_para']}: média={peso_info['media']:.4f} | min={peso_info['min']:.4f} | max={peso_info['max']:.4f}\n"
        
        text += "\nVieses médios por camada:\n"
        for vies_info in estrutura_info["vieses_por_camada"]:
            text += f"  Camada {vies_info['camada']}: média={vies_info['media']:.4f} | min={vies_info['min']:.4f} | max={vies_info['max']:.4f}\n"
        
        text += "\nDetalhes dos pesos e vieses:\n"
        for i in range(1, len(self.rede.camadas)):
            text += f"  Pesos {i-1}->{i}:\n"
            for peso in detalhes["pesos"]:
                if peso["destino"].startswith(self.rede.camadas[i].get_neuronios()[0].id[:4]):
                    text += f"    {peso['origem']}->{peso['destino']}: {peso['peso']:.4f}\n"
        
        text += "\nVieses:\n"
        for vies in detalhes["vieses"]:
            text += f"  Camada {vies['camada']} Neurônio {vies['neuronio']}: {vies['vies']:.4f}\n"
        
        self.analysis_text.setPlainText(text)
    
    # --- Avaliação ---
    def avaliar_rede(self):
        if self.dataset_test_normalizado is None or not self.entrada_cols or not self.saida_cols:
            QMessageBox.warning(self, "Erro", "Conjunto de teste não disponível ou colunas não definidas")
            return

        erro_total = 0
        acertos = 0
        total = len(self.dataset_test_normalizado)

        for idx, row in self.dataset_test_normalizado.iterrows():
            entradas = row[self.entrada_cols].values.tolist()
            saidas_esperadas = row[self.saida_cols].values.tolist()
            saidas_obtidas = self.rede.feedforward(entradas)

            # Para regressão: erro médio quadrático
            erro_total += np.mean((np.array(saidas_esperadas) - np.array(saidas_obtidas))**2)

            # Para classificação binária: contar acertos
            pred = [1 if s >= 0.5 else 0 for s in saidas_obtidas]
            esperado = [1 if s >= 0.5 else 0 for s in saidas_esperadas]
            if pred == esperado:
                acertos += 1

        erro_medio = erro_total / total
        acuracia = acertos / total

        QMessageBox.information(self, "Avaliação", f"Acurácia (teste): {acuracia*100:.2f}%\nErro médio: {erro_medio:.6f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NeuralGUI()
    window.show()
    sys.exit(app.exec_())