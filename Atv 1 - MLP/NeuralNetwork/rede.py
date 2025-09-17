import numpy as np
from .camada import Camada
from .conexao import Conexao
from .neuronio import Neuronio

class RedeNeural:
    """
    Classe que representa uma rede neural simples (MLP - Perceptron Multicamadas)
    Atributos:
        - camadas: Lista de objetos Camada
        - conexoes: Lista de conexões entre os neurônios
        - func_ativacao: Função de ativação usada nos neurônios
    """
    def __init__(self, func_ativacao="sigmoid"):
        self.camadas = []   # Lista de objetos Camada
        self.conexoes = []  # Lista de conexões entre os neurônios
        self.func_ativacao = func_ativacao

    def adicionar_camada(self, num_neuronios):
        camada = Camada()
        for _ in range(num_neuronios):
            vies = np.random.uniform(-1, 1)  # inicialização aleatória
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
            neuronio.delta = erro * neuronio.saida * (1 - neuronio.saida)

        # Erro nas camadas ocultas
        for i in reversed(range(1, len(self.camadas) - 1)):
            for neuronio in self.camadas[i].get_neuronios():
                erro = sum(
                    conexao.peso * conexao.neuronio_destino.delta
                    for conexao in self.conexoes
                    if conexao.neuronio_origem == neuronio
                )
                neuronio.delta = erro * neuronio.saida * (1 - neuronio.saida)

        # Atualização dos pesos
        for conexao in self.conexoes:
            gradiente = conexao.neuronio_destino.delta * conexao.neuronio_origem.saida
            conexao.peso -= taxa_aprendizado * gradiente  # usa -= para descida do gradiente

        # Atualização dos vieses
        for camada in self.camadas[1:]:
            for neuronio in camada.get_neuronios():
                neuronio.vies -= taxa_aprendizado * neuronio.delta  # usa -= para descida do gradiente