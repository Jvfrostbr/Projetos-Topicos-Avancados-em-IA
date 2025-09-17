import uuid
import numpy as np
from .camada import Camada
from .neuronio import Neuronio
from .conexao import Conexao

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
            vies = np.random.uniform(-2, 2)  # inicialização aleatória
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
    
    def derivada_ativacao(self, saida, entrada=None):
        if self.func_ativacao == "sigmoid":
            return saida * (1 - saida)
        elif self.func_ativacao == "tanh":
            return 1 - saida ** 2
        elif self.func_ativacao == "relu":
            # Para relu, use a entrada antes da ativação
            return 1.0 if entrada is not None and entrada > 0 else 0.0
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

        # Atualização dos pesos (descida do gradiente: -=)
        for conexao in self.conexoes:
            gradiente = conexao.neuronio_destino.delta * conexao.neuronio_origem.saida
            conexao.peso -= taxa_aprendizado * gradiente

        # Atualização dos vieses (descida do gradiente: -=)
        for camada in self.camadas[1:]:
            for neuronio in camada.get_neuronios():
                neuronio.vies -= taxa_aprendizado * neuronio.delta

    def get_estrutura_info(self):
        """Retorna informações sobre a estrutura da rede"""
        info = {
            "camadas": [],
            "pesos_por_camada": [],
            "vieses_por_camada": []
        }
        
        for idx, camada in enumerate(self.camadas):
            tipo = "Entrada" if idx == 0 else "Saída" if idx == len(self.camadas) - 1 else f"Oculta {idx}"
            info["camadas"].append({
                "tipo": tipo,
                "quantidade_neuronios": len(camada.get_neuronios())
            })
        
        for i in range(1, len(self.camadas)):
            pesos = [c.peso for c in self.conexoes if c.neuronio_destino in self.camadas[i].get_neuronios()]
            if pesos:
                info["pesos_por_camada"].append({
                    "de_para": f"{i-1}->{i}",
                    "media": np.mean(pesos),
                    "min": np.min(pesos),
                    "max": np.max(pesos)
                })
        
        for idx, camada in enumerate(self.camadas[1:], 1):
            vieses = [n.vies for n in camada.get_neuronios()]
            if vieses:
                info["vieses_por_camada"].append({
                    "camada": idx,
                    "media": np.mean(vieses),
                    "min": np.min(vieses),
                    "max": np.max(vieses)
                })
        
        return info

    def get_detalhes_pesos_vieses(self):
        """Retorna detalhes específicos de pesos e vieses"""
        detalhes = {
            "pesos": [],
            "vieses": []
        }
        
        for i in range(1, len(self.camadas)):
            for c in self.conexoes:
                if c.neuronio_destino in self.camadas[i].get_neuronios():
                    detalhes["pesos"].append({
                        "origem": c.neuronio_origem.id[:4],
                        "destino": c.neuronio_destino.id[:4],
                        "peso": c.peso
                    })
        
        for idx, camada in enumerate(self.camadas[1:], 1):
            for n in camada.get_neuronios():
                detalhes["vieses"].append({
                    "camada": idx,
                    "neuronio": n.id[:4],
                    "vies": n.vies
                })
        
        return detalhes

    def reset_treinamento(self):
        """Reseta os pesos e vieses da rede para valores aleatórios"""
        for conexao in self.conexoes:
            conexao.peso = np.random.uniform(-1, 1)
        
        for camada in self.camadas[1:]:
            for neuronio in camada.get_neuronios():
                neuronio.vies = np.random.uniform(-1, 1)
                neuronio.delta = 0.0
                neuronio.entrada = 0.0
                neuronio.saida = 0.0