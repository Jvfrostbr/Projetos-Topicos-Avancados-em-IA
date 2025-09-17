import numpy
import uuid

''' 
    Classe que representa um neurônio em uma rede neural 
    Atributos:
        - id: Identificador único do neurônio
        - entrada: Valor de entrada do neurônio
        - vies: Viés do neurônio
        - saida: Valor de saída do neurônio após a ativação
        - delta: Valor do delta (erro) usado no treinamento (backpropagation)
    Os pesos das entradas são representados nas arestas de conexão entre os neurônios.
'''

class Neuronio:
    def __init__(self, vies):
        self.id = str(uuid.uuid4())
        self.entrada = 0.0
        self.vies = vies if isinstance(vies, (int, float)) else numpy.random.uniform(-1, 1)
        self.saida = 0.0
        self.delta = 0.0
