import uuid
import numpy

'''
    Classe que representa uma conexão (aresta) entre dois neurônios em uma rede neural
    Atributos:
        - id: Identificador único da conexão
        - peso: Peso da conexão
        - neuronio_origem: Neurônio de origem da conexão
        - neuronio_destino: Neurônio de destino da conexão
'''

class Conexao:
    def __init__(self, peso=None, neuronio_origem=None, neuronio_destino=None):
        self.id = str(uuid.uuid4())
        self.peso = peso if peso is not None else numpy.random.uniform(-1, 1)
        self.neuronio_origem = neuronio_origem
        self.neuronio_destino = neuronio_destino