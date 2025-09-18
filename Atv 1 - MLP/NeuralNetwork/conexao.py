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
        # Inicialização Xavier para pesos
        if peso is not None:
            self.peso = peso
        else:
            # Xavier initialization: sqrt(6/(fan_in + fan_out))
            fan_in = 1  # será ajustado na rede
            fan_out = 1  # será ajustado na rede
            limit = numpy.sqrt(6.0 / (fan_in + fan_out))
            self.peso = numpy.random.uniform(-limit, limit)
        self.neuronio_origem = neuronio_origem
        self.neuronio_destino = neuronio_destino