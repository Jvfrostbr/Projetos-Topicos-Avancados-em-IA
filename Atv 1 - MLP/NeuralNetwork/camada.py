import uuid
from .neuronio import Neuronio

'''
    Classe que representa uma camada de neurônios em uma rede neural
    Atributos:
        - id: Identificador único da camada
        - neuronios: Lista de neurônios na camada
    Métodos:
        - adicionar_neuronio: Adiciona um neurônio à camada
        - remover_neuronio: Remove um neurônio da camada
        - get_neuronios: Retorna a lista de neurônios da camada
'''

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