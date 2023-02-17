#Autor: Mariana David 
#Carnet: 201055
#Inteligencia artificial
#Lab2

#importaciones
from pgmpy.models import BayesianNetwork as BN
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
import numpy as np

class BayesModel(object):#Se crea la clase BayesModel, la cual recibe como parámetros: nodos, edge, probabilidad.
    def __init__(self, nodos, edge, probabilidad): #Se define el constructor de la clase 
        self.nodos = nodos #Se asignan los parámetros recibidos al constructor a variables
        self.edge = edge
        self.probabilidad = probabilidad
        self.model = BN() #Se crea el modelo de red bayesiana
        self.inference = None #La variable inference se inicializa como None


    #Construye el modelo bayesiano a partir de los parámetros recibidos en el constructor
    def graficar(self): #Se define la función para graficar el modelo
            self.model.add_nodes_from(self.nodos) #Se agregan los nodos al modelo a partir de la lista de nodos
            self.model.add_edges_from(self.edge) #Se agregan las aristas al modelo a partir de la lista de aristas
            return self.model.edges #Se retorna una lista con las aristas del modelo

    #Función para verificar si el modelo está completamente descrito
    def descripcionCompleta(self):
            for i in range(len(self.probabilidad)): #Se recorre la lista de probabilidades
                if not self.probabilidad[i][1]: #Se verifica que exista alguna probabilidad
                    return False  #En caso de no existir se retorna False
            return True #En caso de existir las probabilidades se retorna True

    #Función para aplicar el algoritmo de enumeración
    def lista(self, variables, evidence):
            en = self.inference.query(variables=variables, evidence=evidence) #Se aplica el algoritmo de enumeración
            return en.values #Se retornan los resultados del algoritmo de enumeración

    #Función para construir el modelo bayesiano
    def modelo(self):
            for nodos in self.nodos: #Se recorre la lista de nodos
                self.model.add_node(nodos) #Se agregan los nodos al modelo
            for edge in self.edge: #Se recorre la lista de aristas
                self.model.add_edge(edge[0],edge[1]) #Se agregan las aristas al modelo
            for probabilidad in self.probabilidad: #Se recorre la lista de probabilidades
                temp_cpd = TabularCPD(probabilidad[0], 2,  values = probabilidad[1], evidence=probabilidad[2], evidence_card=probabilidad[3]) #Se crea la una tabla CPD para cada probabilidad
                self.model.add_cpds(temp_cpd) #Se agregan las tablas CPD al modelo
            self.inference = VariableElimination(self.model) #Se instancia el algoritmo de eliminación de variables

    #Función para obtener los factores del modelo
    def factores(self):
            factores = [] #Se crea una lista vacía
            for i in range(len(self.probabilidad)): #Se recorre la lista de probabilidades
                if self.probabilidad[i][1]: #Se verifica que existan probabilidades
                    factores.append(self.probabilidad[i][0]) #Se agregan los factores a la lista
            return factores #Se retorna la lista con los factores



#PRUEBA

bayes = BayesModel(['Terremoto', 'Maremoto', 'Alarma', 'Boby', 'Juan'], [['Terremoto','Alarma'], ['Maremoto','Alarma'], ['Boby','Alarma'], ['Boby','Juan'], ['Alarma','Juan']],     [['Terremoto', [[0.95], [0.05]],[], []], ['Maremoto', [[0.85], [0.15]],[], []], ['Boby', [[0.90], [0.10]],[], []], ['Juan', [[0.98, .88, .95, .6], [.02, .12, .05, .40]],['Alarma','Boby'], [2,2]],
        ['Alarma', [[0.96, .86, .94, .82, .24, .15, .10, .05], [.04, .14, .06, .18, .76, .85, .90, .95]],['Terremoto','Boby', 'Maremoto'], [2,2,2]]])
bayes.modelo()

print("1. Es completamente Descrita?:", bayes.descripcionCompleta())
print("\n")
print("2.Representacion compactada de la red:", bayes.graficar())
print("\n")
print("3. Factores:", bayes.factores())
print("\n")
print("4. Resultado del algoritmo de enumeracion: \n", bayes.lista(['Alarma'], {'Terremoto': 1,}))
print("\n")
