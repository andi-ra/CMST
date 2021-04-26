"""Questo modulo è il principale per lanciare tutta l'applicazione. Eseguire il main"""

from __future__ import division

__revision__ = " $Id: population_generator.py.py 1586 2021-04-20 15:56:25Z andi $ "
__docformat__ = 'reStructuredText'

import random
import CMST_package.individual as individual
import CMST_package.hall_of_fame as hall_of_fame
import sys
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

ADJACENCY_MATRIX = []
INDEX_RANGE = 256
MINIMIZE = True
MAXIMUM_FITNESS = 1000
POP_SIZE = 100
CROSS_MUTATE_PROB = .7
SELF_MUTATE_PROB = .3
NUMB_GENER = 200
HOF_SIZE = 4
TOURNAMENT_SIZE = 30
NUM_NODES = 10
ROOT = 1
CAPACITY = 4
TREE_LIKE_LAYOUT = 1
MESH_LIKE_LAYOUT = 0
PLANAR_LAYOUT = 4
SPECTRAL_LAYOUT = 3
ADJACENCY_MATRIX = [[0 for x in range(NUM_NODES)] for x in range(NUM_NODES)]


def draw_graph(g, layout=MESH_LIKE_LAYOUT):
    """
    Funzione per la visualizzazione del grafo, questa usa Networkx per il disegno
        1. Faccio una conversione da lista generica (mia rappresentazione precedente) in grafo
        2. Setto i parametri che voglio nel disegno
        3. Disegno gli archi con le loro etichette
        4. Disegno i nodi

    :param g: grafo da visualizzare
    :param layout: tipo di layout che si vuole (concentrico/gerarchico)
    :return:
    """
    if layout == 1:
        pos = nx.spring_layout(g)
    elif layout == 0:
        pos = nx.circular_layout(g)
    elif layout == 3:
        pos = nx.kamada_kawai_layout(g)
    else:
        pos = nx.planar_layout(g)
    # nodes
    nx.draw_networkx_nodes(g, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(g, pos, width=1)
    arc_weight = nx.get_edge_attributes(g, 'weight')

    nx.draw_networkx_edge_labels(g, pos, edge_labels=arc_weight)

    # labels
    nx.draw_networkx_labels(g, pos, font_size=20, font_family="sans-serif")

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.figure(figsize=(12, 12))
    plt.tight_layout()
    plt.show()


def random_adjacency_matrix(matrix):
    """
    Funzione che crea una matrice di adiacenza aleatoria per la simulazione dell'algoritmo. Questa sezione è necessaria
    per testare come si comporta l'algoritmo al variare del grafo, dei pesi o degli archi cioè di chi collego o no.

    QUI IL SEED È PREFISSATO, NON È REALMENTE ALEATORIO, SE LO SI VUOLE ALEATORIO, TOGLIERE IL SEED
    #TODO: la firma è simil-C e fa schifo, restituisci la matrice
    #TODO: valutare comportamenti strani

    :param matrix: matrice in cui appoggiare i risultati
    :return: matrice del grafo su cui si lavora (da implementare il return)
    """
    global MAXIMUM_FITNESS
    for col in range(len(ADJACENCY_MATRIX)):
        for row in range(col, len(ADJACENCY_MATRIX[col])):
            if col != row:
                value = random.randint(0, 100)
            else:
                value = sys.maxsize
            ADJACENCY_MATRIX[col][row] = value
            ADJACENCY_MATRIX[row][col] = value

            if row != col:
                if value > MAXIMUM_FITNESS:
                    MAXIMUM_FITNESS = value
                print('{:3}'.format(ADJACENCY_MATRIX[col][row]), end=" ")
                matrix[col][row] = ADJACENCY_MATRIX[col][row]
        print()


def fitness_evaluation(individuals):
    """
    Questa è la funzione di valutazione della fitness. Questa verrà usata nel loop evolutivo per selezionare i componenti
    migliori (nel caso in esame soluzioni-albero con minor somma dei pesi). Il procedimento è il seguente:

    * Per ogni individuo estrai il genoma dalla sua struttura dati
    * Creo un'altra matrice di appoggio a partire dal genoma
    * Questa nuova matrice sarà quella su cui mi calcolo lo spanning tree
    * Verifico il vincolo di capacità
    * Converto da spanning tree calcolato sul genotipo a spanning tree ottenuto dalla rete vera

    In pratica per non dover ricopiare ogni volta la matrice di adiacenza globale ADJACENCY_MATRIX creata all'inizio, io
    traccio un parallelo fra genotipo di un individuo e matrice di adiacenza. Lo posso fare poiché entrambi sono delle
    rappresentazioni di reti. Quindi lo spanning tree provvisorio me lo calcolo sul genotipo, poi valuto se rispetta i
    vincoli di capacità (cioè non più di N nodi per sotto-albero), osservo che i vincoli di subtour elimination li
    rispetta grazie all'algoritmo di Prim, quindi parto già con uno spanning tree. Se ho un albero ricoprente ammissibile
    a quel punto mi calcolo il peso dell'albero soluzione selezionando gli archi dal genotipo e il peso di tali archi
    dalla matrice globale ADJACENCY_MATRIX. Infine lascio fare il genotipo, popolo la struttura tree[] del mio individuo
    e ripeto questa operazione per ogni membro della popolazione.

    :param individuals: popolazione da valutare come fitness
    :return: restituisco gli individui con la struttura tree[] popolata
    """
    for ind in individuals:
        matrix = ind.reshape2matrix(NUM_NODES)

        # Apply Prim's Algorithm to get Tree
        tree = prim(matrix)

        # Depth first search starting from ROOT
        over_capacity = False
        for edge in tree:
            temp_weight = 0
            if edge[0] == ROOT:
                temp_weight = measure_branch(tree, edge[1], ROOT)
            elif edge[1] == ROOT:
                temp_weight = measure_branch(tree, edge[0], ROOT)
            if temp_weight > CAPACITY:
                ind.tree = tree
                ind.fitness = MAXIMUM_FITNESS
                over_capacity = True

        # Get weight of converted representation if under capacity
        if not over_capacity:
            weight = 0
            for edge in tree:
                (col, row) = edge
                weight += ADJACENCY_MATRIX[col - 1][row - 1]
            ind.tree = tree
            ind.fitness = weight


def measure_branch(tree, parent, prev):
    """
    Questa funzione conta in modo ricorsivo quanti nodi ci sono in un certo sottoalbero/brach e mi accumula il tutto.

    :param tree:
    :param parent:
    :param prev:
    :return:
    """
    weight = 0
    for edge in tree:
        if edge[0] == parent and edge[1] != prev:
            weight += measure_branch(tree, edge[1], parent)
        elif edge[0] != prev and edge[1] == parent:
            weight += measure_branch(tree, edge[0], parent)
    return weight + 1


def prim(matrix):
    """
    Classico algoritmo di Prim per ottenere il minimum spanning tree

    .. note:: (ripreso e modificato da https://www.programiz.com/dsa/prim-algorithm)

    :param matrix: matrice dei costi dell'individuo da cui calcolarmi un SPT
    :return:
    """
    nodes = [1]
    tree = []
    while len(tree) < NUM_NODES - 1:
        # Get all edges connected to the current tree
        poss_edges = {}
        for col in nodes:
            for row in range(len(matrix[col - 1])):
                if not row + 1 in nodes:
                    poss_edges[(col, row + 1)] = matrix[col - 1][row]
        # Choose smallest edge
        smallest = ((0, 0), sys.maxsize)
        for edge, weight in poss_edges.items():
            if weight < smallest[1]:
                smallest = (edge, weight)
        # Add edge to tree
        tree.append(smallest[0])
        nodes.append(smallest[0][1])
    return tree


def breeding_mutation(individuals, breeding_mutation_pb):
    """
    Funzione che implementa la mutazione da accoppiamento, qui i due genitori che producono le due soluzioni figlie
    mescolando i loro cromosomi. Le mutazioni ottenute tramite scambio di parti del cromosoma (implementato tramite
    scambio di pezzi di genotipo) avvengono con probabilità breeding_mutation_pb.

    1. In particolare bisogna concentrarsi su questa parte di codice per la selezione::

        for i in range(TOURNAMENT_SIZE):
            ind = individuals[random.randint(0, len(individuals) - 1)]
            if best is None or ind.fitness > best.fitness:
                best = ind
        mating_group.append(best)

    2. Attenzione anche al single crossover, dentro il for faccio lo scambio::

        if random.random() < breeding_mutation_pb:
            for i in range(random.randint(1, len(ind1))):
                temp = ind1[i]
                ind1[i] = ind2[i]
                ind2[i] = temp


    :param individuals: popolazione da cui estrarre i genitori
    :param breeding_mutation_pb: probabilità di mutazione nel cromosoma
    :return: popolazione con aggiunti i 2 figli generati
    """
    group_A = list()
    group_B = list()

    # Selezione di chi si accoppia Tournament style
    mating_group = list()
    while len(mating_group) < len(individuals):
        best = None
        for i in range(TOURNAMENT_SIZE):
            # Scegli un individuo a caso
            ind = individuals[random.randint(0, len(individuals) - 1)]
            # Se l'individuo scelto a caso ha una fitness maggiore -> prendilo
            if best is None or ind.fitness > best.fitness:
                best = ind
        mating_group.append(best)

    # estrai e ripopola le liste uno per uno se no Python rompe...
    while mating_group:
        group_A.append(mating_group.pop())
        group_B.append(mating_group.pop())

    ret = list()
    for i in range(len(group_A)):
        ind1 = group_A.pop()
        ind2 = group_B.pop()
        if random.random() < breeding_mutation_pb:
            for i in range(random.randint(1, len(ind1))):
                temp = ind1[i]
                ind1[i] = ind2[i]
                ind2[i] = temp
        ret.append(ind1)
        ret.append(ind2)
    return ret


def self_mutation(individuals, mutation_pb):
    """
    Self-mutation qui con questa funzione modifico il genotipo del singolo individuo, in pratica è come se gli
    modificassi la matrice equivalente dei costi in modo casuale.
    Potrei modificare un ramo, ma se così facessi otterrei una euristica di scambio (branch exchange heuristic).
    Non seguo questa seconda strada (altrimenti otterrei qualcosa come Esau-Williams) modifico la matrice di adiacenza.

    :param individuals: popolazione da mutare (in pratica un insieme di individui)
    :param mutation_pb: probabilità di mutazione di un gene
    :return:
    """
    for individual in individuals:
        for value in individual:
            if random.random() < mutation_pb:
                value = random.randrange(0, INDEX_RANGE)
    return individuals


def generate_population(num_individuals: int, ind_size: int):
    """
    Questa funzione genera un insieme di individui che saranno la mia popolazione. Qui per ora sono tutti individui
    generati casualmente. Il genotipo di ogni individuo è inizializzato in maniera aleatoria, un po' come la matrice
    di adiacenza. Sarà questa sequenza di individui con i loro genotipi a costituire tutto il genoma che sarà oggetto
    delle operazioni successive.

    :param num_individuals:
    :param ind_size:
    :return: popolazione costruita con individui
    """
    generated_pop = list()
    for _ in range(num_individuals):
        generated_pop.append(
            individual.Individual(ind_size, MINIMIZE, INDEX_RANGE))
    return generated_pop


if __name__ == "__main__":
    # Lanciare solo se sei nella fase di test

    random.seed(10)
    Mat = np.zeros((NUM_NODES, NUM_NODES))
    random_adjacency_matrix(Mat)
    g = nx.from_numpy_matrix(Mat, create_using=nx.DiGraph)
    draw_graph(g, MESH_LIKE_LAYOUT)

    hall_of_fame = hall_of_fame.Hof(HOF_SIZE)

    genome = generate_population(POP_SIZE + (POP_SIZE % 2), int(NUM_NODES * (NUM_NODES + 1) / 2))

    fitness_evaluation(genome)
    hall_of_fame.update(genome)

    start = time.process_time()

    for cur_gen in range(NUMB_GENER):
        new_chromosome = breeding_mutation(genome, CROSS_MUTATE_PROB)
        genome = self_mutation(new_chromosome, SELF_MUTATE_PROB)
        fitness_evaluation(genome)
        hall_of_fame.update(genome)

    print(hall_of_fame)
    print("Time: ", time.process_time() - start)
    print("ROOT: ", ROOT)
    print("CAPACITY: ", CAPACITY)

    for candidate in hall_of_fame.paths:
        print(candidate)
        G = nx.Graph()
        G.add_edges_from(candidate)
        draw_graph(G, TREE_LIKE_LAYOUT)

    comp = prim(ADJACENCY_MATRIX)
    weight = 0
    for edge in comp:
        (col, row) = edge
        weight += ADJACENCY_MATRIX[col - 1][row - 1]
    print("Weight Prim: ", weight)
