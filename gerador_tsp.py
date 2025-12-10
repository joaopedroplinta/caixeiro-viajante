import random
import math
import os

def calcular_distancia(p1, p2):
    """Calcula distância euclidiana arredondada para inteiro."""
    dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return int(round(dist))

def gerar_arquivo_tsp(num_cidades, nome_arquivo, seed=42):
    print(f"Gerando {nome_arquivo} com {num_cidades} cidades...")
    
    random.seed(seed)
    
    # 1. Gerar Coordenadas (X, Y entre 0 e 100 para manter a escala do exemplo)
    coords = []
    for _ in range(num_cidades):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        coords.append((x, y))
    
    with open(nome_arquivo, 'w') as f:
        # Cabeçalho da Seed
        f.write(f"Seed utilizada:{seed}\n\n")
        
        # Seção de Coordenadas
        f.write("Coordenadas:\n")
        for i, (x, y) in enumerate(coords):
            f.write(f"{i}: {x:.4f}, {y:.4f}\n")
        
        f.write("\n\n")
        
        # Seção da Matriz
        f.write("Matriz de Distâncias:\n")
        
        # Calcula e escreve a matriz linha por linha para economizar memória
        for i in range(num_cidades):
            linha = []
            for j in range(num_cidades):
                d = calcular_distancia(coords[i], coords[j])
                linha.append(str(d))
            
            # Escreve a linha inteira separada por espaços
            f.write("   " + "   ".join(linha) + "\n")
            
    print(f"Arquivo {nome_arquivo} concluído com sucesso!")

# Configuração dos arquivos a serem gerados
instancias = [500, 1000, 2000]

if __name__ == "__main__":
    for n in instancias:
        # Usa uma seed diferente para cada arquivo (opcional, aqui usei n como seed)
        gerar_arquivo_tsp(n, f"{n}_tsp.txt", seed=n)