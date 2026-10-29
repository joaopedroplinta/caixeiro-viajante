# Caixeiro Viajante

## Descrição
Solução para o Problema do Caixeiro Viajante (TSP - Traveling Salesman Problem), um dos problemas clássicos de otimização computacional.

## Objetivo
Encontrar a rota mais curta que visita todas as cidades exatamente uma vez e retorna ao ponto de partida.

## Algoritmo
1. Tour inicial por curva de Hilbert (O(n log n)).
2. **Lin-Kernighan** (cadeia de flips 2-opt, profundidade até 30, largura 5/3/2/1…) com fila de cidades alteradas e **6 candidatos por cidade escolhidos por alpha-nearness** (subgradiente de Held-Karp + MST sobre o grafo dos 16 vizinhos mais próximos, como no LKH).
3. **ILS**: perturbação double-bridge (troca de dois segmentos adjacentes; metade das vezes de até 50 cidades, metade de até n/3 ou 1000, só até 5000 nós), nova busca só nas pontas alteradas, desfeita por um log de reversões (custo proporcional ao que mudou, não a n). Aceita pioras com probabilidade `exp(-delta/T)` (T = 0,1 × aresta média, só até 20 mil nós) e guarda o melhor tour.
4. **Paralelo**: um processo por núcleo. Até 20 mil nós todos otimizam o tour inteiro e, a cada 1/20 do tempo, recomeçam do melhor. A partir de 20 mil nós cada processo otimiza um trecho contíguo do tour (na curva de Hilbert, uma região compacta) com as pontas fixas, e as melhorias de todos são aproveitadas; os cortes mudam a cada rodada para que as fronteiras também sejam otimizadas.
5. **Instâncias grandes** (> 12 mil nós): a busca inicial roda em dois estágios (reversões curtas, depois sem limite) e o ILS limita a 1000 cidades o trecho que o LK inverte nas tentativas. Em n grande o custo do ILS era dominado por inverter trechos longos do array e desfazê-los.

Sem matriz de distâncias: o núcleo calcula EUC_2D das coordenadas (vizinhos por grade uniforme), então a memória é O(n). A matriz só é usada quando vem no arquivo (formato próprio de `gerador_tsp.py`).

O núcleo está em `tsp_core.c`, compilado com `cc` na primeira execução e chamado por `ctypes`. Sem compilador, `tsp.py` cai numa versão Python (2-opt + Or-opt + ILS) que usa matriz e só serve para instâncias pequenas.

## Requisitos
- Python 3.8+
- Bibliotecas: numpy, matplotlib
- `cc`/`gcc` (opcional, mas recomendado)

## Instalação
```bash
pip install -r requirements.txt
```

## Uso
```bash
python tsp.py instancia.txt [tempo_ils_segundos]
python test_tsp.py
python benchmark.py [tempo_s] [n_seeds] [instancias...]   # várias seeds em paralelo
```
Aceita o formato próprio (`gerador_tsp.py`) e arquivos TSPLIB `EUC_2D` (`tsplib/`). Gera `<instancia>_solucao.txt` e `<instancia>_grafico.png`.

## Exemplo de Saída
![Visualização da rota TSP](./300_tsp_grafico.png)

## Estrutura
```
├── tsp.py            leitura, solver Python (fallback), saída e gráfico
├── tsp_core.c        núcleo Lin-Kernighan + ILS
├── benchmark.py      várias seeds em paralelo
├── test_tsp.py       validade do tour + qualidade no eil51
├── gerador_tsp.py    gera instâncias aleatórias
├── tsplib/           instâncias TSPLIB com ótimo conhecido
├── *_tsp.txt         instâncias aleatórias
└── requirements.txt
```

## Resultados
Todos os núcleos (8 processos), 4 seeds por instância. Gap contra o ótimo TSPLIB; nas aleatórias (sem ótimo conhecido), contra o melhor valor visto.

| Instância | Nós | Tempo | Melhor | Média | Pior | No ótimo |
|-----------|----:|------:|-------:|------:|-----:|---------:|
| eil51, berlin52, kroA100, eil101, ch150, kroA200, a280, pcb442 | 51–442 | 5 s | 0,00% | 0,00% | 0,00% | 4/4 cada |
| rat783 | 783 | 30 s | 0,00% | 0,00% | 0,00% | 4/4 |
| pr1002 | 1002 | 30 s | 0,00% | 0,00% | 0,00% | 4/4 |
| pr2392 | 2392 | 30 s | 0,01% | 0,09% | 0,17% | 0/4 |
| 1000_tsp (aleatória) | 1000 | 30 s | 2278 | 2278,0 | 2278 | – |
| 2000_tsp (aleatória) | 2000 | 30 s | 3192 | 3194,5 | 3197 | – |

Instâncias grandes (uma execução, 30 s de ILS, 8 processos; o tempo de parede inclui leitura, preparo e a busca local inicial). Não há ótimo conhecido; o "vs. estimativa" usa a fórmula assintótica 0,7124·√(n·A) para pontos uniformes, que ignora o efeito de borda:

| Nós | Parede | Memória | Comprimento | vs. estimativa |
|----:|-------:|--------:|------------:|---------------:|
| 5 000 | 33 s | 175 MB | 511 479 (média de 4 seeds) | +1,5% |
| 10 000 | 32 s | 180 MB | 720 508 | +1,1% |
| 25 000 | 32 s | – | 11 396 980 | – |
| 50 000 | 36 s | 211 MB | 16 048 554 | +0,8% |
| 100 000 | 46 s | 246 MB | 22 783 985 | +1,1% |
| 200 000 | 67 s | 314 MB | 32 264 182 | +1,3% |

Antes do ILS por trechos e do limite de reversão (mesmo hardware): 50 mil nós 16 162 557 (40 s), 100 mil 23 006 724 (65 s), 200 mil 32 619 403 (135 s).

O ILS por trechos só vale a partir de ~20 mil nós: em 5 mil e 10 mil nós o tour inteiro sincronizado é melhor (511 444 contra 513 370 e 719 997 contra 724 351, uma seed). O limite de reversão também só vale acima de 12 mil nós (em 10 mil piora). O tour em array tem inversão O(n) no pior caso, o que vai pesar além de algumas centenas de milhares de nós.

Sensibilidade em pr2392 (média de 4 seeds, 30 s): janela do kick 10 → 0,34%, 50 → 0,19%, 200 → 0,08%, 500 → 0,09%, n/3 (796) → 0,01%, mistura 50/n÷3 → 0,08%. Janelas pequenas só mexem no tour localmente e não corrigem a estrutura global da solução inicial; em instâncias aleatórias grandes (≥ 10 mil nós) a janela grande atrapalha (5× menos iterações), por isso só vale até 5000 nós.

Uma medição anterior de pr2392 deu 0,01% com a versão que usava matriz densa. Reconstruí essa versão e ela reproduz (0,04%), mas o resultado depende do conjunto de candidatos: com o mesmo alpha denso, variar as iterações do subgradiente deu 0,44% (70 iterações), 0,06% (100) e 0,10% (130). Ou seja, aquele valor era um conjunto de candidatos favorável, não uma propriedade do algoritmo. Em outras palavras: **comparar sempre com várias seeds e mais de um ajuste, nunca uma rodada só**.

Evolução (pr2392, média): Python 2-opt + Or-opt 1,15%; núcleo C com 10 vizinhos 0,12%; versão atual 0,08%. Versão original do projeto (clusters + Dijkstra + 3-opt + tabu): 200_tsp = 1052 em 11 s e 500_tsp = 1716 em 94 s.

Testado e descartado: Or-opt dentro do LK (pr1002 0,21% vs 0,15% e pr2392 0,31% vs 0,28% sem ele, 16 seeds).

O gráfico de gap é relativo à melhor solução da própria execução, não ao ótimo.

## Licença

Copyright © 2025 Beatriz Nahas, João Plinta, Rafael Alves  
Todos os direitos reservados.

Este código é propriedade privada. Nenhuma parte deste software pode ser reproduzida, distribuída ou utilizada sem permissão prévia por escrito dos autores.

Para mais detalhes, consulte o arquivo [LICENSE](LICENSE).

## Contato

Para dúvidas ou sugestões, entre em contato com os autores.