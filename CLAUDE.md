# CLAUDE.md

Solver heurístico de TSP (Caixeiro Viajante) em Python. Trabalho da disciplina de Otimização (IFPR Pinhais), co-autoria com Beatriz Nahas e Rafael Alves.

Estado atual: `tsp.py` lê a instância e chama o núcleo C (`tsp_core.c`): tour inicial de Hilbert → Lin-Kernighan por flips com candidatos alpha-nearness → ILS com double-bridge local e aceite por temperatura; um processo por núcleo (tour inteiro sincronizado até 20 mil nós; trechos com pontas fixas acima disso). Distâncias calculadas das coordenadas (sem matriz, memória O(n)); a matriz só é usada se vier no arquivo. Sem `cc`, cai no fallback Python (com matriz). Funciona de 1 a 200 mil nós; resultados no README.

## Stack

Python 3.8+, numpy, matplotlib, C (via `cc` + ctypes). Instâncias no formato próprio (`<n>_tsp.txt`, gerado por `gerador_tsp.py`) ou TSPLIB `EUC_2D` (`tsplib/`, ótimos conhecidos).

## Comandos

```bash
pip install -r requirements.txt
python gerador_tsp.py                      # gera instâncias aleatórias
python tsp.py 200_tsp.txt [tempo_s]        # gera 200_tsp_solucao.txt e 200_tsp_grafico.png
MPLBACKEND=Agg python test_tsp.py          # tour válido + gap no eil51
MPLBACKEND=Agg python benchmark.py 30 4 tsplib/pr1002.tsp   # tempo, seeds, instâncias
TSP_CFG="workers=1,accept_temp=0.3" python benchmark.py ...  # sobrescreve CONFIG
```

## Convenções

- Comentários e mensagens em português.
- **Todo tour tem de ser permutação válida dos nós** (sem repetição, sem perda). Conferir após qualquer mudança em otimização.
- Não recalcular o comprimento do tour inteiro dentro de laços de busca; usar delta. Nada O(n) por iteração do ILS: o desfazer usa o log de reversões.
- Nada O(n²) em memória ou tempo no caminho de coordenadas (a matriz é só para instâncias com matriz explícita).
- Mudança de heurística só entra com `benchmark.py` antes/depois (várias seeds, mesma instância). Uma seed só engana; ruído entre rodadas chega a ±0,1% em pr2392. O conjunto de candidatos (alpha) é o mesmo para todas as seeds de uma execução, então variar `alpha_iters` também conta como amostra.
- Em instâncias clusterizadas (pr2392) `kick_window` é a alavanca principal; em n grande a janela grande é desligada (`large_kick_max_n`).
- **Commitar antes de refatorar o núcleo**: já perdemos a versão de referência uma vez e foi preciso reconstruí-la.
- `libtspcore.so` é gerada em runtime (`-march=native`), não versionar. `tsp_core.c` mais novo que o `.so` recompila sozinho.
- Código só entra no README se existir no `tsp.py`.
- Arquivos `*_solucao.txt` e `*_grafico.png` são saída gerada; só versionar os de exemplo já existentes.
