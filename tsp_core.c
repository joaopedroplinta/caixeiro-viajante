// Núcleo do solver TSP: Lin-Kernighan (cadeia de flips 2-opt) + ILS + candidatos alpha-nearness.
// Compilado sob demanda por tsp.py: cc -O3 -march=native -shared -fPIC tsp_core.c -o libtspcore.so
// Duas fontes de distância: matriz n*n (D) ou coordenadas (xy, EUC_2D arredondado como a TSPLIB).
// Tour em array com posição inversa; cada flip inverte o lado mais curto.
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXD 30            // profundidade máxima da cadeia LK
#define MAXCAND 8

static const int BREADTH[MAXD] = {5, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

typedef struct {
    int n, K;
    const int *D;      // matriz n*n (NULL no modo coordenadas)
    const double *xy;  // n*2 coordenadas (NULL no modo matriz)
    const int *neigh;  // n*K candidatos por cidade, em ordem crescente de distância
    int *tour, *pos;
    int *queue; char *inq; int qh, qc;
    int fi[MAXD + 1], fj[MAXD + 1], nflips;   // flips do passo LK corrente
    int addA[MAXD], addB[MAXD];
    int touched[3 * MAXD + 1], ntouched;
    long last_gain;
    unsigned long long rng;
    int *jr; long jn, jcap; int journaling;   // reversões da iteração do ILS, para desfazer
} S;

static inline int dist(const S *s, int a, int b) {
    if (s->xy) {
        double dx = s->xy[2 * a] - s->xy[2 * b], dy = s->xy[2 * a + 1] - s->xy[2 * b + 1];
        return (int)(sqrt(dx * dx + dy * dy) + 0.5);
    }
    return s->D[(size_t)a * s->n + b];
}
#define DIST(a, b) dist(s, (a), (b))

static inline int succ(S *s, int c) { int p = s->pos[c] + 1; return s->tour[p == s->n ? 0 : p]; }
static inline int pred(S *s, int c) { int p = s->pos[c]; return s->tour[p == 0 ? s->n - 1 : p - 1]; }

static unsigned long long rnd(S *s) {
    s->rng ^= s->rng >> 12; s->rng ^= s->rng << 25; s->rng ^= s->rng >> 27;
    return s->rng * 2685821657736338717ULL;
}

static void push(S *s, int c) {
    if (s->inq[c]) return;
    s->inq[c] = 1;
    int t = s->qh + s->qc; if (t >= s->n) t -= s->n;
    s->queue[t] = c; s->qc++;
}

// Inverte exatamente as posições i..j (cíclico, inclusivo).
static void rev_raw(S *s, int i, int j) {
    int n = s->n;
    if (s->journaling) {
        if (s->jn + 2 > s->jcap) { s->jcap *= 2; s->jr = realloc(s->jr, s->jcap * sizeof(int)); }
        s->jr[s->jn++] = i; s->jr[s->jn++] = j;
    }
    int len = j - i; if (len < 0) len += n; len += 1;
    for (int k = 0; k < len / 2; k++) {
        int a = s->tour[i], b = s->tour[j];
        s->tour[i] = b; s->pos[b] = i; s->tour[j] = a; s->pos[a] = j;
        if (++i == n) i = 0;
        if (--j < 0) j = n - 1;
    }
}

// Inverte o caminho de posições i..j, ou o complemento se for menor; registra para desfazer.
static void flip(S *s, int i, int j) {
    int n = s->n, len = j - i; if (len < 0) len += n; len += 1;
    if (2 * len > n) {
        int ni = j + 1 == n ? 0 : j + 1, nj = i == 0 ? n - 1 : i - 1;
        i = ni; j = nj;
    }
    s->fi[s->nflips] = i; s->fj[s->nflips] = j; s->nflips++;
    rev_raw(s, i, j);
}

static void undo_flip(S *s) {
    s->nflips--;
    int jo = s->journaling;
    s->journaling = 0;
    rev_raw(s, s->fi[s->nflips], s->fj[s->nflips]);
    s->journaling = jo;
    if (jo) s->jn -= 2;  // o flip e seu desfazer se anulam no log
}

typedef struct { int t3, t4, val; } Cand;

// Um passo da cadeia LK: remove (t1,t2) e (t3,t4), adiciona (t2,t3) e (t4,t1) via flip.
// Retorna 1 se achou sequência com ganho > 0 (tour fica melhorado), 0 se desfez tudo.
static int lk(S *s, int t1, int t2, long G, int depth) {
    int fw = succ(s, t1) == t2;
    Cand cand[MAXCAND]; int nc = 0, breadth = BREADTH[depth];
    for (int k = 0; k < s->K; k++) {
        int t3 = s->neigh[(size_t)t2 * s->K + k];
        long g1 = G - DIST(t2, t3);
        if (g1 <= 0) break;
        if (t3 == t1) continue;
        int t4 = fw ? pred(s, t3) : succ(s, t3);
        if (t4 == t2) continue;
        int bad = 0;
        for (int d = 0; d < depth; d++)
            if ((s->addA[d] == t3 && s->addB[d] == t4) || (s->addA[d] == t4 && s->addB[d] == t3)) { bad = 1; break; }
        if (bad) continue;
        int val = (int)(g1 + DIST(t3, t4));
        int p = nc < breadth ? nc++ : (val > cand[nc - 1].val ? nc - 1 : -1);
        if (p < 0) continue;
        while (p > 0 && cand[p - 1].val < val) { cand[p] = cand[p - 1]; p--; }
        cand[p].t3 = t3; cand[p].t4 = t4; cand[p].val = val;
    }
    for (int c = 0; c < nc; c++) {
        int t3 = cand[c].t3, t4 = cand[c].t4; long G2 = cand[c].val;
        if (fw) flip(s, s->pos[t2], s->pos[t4]); else flip(s, s->pos[t4], s->pos[t2]);
        long closed = G2 - DIST(t4, t1);
        s->touched[s->ntouched++] = t2; s->touched[s->ntouched++] = t3; s->touched[s->ntouched++] = t4;
        if (closed > 0) { s->last_gain = closed; return 1; }
        s->addA[depth] = t2; s->addB[depth] = t3;
        if (depth + 1 < MAXD && lk(s, t1, t4, G2, depth + 1)) return 1;
        undo_flip(s);
        s->ntouched -= 3;
    }
    return 0;
}

static long improve(S *s, int t1) {
    for (int dir = 0; dir < 2; dir++) {
        int t2 = dir ? pred(s, t1) : succ(s, t1);
        s->nflips = 0; s->ntouched = 0;
        if (lk(s, t1, t2, DIST(t1, t2), 0)) {
            push(s, t1);
            for (int k = 0; k < s->ntouched; k++) push(s, s->touched[k]);
            return s->last_gain;
        }
    }
    return 0;
}

// Esvazia a fila de cidades sujas; retorna o ganho total.
static long local_search(S *s) {
    long gain = 0;
    while (s->qc) {
        int c = s->queue[s->qh];
        if (++s->qh == s->n) s->qh = 0;
        s->qc--; s->inq[c] = 0;
        long g;
        while ((g = improve(s, c)) > 0) gain += g;
    }
    return gain;
}

static long tour_length(S *s) {
    long L = 0;
    for (int i = 0; i < s->n; i++) L += DIST(s->tour[i], s->tour[i + 1 == s->n ? 0 : i + 1]);
    return L;
}

// Double-bridge local: troca dois segmentos adjacentes de até w cidades (3 reversões,
// registradas no log como qualquer outra). Retorna o delta.
static long kick(S *s, int w) {
    int n = s->n;
    int r = (int)(rnd(s) % n);
    int l1 = 1 + (int)(rnd(s) % w), l2 = 1 + (int)(rnd(s) % w);
    #define AT(o) s->tour[(r + (o)) % n]
    int a = AT(0), b1 = AT(1), b2 = AT(l1), c1 = AT(l1 + 1), c2 = AT(l1 + l2), d = AT(l1 + l2 + 1);
    #undef AT
    long delta = (long)DIST(a, c1) + DIST(c2, b1) + DIST(b2, d) - DIST(a, b1) - DIST(b2, c1) - DIST(c2, d);
    #define P(o) (((r + (o)) % n))
    rev_raw(s, P(1), P(l1 + l2));   // B C -> rev(C) rev(B)
    rev_raw(s, P(1), P(l2));        // rev(C) -> C
    rev_raw(s, P(l2 + 1), P(l1 + l2));  // rev(B) -> B
    #undef P
    push(s, a); push(s, b1); push(s, b2); push(s, c1); push(s, c2); push(s, d);
    return delta;
}

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Otimiza `tour` (in/out): [busca local completa se init] + ILS por time_limit segundos.
// Exatamente um entre D (matriz) e xy (coordenadas) deve ser não nulo.
// Retorna o comprimento final; *iters recebe o número de iterações do ILS.
long tsp_solve(int n, const int *D, const double *xy, int K, const int *neigh, int *tour,
               double time_limit, unsigned long long seed, int kick_window, int kick_window_large,
               double temp, int init, long *iters) {
    S s; memset(&s, 0, sizeof s);
    s.n = n; s.K = K; s.D = D; s.xy = xy; s.neigh = neigh; s.tour = tour;
    s.pos = malloc(n * sizeof(int)); s.queue = malloc(n * sizeof(int)); s.inq = calloc(n, 1);
    s.jcap = 1 << 12; s.jr = malloc(s.jcap * sizeof(int));
    s.rng = seed * 0x9E3779B97F4A7C15ULL + 88172645463325252ULL;
    for (int i = 0; i < n; i++) s.pos[tour[i]] = i;

    long cur = tour_length(&s);
    if (init) {
        for (int i = 0; i < n; i++) push(&s, tour[i]);
        cur -= local_search(&s);
    }

    int w = kick_window < 1 ? 1 : kick_window;
    int wl = kick_window_large < w ? w : kick_window_large;  // janela grande: rearranjos mais globais
    if (w > (n - 2) / 3) w = (n - 2) / 3;
    if (wl > (n - 2) / 3) wl = (n - 2) / 3;
    long it = 0;
    int *best = NULL; long best_len = cur;
    if (n >= 8 && time_limit > 0) {
        // Temperatura: fração da aresta média. Em instâncias grandes o ILS está longe de convergir
        // e guardar o melhor tour custaria O(n) por melhoria; aí só aceita não-pioras.
        double T = n <= 20000 ? temp * cur / n : 0;
        if (T > 0) { best = malloc(n * sizeof(int)); memcpy(best, tour, n * sizeof(int)); }
        double deadline = now() + time_limit;
        s.journaling = 1;
        while (now() < deadline) {
            it++;
            s.jn = 0;
            long delta = kick(&s, (rnd(&s) & 1) ? w : wl);  // metade pequenos, metade grandes
            delta -= local_search(&s);
            int ok = delta <= 0 || (T > 0 && delta < -T * log((rnd(&s) >> 11) * (1.0 / 9007199254740992.0) + 1e-18));
            if (ok) {
                cur += delta;
                if (best && cur < best_len) { best_len = cur; memcpy(best, tour, n * sizeof(int)); }
            } else {
                s.journaling = 0;
                for (long k = s.jn - 2; k >= 0; k -= 2) rev_raw(&s, s.jr[k], s.jr[k + 1]);
                s.journaling = 1;
            }
        }
        if (best) { memcpy(tour, best, n * sizeof(int)); cur = best_len; }
    }
    if (iters) *iters = it;
    free(s.pos); free(s.queue); free(s.inq); free(s.jr); free(best);
    return cur;
}

// ---------------------------------------------------------------------------
// k vizinhos mais próximos por grade uniforme (modo coordenadas). out: n*K, por distância crescente.
void tsp_knn(int n, const double *xy, int K, int *out) {
    double minx = 1e300, maxx = -1e300, miny = 1e300, maxy = -1e300;
    for (int i = 0; i < n; i++) {
        if (xy[2 * i] < minx) minx = xy[2 * i];
        if (xy[2 * i] > maxx) maxx = xy[2 * i];
        if (xy[2 * i + 1] < miny) miny = xy[2 * i + 1];
        if (xy[2 * i + 1] > maxy) maxy = xy[2 * i + 1];
    }
    double w = maxx - minx, h = maxy - miny, big = w > h ? w : h;
    if (big < 1e-9) big = 1e-9;
    double cs = sqrt((w > 1e-9 ? w : 1e-9) * (h > 1e-9 ? h : 1e-9) * 2.0 / n);
    if (cs < 2.0 * big / n) cs = 2.0 * big / n;  // instâncias quase em linha
    int gx = (int)(w / cs) + 1, gy = (int)(h / cs) + 1;
    int *cstart = calloc((size_t)gx * gy + 1, sizeof(int)), *cpts = malloc(n * sizeof(int)), *cell = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        int cx = (int)((xy[2 * i] - minx) / cs), cy = (int)((xy[2 * i + 1] - miny) / cs);
        if (cx >= gx) cx = gx - 1;
        if (cy >= gy) cy = gy - 1;
        cell[i] = cy * gx + cx; cstart[cell[i] + 1]++;
    }
    for (size_t c = 0; c < (size_t)gx * gy; c++) cstart[c + 1] += cstart[c];
    int *fillp = calloc((size_t)gx * gy, sizeof(int));
    for (int i = 0; i < n; i++) cpts[cstart[cell[i]] + fillp[cell[i]]++] = i;
    double *d2 = malloc(K * sizeof(double)); int *id = malloc(K * sizeof(int));
    for (int i = 0; i < n; i++) {
        int cx = cell[i] % gx, cy = cell[i] / gx, nf = 0;
        double xi = xy[2 * i], yi = xy[2 * i + 1];
        for (int r = 0; r <= (gx > gy ? gx : gy); r++) {
            for (int y = cy - r; y <= cy + r; y++) {
                if (y < 0 || y >= gy) continue;
                int step = (y == cy - r || y == cy + r) ? 1 : 2 * r;  // só a borda do anel
                if (step == 0) step = 1;
                for (int x = cx - r; x <= cx + r; x += step) {
                    if (x < 0 || x >= gx) continue;
                    int c = y * gx + x;
                    for (int q = cstart[c]; q < cstart[c + 1]; q++) {
                        int j = cpts[q]; if (j == i) continue;
                        double dx = xy[2 * j] - xi, dy = xy[2 * j + 1] - yi, dd = dx * dx + dy * dy;
                        if (nf == K && dd >= d2[K - 1]) continue;
                        int p = nf < K ? nf++ : K - 1;
                        while (p > 0 && d2[p - 1] > dd) { d2[p] = d2[p - 1]; id[p] = id[p - 1]; p--; }
                        d2[p] = dd; id[p] = j;
                    }
                }
            }
            if (nf == K && d2[K - 1] <= (r * cs) * (r * cs)) break;
        }
        for (int k = 0; k < K; k++) out[(size_t)i * K + k] = id[k < nf ? k : nf - 1];
    }
    free(cstart); free(cpts); free(cell); free(fillp); free(d2); free(id);
}

// ---------------------------------------------------------------------------
// Candidatos por alpha-nearness (Helsgaun) sobre o grafo esparso `cand` (n*K0 vizinhos):
// ascensão de subgradiente para os pesos pi (limite de Held-Karp) numa 1-árvore, depois
// alpha(i,j) = d'(i,j) - maior aresta da MST no caminho i..j, d' = d + pi_i + pi_j.
// Devolve, por cidade, os K (<= K0) de menor alpha, ordenados por distância.

typedef struct { int *h, *pos; const double *key; int n; } Heap;

static void hup(Heap *H, int i) {
    int v = H->h[i]; double k = H->key[v];
    while (i > 0) {
        int p = (i - 1) >> 1, pv = H->h[p];
        if (H->key[pv] <= k) break;
        H->h[i] = pv; H->pos[pv] = i; i = p;
    }
    H->h[i] = v; H->pos[v] = i;
}
static void hdown(Heap *H, int i) {
    int v = H->h[i]; double k = H->key[v];
    for (;;) {
        int c = 2 * i + 1;
        if (c >= H->n) break;
        if (c + 1 < H->n && H->key[H->h[c + 1]] < H->key[H->h[c]]) c++;
        int cv = H->h[c];
        if (H->key[cv] >= k) break;
        H->h[i] = cv; H->pos[cv] = i; i = c;
    }
    H->h[i] = v; H->pos[v] = i;
}

// MST (floresta ligada à raiz 0) sobre as cidades 0..m-1 no grafo esparso, pesos d' = d + pi.
// order = ordem de inserção (pai antes do filho); parent[]/pw[] = pai e peso da aresta ao pai.
static void prim_sparse(S *s, int m, const int *start, const int *adj, const double *pi,
                        int *parent, double *pw, int *order, int *heap, int *hpos, double *key, char *in) {
    for (int i = 0; i < m; i++) { key[i] = 1e300; hpos[i] = -1; in[i] = 0; parent[i] = -1; }
    Heap H = {heap, hpos, key, 0};
    int cnt = 0, nxt = 0;
    while (cnt < m) {
        int u;
        if (H.n == 0) {  // começa componente nova (grafo kNN desconexo): liga à raiz
            while (in[nxt]) nxt++;
            u = nxt;
            if (cnt == 0) { parent[u] = -1; pw[u] = 0; }
            else { parent[u] = order[0]; pw[u] = DIST(u, order[0]) + pi[u] + pi[order[0]]; }
        } else {
            u = H.h[0]; pw[u] = key[u];
            H.n--;
            if (H.n > 0) { H.h[0] = H.h[H.n]; hpos[H.h[0]] = 0; hdown(&H, 0); }
        }
        in[u] = 1; order[cnt++] = u;
        for (int e = start[u]; e < start[u + 1]; e++) {
            int v = adj[e];
            if (v >= m || in[v]) continue;
            double wgt = DIST(u, v) + pi[u] + pi[v];
            if (wgt < key[v]) {
                key[v] = wgt; parent[v] = u;
                if (hpos[v] < 0) { H.h[H.n] = v; hpos[v] = H.n; H.n++; hup(&H, H.n - 1); }
                else hup(&H, hpos[v]);
            }
        }
    }
}

static double path_max(int n, const int *up, const double *mx, const int *depth, int a, int b) {
    double m = -1e300;
    if (depth[a] < depth[b]) { int t = a; a = b; b = t; }
    int diff = depth[a] - depth[b];
    for (int l = 0; diff; l++, diff >>= 1)
        if (diff & 1) { double v = mx[(size_t)l * n + a]; if (v > m) m = v; a = up[(size_t)l * n + a]; }
    if (a == b) return m;
    int LOG = 1; while ((1 << LOG) < n) LOG++;
    for (int l = LOG - 1; l >= 0; l--) {
        int ua = up[(size_t)l * n + a], ub = up[(size_t)l * n + b];
        if (ua != ub) {
            double v1 = mx[(size_t)l * n + a], v2 = mx[(size_t)l * n + b];
            if (v1 > m) m = v1;
            if (v2 > m) m = v2;
            a = ua; b = ub;
        }
    }
    double v1 = mx[a], v2 = mx[b];  // nível 0: arestas até o ancestral comum
    if (v1 > m) m = v1;
    if (v2 > m) m = v2;
    return m;
}

typedef struct { double a; int d, j; } AlphaCand;

void tsp_alpha_candidates(int n, const int *D, const double *xy, int K0, const int *cand,
                          int K, long ub, int iters, int *out) {
    S sv; memset(&sv, 0, sizeof sv); sv.n = n; sv.D = D; sv.xy = xy;
    S *s = &sv;

    // adjacência simétrica (CSR) do grafo de candidatos
    int *start = calloc(n + 1, sizeof(int));
    for (int i = 0; i < n; i++) for (int k = 0; k < K0; k++) { start[i + 1]++; start[cand[(size_t)i * K0 + k] + 1]++; }
    for (int i = 0; i < n; i++) start[i + 1] += start[i];
    int *adj = malloc((size_t)start[n] * sizeof(int)), *fill = calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) for (int k = 0; k < K0; k++) {
        int j = cand[(size_t)i * K0 + k];
        adj[start[i] + fill[i]++] = j; adj[start[j] + fill[j]++] = i;
    }

    double *pi = calloc(n, sizeof(double)), *best_pi = calloc(n, sizeof(double));
    int *parent = malloc(n * sizeof(int)), *order = malloc(n * sizeof(int)), *deg = malloc(n * sizeof(int));
    int *heap = malloc(n * sizeof(int)), *hpos = malloc(n * sizeof(int));
    double *pw = malloc(n * sizeof(double)), *key = malloc(n * sizeof(double));
    char *in = malloc(n);
    double best_W = -1e300, lambda = 2.0;
    int no_imp = 0, sp = n - 1;  // cidade especial da 1-árvore
    if (n > 5000) { iters = (int)(iters * sqrt(5000.0 / n)); if (iters < 10) iters = 10; }

    for (int it = 0; it < iters; it++) {
        prim_sparse(s, n - 1, start, adj, pi, parent, pw, order, heap, hpos, key, in);
        double W = 0;
        memset(deg, 0, n * sizeof(int));
        for (int k = 1; k < n - 1; k++) { int i = order[k]; W += pw[i]; deg[i]++; deg[parent[i]]++; }
        int e1 = -1, e2 = -1; double w1 = 1e300, w2 = 1e300;  // duas menores arestas da cidade especial
        for (int k = 0; k < K0; k++) {
            int j = cand[(size_t)sp * K0 + k];
            if (j == sp || j >= n - 1) continue;
            double w = DIST(sp, j) + pi[sp] + pi[j];
            if (w < w1) { w2 = w1; e2 = e1; w1 = w; e1 = j; } else if (w < w2 && j != e1) { w2 = w; e2 = j; }
        }
        if (e2 < 0) break;
        W += w1 + w2; deg[e1]++; deg[e2]++; deg[sp] = 2;
        double sum_pi = 0; for (int i = 0; i < n; i++) sum_pi += pi[i];
        W -= 2 * sum_pi;
        if (W > best_W + 1e-9) { best_W = W; memcpy(best_pi, pi, n * sizeof(double)); no_imp = 0; }
        else if (++no_imp >= 5) { lambda *= 0.5; no_imp = 0; if (lambda < 1e-4) break; }
        double norm = 0; for (int i = 0; i < n; i++) norm += (double)(deg[i] - 2) * (deg[i] - 2);
        if (norm == 0) break;  // 1-árvore já é um tour
        double gap = ub - W; if (gap < 0.001 * ub) gap = 0.001 * ub;
        double t = lambda * gap / norm;
        for (int i = 0; i < n; i++) pi[i] += t * (deg[i] - 2);
    }

    // MST com os melhores pi, tabelas de ancestrais (binary lifting) com a maior aresta do salto
    prim_sparse(s, n, start, adj, best_pi, parent, pw, order, heap, hpos, key, in);
    int LOG = 1; while ((1 << LOG) < n) LOG++;
    int *up = malloc((size_t)LOG * n * sizeof(int)), *depth = calloc(n, sizeof(int));
    double *mx = malloc((size_t)LOG * n * sizeof(double));
    for (int v = 0; v < n; v++) {
        up[v] = parent[v] < 0 ? v : parent[v];
        mx[v] = parent[v] < 0 ? -1e300 : pw[v];
    }
    for (int k = 1; k < n; k++) depth[order[k]] = depth[parent[order[k]]] + 1;
    for (int l = 1; l < LOG; l++)
        for (int v = 0; v < n; v++) {
            int mid = up[(size_t)(l - 1) * n + v];
            up[(size_t)l * n + v] = up[(size_t)(l - 1) * n + mid];
            double a = mx[(size_t)(l - 1) * n + v], b = mx[(size_t)(l - 1) * n + mid];
            mx[(size_t)l * n + v] = a > b ? a : b;
        }

    AlphaCand *c = malloc(K * sizeof(AlphaCand));
    for (int i = 0; i < n; i++) {
        int nc = 0;
        for (int k = 0; k < K0; k++) {
            int j = cand[(size_t)i * K0 + k], dij = DIST(i, j);
            double a = dij + best_pi[i] + best_pi[j] - path_max(n, up, mx, depth, i, j);
            if (nc == K && (a > c[K - 1].a || (a == c[K - 1].a && dij >= c[K - 1].d))) continue;
            int dup = 0;
            for (int x = 0; x < nc; x++) if (c[x].j == j) { dup = 1; break; }
            if (dup) continue;
            int p = nc < K ? nc++ : K - 1;
            while (p > 0 && (c[p - 1].a > a || (c[p - 1].a == a && c[p - 1].d > dij))) { c[p] = c[p - 1]; p--; }
            c[p].a = a; c[p].d = dij; c[p].j = j;
        }
        // ordena por distância (a poda do LK assume ordem crescente)
        for (int x = 1; x < nc; x++) { AlphaCand t = c[x]; int y = x; while (y > 0 && c[y - 1].d > t.d) { c[y] = c[y - 1]; y--; } c[y] = t; }
        for (int x = 0; x < K; x++) out[(size_t)i * K + x] = c[x < nc ? x : nc - 1].j;
    }
    free(start); free(adj); free(fill); free(pi); free(best_pi); free(parent); free(order); free(deg);
    free(heap); free(hpos); free(pw); free(key); free(in); free(up); free(depth); free(mx); free(c);
}
