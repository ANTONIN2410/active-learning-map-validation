# Active learning to validate satellite maps with fewer field observations

Validating a land cover map means checking it against field observations, and field observations are
expensive. This project builds an **active learning engine** that decides *where* to observe, wave after
wave, while keeping a statistically valid estimate of the map's error rate.

On 10,583 LUCAS field points and four European land cover maps, the engine needs **about a third fewer
observations** than random sampling for the same precision, and still **17% fewer with no prior data at
all**.

![Field observations saved](reports/figures/savings.png)

## Results at a glance

Median label saving at equal precision (16 map × region cases, held-out half of the data, 200
observations, 1,000 simulated campaigns per design):

| What the engine knows before the first wave | Engine | Best static design, same model | Classic stratification |
|---|---|---|---|
| Error model trained in other regions | **32.8%** | 33.1% | 6.9% |
| Same, without satellite embeddings | **30.4%** | 31.0% | 6.9% |
| A map never validated before (model trained on other maps) | **32.0%** | 32.1% | 6.9% |
| Nothing (cold start) | **17.5%** | not possible | 6.9% |

The 95% confidence intervals keep their nominal coverage (93% to 96%) in every scenario.

The honest reading: when a good error model already exists, a well-designed static stratified sample
using that model is just as efficient, so the gain comes from the model rather than from the loop. The
loop is what makes the difference when there is no model to start from, which is the usual situation for
a new map.

## How the engine works

```
          ┌────────────────────────────────────────────────────────────┐
          ▼                                                            │
  1. predict f = P(map wrong | x)  →  2. draw a wave (strata by f,  →  3. observe, update
     at every candidate point          Neyman allocation)               the error model
```

* **Error model**: a logistic regression on features available everywhere (disagreement between
  maps, landscape heterogeneity, confidence layers, and a "fifth opinion" from satellite embeddings).
  It is either transferred from other regions and recalibrated after each wave (maximum a posteriori
  with a prior centred on the transferred model), or learned from scratch.
* **Sampling**: each wave is a stratified random sample designed before its observations, so the
  probabilities of selection are always known.
* **Estimation**: a stratified difference estimator per wave, averaged over waves. It is unbiased
  whatever the quality of the model, and the waves form a martingale, which gives the variance and the
  confidence interval. A bad model costs efficiency, never validity.

## The path, notebook by notebook

| Notebook | What it does |
|---|---|
| [`01_exploration_pca`](notebooks/01_exploration_pca.ipynb) | Error rates with confidence intervals, chi-square tests, normed PCA of the indicators with supplementary variables (eigenvalues, cos2, contributions, correlation circle) |
| [`02_logistic_regression`](notebooks/02_logistic_regression.ipynb) | Binomial GLM: likelihood-ratio tests, backward selection, odds ratios, simulated residuals (DHARMa-style); then leave-one-region-out AUC of three feature sets |
| [`03_active_learning`](notebooks/03_active_learning.ipynb) | The engine on one campaign, what it learns wave by wave, and the full benchmark |

I started with the classical statistical toolbox of my Master's course (PCA, frequentist inference,
generalised linear models) before moving to the active learning part: the PCA showed that map errors sit
on one main gradient of landscape heterogeneity, the GLM quantified which indicators matter, and that
model became the engine's starting point.

![PCA](reports/figures/pca.png)

## Repository layout

```
├── data/                  LUCAS points with the four maps and the features (+ sources)
├── notebooks/             the three notebooks above, executed
├── reports/
│   ├── figures/
│   └── results/           benchmark.csv, error_models.csv
├── scripts/
│   └── run_benchmark.py   full benchmark (16 cases)
├── src/alval/
│   ├── data.py            loading, legend, indicators
│   ├── pca.py             normed PCA with interpretation aids
│   ├── glm.py             logistic GLM, LRT, backward selection, simulated residuals
│   ├── features.py        features and transferred error models
│   ├── sampling.py        static designs: exact variances and simulation
│   ├── engine.py          the active learning engine
│   ├── benchmark.py       retrospective benchmark
│   └── plots.py
└── tests/
```

## Reproduce

```bash
git clone https://github.com/ANTONIN2410/active-learning-map-validation.git
cd active-learning-map-validation
pip install -e ".[dev]"
pytest -q
python scripts/run_benchmark.py --replicates 1000 --workers 8   # about 20 min on 10 cores
```

## Experimental design

* **Data**: LUCAS 2018 and 2022 field points in Occitanie, Brittany, Grand Est and the Massif Central;
  ESA WorldCover, Dynamic World, Theia OSO and CLC+ Backbone (2021), harmonised to seven classes.
* **No leakage**: every 20 km spatial block goes entirely to a development half (used to design the
  engine) or a test half (used once for the benchmark). Error models are always trained in the *other*
  regions.
* **Fair baselines**: simple random sampling, stratification by map class, and five static designs that
  use the same error model as the engine, including stratification on the predicted error with Neyman
  allocation.
* **Precise comparisons**: static designs are compared through their exact variance; for the engine,
  the exact conditional variance of each wave is averaged over simulated campaigns.

## Limitations

The benchmark is retrospective: LUCAS labels come from trained surveyors, and the points stand for the
territory. The four regions are all in France, which favours transfer. Only the overall error rate is
estimated, with a constant cost per observation. Observer errors are not modelled: the reference labels are
taken as correct.

## Data and licences

See [`data/README.md`](data/README.md). LUCAS (Eurostat), ESA WorldCover and Dynamic World (CC BY 4.0),
Theia OSO (Etalab), CLC+ Backbone (Copernicus), AlphaEarth satellite embeddings (CC BY 4.0).

## About

I am Antonin Gauthé, Master's student. This project grew out of my work-study
placement at CNES (French space agency), on using citizen observations to validate Earth observation
products.

Code under the MIT licence.

---

### Résumé en français

Un moteur d'apprentissage actif qui choisit où envoyer des observateurs de terrain pour valider une carte
d'occupation du sol, vague après vague, en gardant une estimation sans biais. Sur 10 583 points LUCAS et
quatre cartes, il économise environ un tiers des observations à précision égale, et 17 % sans aucune
donnée préalable. La démarche part de l'ACP et de l'inférence fréquentiste, puis passe par la régression
logistique (GLM binomial) avant le moteur.
