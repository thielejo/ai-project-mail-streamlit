# Tuning-Experiment — Stage 1 (CatBoost)

> **Autor:** Moritz Binder · **Status:** laufendes Experiment im Branch `Mail_project_moritz`
> Ziel: das CatBoost-Stage-1-Modell (Ausgangswert MAE $1.120) durch systematisches
> Tuning weiter verbessern — methodisch sauber, mit Kreuzvalidierung.

## Vorgehen (4 Schritte)

| Schritt | Datei | Frage |
|---|---|---|
| 1 | `01_cv_target_loss.py` | Rohpreis vs. Log-Ziel, MAE- vs. RMSE-Loss? (4-fache CV) |
| 2 | `02_optuna_hpo.py` | Beste Hyperparameter (Optuna/TPE, resume-sicher) |
| 3 | `03_features_monotonic.py` | Helfen abgeleitete Features + Monotonie-Constraints? |
| 4 | `04_final_full_dataset.py` | Bestes Setup auf dem VOLLEN Datensatz, Vergleich zur Baseline |

Gemeinsame Daten-/Hilfslogik: `_common.py`. Getunt wird auf einer
Teilstichprobe (150k) zur Geschwindigkeit; das Siegermodell wird in Schritt 4
auf dem vollen Datensatz (529k) bestätigt.

## Warum so?

- **CV statt Einzelsplit:** Tuning auf einem einzigen Split optimiert den Split,
  nicht das Problem. 4-fache CV gibt belastbarere Schätzungen.
- **Zielgröße zuerst:** Autopreise sind rechtsschief/multiplikativ — die Wahl
  Rohpreis vs. Log-Ziel wirkt oft stärker als Feinparameter und wird daher
  zuerst geklärt.
- **Monotonie-Constraints:** Domänenwissen (Preis fällt mit Alter/Laufleistung)
  erhöht Plausibilität und verhindert unsinnige Vorhersagen.

## Ergebnisse

> Ausgangswert (ungetuntes CatBoost + Hubraum, voller Datensatz): **MAE $1.120**.
> Schritte 1–3 laufen auf der 150k-Stichprobe (Tuning), Schritt 4 bestätigt auf vollem Datensatz.

### Schritt 1 — Zielgröße/Loss (4-fache CV, 150k) ✅
| Orientierung | MAE | MAPE | R² |
|---|---:|---:|---:|
| Rohpreis + MAE | $1.239 | 13,5 % | 0,939 |
| Log + RMSE | $1.277 | 12,9 % | 0,942 |
| **Log + MAE** (gewählt) | **$1.225** | 13,1 % | 0,940 |

→ Log-Ziel + MAE-Loss minimiert den MAE → Grundlage für die folgenden Schritte.

### Schritt 2 — Hyperparameter (Optuna/TPE, 29 Trials) ✅
- Bester Validierungs-MAE: **$1.200** (Stichprobe) → ~2 % unter ungetunt ($1.225).
- Beste Parameter: `depth=10`, `learning_rate≈0,060`, `l2_leaf_reg≈3,37`,
  `random_strength≈0,76`, `bagging_temperature≈1,76`, `min_data_in_leaf=15`.
- Hinweis: 29 statt 40 Trials (Rechenzeit-Abbrüche); Bestwert war früh stabil.

### Schritt 3 — Features/Monotonie (4-fache CV, 150k) ✅
| Variante | MAE | MAPE | R² |
|---|---:|---:|---:|
| (a) Tuned, Basis-Features | **$1.186** | 12,8 % | 0,944 |
| (b) + abgeleitete Features | $1.194 | 12,8 % | 0,943 |
| (c) + Monotonie-Constraints | $2.369 | 18,8 % | 0,860 |

→ Abgeleitete Features helfen **nicht** (redundant). Monotonie-Constraints
**schaden** in dieser Konstellation stark (zu starr mit Log-Ziel). Gewählt: (a).

### Schritt 4 — Finale auf vollem Datensatz (529k) ✅
Setup: Log-Ziel + MAE-Loss, getunte Parameter, Basis-Features (kein Derived, keine Monotonie).

| Metrik | Ungetunt (Test 9) | **Getunt (Finale)** |
|---|---:|---:|
| MAE | $1.120 | **$1.042** |
| RMSE | $2.035 | **$1.875** |
| R² | 0,954 | **0,961** |
| MAPE | 12,6 % | **11,8 %** |

→ **−$78 (−6,9 %)** durch Tuning (v. a. Log-Zielgröße). Segment-MAPE:
Mid-Range 6,3 %, Premium 5,9 %, Luxus 8,2 %.

**Fazit:** ~$1.042 MAE (~7,5 % vom Ø-Preis). Die $1.000-Grenze wird nicht
unterschritten — der Rest ist weitgehend irreduzibles Auktionsrauschen bzw.
liegt an fehlenden Merkmalen (Unfallhistorie, Ausstattung), die der Datensatz
nicht enthält. Modell (`price_model_catboost_tuned.cbm`, 107 MB) ist gitignored
und via `04_final_full_dataset.py` reproduzierbar.

### Schritt 5 — Gezielte Monotonie-Constraints ❌ verworfen (`05_monotonic_targeted.py`)
Idee: `condition` ↑ und `odometer` ↓ als harte Domänen-Nebenbedingung erzwingen
(Plausibilität). `vehicle_age`/`model_year` **bewusst frei** gelassen, weil
Oldtimer die Alters-Monotonie verletzen — an den Daten geprüft: Preis steigt bei
26+ Jahren wieder (U-Kurve; nur 64 Fahrzeuge / 0,01 %).

| Modell (voller Datensatz) | MAE | R² |
|---|---:|---:|
| Getunt, ohne Constraints (Schritt 4) | $1.042 | 0,961 |
| + Monotonie (condition/odometer) | $1.726 | 0,924 |

→ **+66 % schlechter.** Zweiter Negativbefund nach Schritt 3 (globale Monotonie
+112 %). **Monotonie-Constraints werden nicht verwendet** — sie überschränken das
CatBoost-Modell und zerstören Genauigkeit aus feinen Interaktionen. Plausibilität
(z. B. Lamborghini-Fehlprognose) wird stattdessen über **Eingabe-Plausibilisierung
in der App** adressiert, nicht über Modell-Constraints.

## Gesamtfazit

Bestes Stage-1-Modell: **getuntes CatBoost + Hubraum, MAE $1.042 / R² 0,961 /
MAPE 11,8 %**. Zwei ehrliche Negativbefunde (abgeleitete Features, Monotonie)
gehören zur sauberen Methodik dazu. Die $1.000-Grenze ist mit diesen Daten nicht
erreichbar (Auktionsrauschen + fehlende Merkmale).
