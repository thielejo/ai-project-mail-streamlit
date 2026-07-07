# Projekt-Handoff / neuer Chat

Stand: 01.07.2026  
Repo lokal: `C:\Users\xariv\Documents\GitHub\ai-project-mail-stage3`  
Remote: `https://github.com/digital-business-lectures/ai-project-mail.git`  
Aktueller Branch laut letztem Check: `main`  
Letzter sichtbarer Commit: `e349888 Document AI usage in project`

Diese Datei fasst den bisherigen Chat-Verlauf und den aktuellen Projektstand zusammen, damit später direkt weitergearbeitet werden kann.

Wichtig: Diese Datei ist nur eine Übergabe-/Kontextdatei. Sie wurde nicht automatisch gepusht.

---

## 1. Projekt in einem Satz

Das Projekt ist ein KI-gestützter Demonstrator für dynamische Gebrauchtwagenbewertung. Ein Fahrzeugpreis wird zuerst durch ein Machine-Learning-Modell geschätzt und danach durch Markt- und Saisonfaktoren angepasst.

Die Logik ist dreistufig:

```text
Stage 1: Basispreis des Autos
Stage 2: Markt-/Inflationsanpassung über CPI
Stage 3: saisonale Korrektur nach Karosserieform und Monat
```

---

## 2. Aktuelle fachliche Einordnung

Im Chat wurde mehrfach geklärt, dass manche Arbeiten ursprünglich unter „Stage 3“ liefen, fachlich aber zu Stage 1 gehören.

Finale Einordnung:

- Stage 1 ist das eigentliche Preisprognosemodell.
- Stage 2 ist nur die Marktpreisanpassung über den Used-Car-CPI.
- Stage 3 ist nur die saisonale Korrektur.
- Das neue V2-XGBoost-Modell gehört zu Stage 1, nicht zu Stage 2 oder Stage 3.
- Die große Fehlerverbesserung von ca. 25 % gehört daher zu Stage 1.
- Stage 2 und Stage 3 sind Zusatzlogiken nach dem Basispreis.

Diese Einordnung wurde im Repo dokumentiert, unter anderem in:

- `Aktueller_Stand.md`
- `README.md`
- `docs/stage_ownership.md`

---

## 3. Git-Stand und wichtige Commits

Aktueller Branch laut letztem Check:

```text
main
```

Letzte sichtbare Commits:

```text
e349888 Document AI usage in project
6f0fdfa Reorganize project stages and archive legacy handoff
46f33fb Merge Stage 3 Pascal into main
```

Wichtige frühere Arbeitsstände:

- Branch `codex/stage3-seasonality` wurde vorher für Stage-2-/Stage-3-Arbeiten genutzt.
- Ein früherer Commit dort war `6edddd1 Validate Stage 2 and Stage 3 and update presentation`.
- Danach wurde ein optimiertes Stage-1-V2-Modell erstellt und mit `a5a7ebe Add optimized Stage 1 V2 price model` gepusht.
- Später wurden diese Arbeiten offenbar in `main` integriert.

Aktueller Hinweis:

- Nicht blind `git add .` verwenden.
- Es liegen lokale, unversionierte Dateien im Arbeitsordner.
- Vor jedem Commit zuerst `git status` ausführen.

Zuletzt sichtbare unversionierte Dateien:

```text
CHAT_HANDOFF_FUER_ANDERE_KIS.md
PRAESI_STAGE2_STAGE3_ERKLAERUNG_UND_RUECKFRAGEN.md
docs/hausarbeit_entwurf_v1.docx
docs/hausarbeit_gliederung.docx
docs/hausarbeit_literatur_intro_theorie.docx
scripts/create_hausarbeit_entwurf_v1_docx.py
scripts/create_hausarbeit_gliederung_docx.py
scripts/create_literatur_intro_theorie_docx.py
```

`PRAESI_STAGE2_STAGE3_ERKLAERUNG_UND_RUECKFRAGEN.md` sollte ursprünglich ausdrücklich nicht gepusht werden.

---

## 4. Stage 1 — Preisbasismodell

Stage 1 schätzt den Basiswert eines Fahrzeugs aus Fahrzeugmerkmalen.

Wichtige Dateien:

- `scripts/train_stage1_v2.py`
- `scripts/optimize_stage1_v2.py`
- `scripts/compare_stage1_v1_v2_shared_split.py`
- `scripts/stage1_runtime.py`
- `models/price_model_v2.joblib`
- `models/price_model_v2_metrics.json`
- `models/stage1_v2_optimization.json`
- `docs/stage1/model_results_stage1_v2.md`
- `docs/stage1/model_results_stage1_v1_v2_shared_split.md`

Das neue V2-Modell:

- ist ein 50/50-Ensemble aus zwei XGBoost-Modellen,
- ein Modell schätzt den Preis direkt,
- ein Modell arbeitet mit logarithmiertem Preis,
- beide Prognosen werden kombiniert,
- Ziel war vor allem eine niedrigere absolute Fehlerquote, also ein besserer MAE.

Genutzte Features unter anderem:

- `model_year`
- `vehicle_age`
- `odometer`
- `condition`
- `make`
- `model`
- `trim`
- `body`
- `transmission`
- `state`
- `color`
- `interior`
- `make_model`

Bewusst nicht genutzt:

- `MMR`, weil das ein marktinterner Referenzwert ist und als Leakage/Shortcut gelten kann,
- VIN,
- Seller,
- Verkaufsmonat,
- `year_month`.

Warum kein Verkaufsmonat?

Damit Stage 2 und Stage 3 fachlich sauber bleiben. Das Basismodell soll den Fahrzeugwert schätzen. Marktzeitpunkt und Saison kommen danach.

Wichtige V2-Ergebnisse:

```text
V1 MAE: 1.830,95 USD
V2 MAE: 1.370,15 USD
Verbesserung: 460,80 USD
Relative Verbesserung: 25,17 %
V2 RMSE: 2.400,34 USD
V2 R²: 0,9366
V2 MAPE: 15,13 %
```

Interpretation:

Das neue Modell macht im Schnitt ca. 461 USD weniger Fehler als das alte Modell. Die Verbesserung kommt nicht durch Magie, sondern durch mehr sinnvolle Fahrzeugmerkmale und eine robustere Modellkombination.

---

## 5. Stage 2 — Marktpreisanpassung

Stage 2 passt den Stage-1-Basispreis an die allgemeine Marktlage für Gebrauchtwagen an.

Wichtige Dateien:

- `scripts/stage2_macro.py`
- `scripts/evaluate_stage2.py`
- `macro_index.csv`
- `models/stage2_evaluation.json`
- `docs/stage2/model_results_stage2.md`

Grundidee:

```text
Stage-2-Preis = Stage-1-Basispreis × CPI-Multiplikator
```

Erklärung für Präsentation:

Stage 1 sagt: „Was ist dieses Auto ungefähr wert?“  
Stage 2 sagt: „Wie hat sich der gesamte Gebrauchtwagenmarkt seit dem Basiszeitpunkt verändert?“

Wichtige Logik:

- Als Referenz wurde ein fester Basiszeitpunkt genutzt.
- Der CPI-Multiplikator skaliert den Preis auf den gewünschten Zielmonat.
- Dadurch kann das Modell auch spätere Marktphasen abbilden, zum Beispiel 2020 bis 2026.

Backtest mit V2:

```text
Stage 1 V2 MAE: 1.370,16 USD
Nach Stage 2: 1.376,22 USD
Änderung: +6,06 USD / +0,44 %
```

Interpretation:

Stage 2 verbessert den historischen Test kaum, weil die Testdaten zeitlich nah am ursprünglichen Datenzeitraum liegen. Der Nutzen liegt eher darin, Preise auf spätere Jahre und Monate übertragen zu können.

Wichtiger Prüfpunkt:

- In älteren Texten wurde teilweise `CUSR0000SETA01` genannt.
- Für „Used Cars and Trucks“ ist auf FRED/BLS üblicherweise `CUSR0000SETA02` relevant.
- Vor finaler Abgabe sollte geprüft werden, ob Code, Doku und Paper denselben korrekten CPI-Series-Code verwenden.

---

## 6. Stage 3 — Saisonalität

Stage 3 korrigiert den Preis nach Saisonmustern.

Wichtige Dateien:

- `scripts/stage3_seasonality.py`
- `scripts/evaluate_stage3.py`
- `models/seasonality_factors.csv`
- `models/seasonality_factors_v2.csv`
- `models/stage3_evaluation.json`
- `docs/stage3/model_results_stage3.md`

Grundidee:

```text
Finaler Preis = Stage-2-Preis × Saisonfaktor(Karosserieform, Monat)
```

Stage 3 ist kein weiteres KI-Modell, sondern eine regelbasierte Korrektur.

Die Saisonfaktoren entstehen ungefähr so:

1. Zuerst wird geschaut, wie stark reale Preise vom erwarteten Preis abweichen.
2. Diese Abweichungen werden nach Karosserieform und Monat gruppiert.
3. Unsichere Faktoren werden Richtung neutralem Wert 1,0 geglättet.
4. Extreme Faktoren werden begrenzt, damit Stage 3 nicht überreagiert.
5. Monate ohne Daten bleiben neutral.

Wichtige Schutzmaßnahmen:

- Faktoren werden auf einen sinnvollen Bereich begrenzt, z. B. 0,85 bis 1,15.
- Es gibt Mindestanforderungen an die Datenmenge.
- Empfehlungen werden nur angezeigt, wenn mindestens genug Monate belastbar sind.
- Fehlende Monate bekommen neutralen Faktor 1,0.

Evaluation:

```text
Ohne Stage 3: MAE 1.353,15 USD
Mit Stage 3:  MAE 1.339,84 USD
Verbesserung: 13,31 USD / 0,98 %
```

Interpretation:

Stage 3 bringt keine riesige Verbesserung, aber eine kleine, fachlich sinnvolle Feinkorrektur. Das passt zur Rolle von Stage 3: Es soll nicht das Grundmodell ersetzen, sondern saisonale Restmuster abfangen.

Wichtiger Datenhinweis:

- Beobachtete Monate im ursprünglichen Datensatz: Januar bis Juli und Dezember.
- August bis November fehlen.
- Für fehlende Monate bleibt der Faktor neutral.

---

## 7. Streamlit-App

Die App liegt hier:

- `app/streamlit_app.py`

Startbefehl:

```cmd
cd C:\Users\xariv\Documents\GitHub\ai-project-mail-stage3
git switch main
git pull origin main
uv run streamlit run app/streamlit_app.py
```

Falls der Browser nicht automatisch öffnet:

```text
http://localhost:8501
```

App-Stand laut bisherigem Chat:

- Deutsche Benutzeroberfläche.
- Kilometer-Eingabe statt Meilen; intern Umrechnung auf Meilen.
- Farben werden deutsch angezeigt und intern passend gemappt.
- Bundesstaaten/Regionen werden ausgeschrieben.
- Zustandsskala als Sterne von 1 bis 5.
- Karosserieform hängt von Marke und Modell ab.
- Technische Details sind hinter Toggles versteckt.
- Ziel: nicht zu technisch wirken, aber genug Erklärung bieten.

---

## 8. Zwischenpräsentation

Der Nutzer hatte eine Markdown-Datei für die Zwischenpräsentation bereitgestellt:

- `C:\Users\xariv\Downloads\zwischenpraesi_sprechtext_nach_cleaning (1).md`

Daraufhin wurden Folien und Sprechtext für Stage 2 und Stage 3 erstellt.

Wichtige Dateien:

- `outputs/zwischenpraesentation_stage2_stage3_repo_aktualisiert.pptx`
- `outputs/sprechtext_stage2_stage3_repo_aktualisiert.md`
- `PRAESI_STAGE2_STAGE3_ERKLAERUNG_UND_RUECKFRAGEN.md`

Wichtig:

- Der Präsentationsteil des Nutzers ist nur Stage 2 und Stage 3.
- Stage 1 machen andere Teammitglieder.
- Trotzdem muss Stage 1 kurz als Grundlage erklärt werden, weil Stage 2 und Stage 3 darauf aufbauen.
- `PRAESI_STAGE2_STAGE3_ERKLAERUNG_UND_RUECKFRAGEN.md` wurde auf Wunsch nicht gepusht.

Kernbotschaft für Präsentation:

```text
Wir haben zuerst einen Basispreis aus Fahrzeugmerkmalen.
Danach passt Stage 2 diesen Preis an die allgemeine Marktlage an.
Stage 3 korrigiert anschließend kleinere saisonale Muster.
```

---

## 9. Erklärung für Prof-Rückfragen

Mögliche Fragen und gute Antworten:

### Warum trennt ihr Stage 1, 2 und 3?

Damit jede Stufe fachlich eine klare Aufgabe hat. Stage 1 bewertet das Auto, Stage 2 bewertet die Marktlage und Stage 3 bewertet saisonale Restmuster. Dadurch ist das System erklärbarer als ein einzelnes Blackbox-Modell.

### Warum ist Stage 2 im Backtest nicht besser?

Weil der Testzeitraum sehr nah am ursprünglichen Datenzeitraum liegt. Da gibt es wenig langfristige Marktbewegung zu korrigieren. Stage 2 ist vor allem wichtig, wenn man Preise auf spätere Marktphasen übertragen will.

### Warum ist Stage 3 nur eine kleine Verbesserung?

Weil der größte Teil des Preises durch Fahrzeugmerkmale erklärt wird. Saison ist eher eine Feinkorrektur. Eine kleine Verbesserung ist hier plausibel und sogar gut, weil Stage 3 nicht künstlich übersteuert.

### Warum fehlen Monate?

Der Datensatz enthält historisch nicht alle Monate gleichmäßig. Für fehlende Monate wird kein Muster erfunden, sondern neutraler Faktor 1,0 genutzt.

### Ist Stage 3 Kausalität?

Nein. Stage 3 zeigt historische saisonale Muster, keine bewiesene Ursache-Wirkung. Deshalb wird vorsichtig geglättet und begrenzt.

### Warum kein MMR im Modell?

MMR ist selbst ein professioneller Marktwert. Wenn man ihn nutzt, würde das Modell sehr stark von einem fertigen Referenzpreis abhängig. Für ein eigenständiges Modell wäre das ein Leakage-Risiko.

---

## 10. Hausarbeit / Paper

Für die Hausarbeit wurden lokal mehrere Word-Entwürfe erstellt:

- `docs/hausarbeit_gliederung.docx`
- `docs/hausarbeit_literatur_intro_theorie.docx`
- `docs/hausarbeit_entwurf_v1.docx`

Außerdem lokale Skripte:

- `scripts/create_hausarbeit_gliederung_docx.py`
- `scripts/create_literatur_intro_theorie_docx.py`
- `scripts/create_hausarbeit_entwurf_v1_docx.py`

Diese Dateien waren zuletzt untracked. Wenn sie ins Repo sollen, gezielt committen.

Geplante Hausarbeit:

- LNCS-Stil
- ca. 12 Seiten
- Deadline laut Projektplan: 31.07.2026

Inhaltliche Schwerpunkte:

- Problemstellung Gebrauchtwagenbewertung
- Datenbasis
- Feature Engineering
- Stage-Architektur
- Modellvergleich
- Evaluation mit MAE/RMSE/R²/MAPE
- Grenzen und Ausblick

Literatur, die im Chat genutzt/erwähnt wurde:

- Akerlof (1970): Informationsasymmetrie / Lemons Market
- Rosen (1974): Hedonic Pricing
- Chen & Guestrin (2016): XGBoost
- Pal et al. (2017): Used-Car-Pricing mit Random Forest
- Madhusudhanan et al. (2024): Probabilistic Used-Car-Pricing
- BLS / FRED: CPI Used Cars and Trucks

---

## 11. KI-Nutzung im Projekt

Eine Datei zur KI-Nutzung wurde erstellt und gepusht:

- `KI_NUTZUNG_IM_PROJEKT.md`

Kernaussage:

- KI wurde unterstützend genutzt.
- Entscheidungen und fachliche Einordnung lagen beim Team.
- Codex/ChatGPT half bei Code, Struktur, Dokumentation, Präsentation und Erklärung.
- Es soll nicht so wirken, als sei das Projekt unreflektiert komplett von KI erzeugt worden.

---

## 12. Bekannte technische Hinweise

- PowerShell/CMD zeigt Umlaute manchmal falsch an. Das heißt nicht automatisch, dass Dateien kaputt sind.
- Codex hatte zeitweise keinen Schreibzugriff auf `.git/index.lock`.
- Deshalb mussten manche Git-Befehle vom Nutzer manuell in CMD ausgeführt werden.
- `git fetch origin --prune` hatte funktioniert.
- `git pull --rebase` scheiterte einmal wegen lokalen unstaged changes.
- Lösung war damals:

```cmd
git stash
git pull --rebase origin codex/stage3-seasonality
git stash pop
```

Danach blieb einmal `models/stage3_evaluation.json` lokal geändert.

---

## 13. Nützliche Befehle

Status prüfen:

```cmd
cd C:\Users\xariv\Documents\GitHub\ai-project-mail-stage3
git status
```

Aktuellen Stand holen:

```cmd
git pull origin main
```

App starten:

```cmd
uv run streamlit run app/streamlit_app.py
```

Gezielt einzelne Datei committen:

```cmd
git add DATEINAME
git commit -m "Nachricht"
git push origin main
```

Nicht empfohlen:

```cmd
git add .
```

Grund: Es gibt lokale Entwürfe und Übergabedateien, die nicht automatisch ins Repo sollen.

---

## 14. Empfohlene nächste Schritte

1. CPI-Series-Code endgültig prüfen und überall vereinheitlichen.
2. Hausarbeitsentwurf fachlich finalisieren und auf ca. 12 LNCS-Seiten bringen.
3. Ergebnistabellen aus `docs/stage1`, `docs/stage2`, `docs/stage3` sauber übernehmen.
4. Architekturdiagramm erstellen.
5. App einmal komplett testen.
6. Vor jedem Commit genau prüfen, welche Dateien wirklich ins Repo sollen.

---

## 15. Wichtigster Kurzkontext für eine andere KI

Wenn eine andere KI nur das Nötigste wissen soll:

```text
Das Repo enthält einen Gebrauchtwagenpreis-Demonstrator mit drei Stufen:
Stage 1 = XGBoost-V2-Basispreismodell, deutlich besser als V1.
Stage 2 = CPI-basierte Marktpreisanpassung.
Stage 3 = saisonale Regelkorrektur nach Karosserie und Monat.

Die große Modellverbesserung gehört zu Stage 1, nicht zu Stage 3.
Stage 2 und Stage 3 sind erklärbare Zusatzkorrekturen.
Aktueller Branch ist main.
Nicht blind git add . nutzen, weil lokale Entwürfe und Übergabedateien untracked sind.
```
