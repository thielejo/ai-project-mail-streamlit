# Stage 3 Evaluation: Seasonal Adjustment

## Methode

Stage 3 ergänzt den Stage-2-Marktpreis um einen saisonalen Faktor nach
Karosserieform und Verkaufsmonat:

```
final_price = stage2_price x seasonal_factor(body, month)
```

Die Berechnung vergleicht CPI-normalisierte Verkaufspreise (Quelle: CPI Used Cars
& Trucks, FRED CUSR0000SETA02) mit Vorhersagen des zeitneutralen
**Stage-1-CatBoost-Modells** (getunt, inkl. Hubraum) — dem aktuellen
Produktionsmodell. CatBoost enthält bewusst keinen Verkaufsmonat. Dadurch werden
Unterschiede im Fahrzeugmix (Modell, Alter, Laufleistung und Zustand) weitgehend
herausgerechnet. Der Zielmonat fließt nicht doppelt in Stage 1 und Stage 3 ein.

> Hinweis: Diese Auswertung ist gegen das **CatBoost-Stage-1** neu berechnet
> (Skript `tuning/06_reeval_stage2_stage3_catboost.py`). Der Hubraum stammt aus
> dem vollständigen, versionierten VIN-Decode-Cache
> (`vin_fin_enrichment/vin_decoded_cache_full.csv`, 550.245 VINs per NHTSA vPIC,
> 99,2 % Hubraum-Abdeckung) — die Auswertung ist damit exakt und voll aus dem
> Repo reproduzierbar.

Pro Karosserieform werden die monatlichen Medianabweichungen relativ zum
Gesamtmedian berechnet. Alle Effekte werden mit einer Prior-Stärke von
1,000 Beobachtungen Richtung 1.0 gedämpft und auf
0.85 bis 1.15 begrenzt.

## Datenabdeckung

- Beobachtete Verkaufsmonate: 1, 2, 3, 4, 5, 6, 7, 12
- August bis November fehlen im Datensatz vollständig und erhalten deshalb
  neutral den Faktor 1.0. Für diese Monate wird keine Empfehlung behauptet.
- Eine Best-/Worst-Month-Empfehlung wird nur ausgegeben, wenn mindestens
  2 Monate jeweils wenigstens 100 Beobachtungen besitzen.
- Aktuell erfüllen 20 von 45 Karosserieformen diese Mindestanforderung.

## Getrennte 80/20-Prüfung der Saisonregel

| Kennzahl | Ohne Stage 3 | Mit Stage 3 | Änderung |
|---|---:|---:|---:|
| MAE auf CPI-normalisierten Preisen (CatBoost-Basis) | $1,022.52 | $993.46 | -2.84% |

Die Faktoren wurden nur aus den 80% Regel-Trainingsdaten abgeleitet und auf den
übrigen 105,688 Zeilen geprüft. Dies ist eine Prüfung der Saisonregel,
kein unabhängiger neuer Stage-1-Modelltest. Die absolute Kennzahl ist die
CPI-normalisierte Residualgröße, nicht der Vorhersagefehler des Modells ($1.056).

## Wichtigste Muster

| Karosserie | Beobachtungen | Monate mit Daten | Bester Monat | Effekt | Schwächster Monat | Effekt |
|---|---:|---:|---|---:|---|---:|
| sedan | 233,725 | 8 | Mar | +2.5% | Jun | -3.7% |
| suv | 139,801 | 8 | Mar | +1.8% | Apr | -4.3% |
| hatchback | 25,620 | 8 | Jan | +2.2% | Jun | -3.9% |
| minivan | 24,619 | 8 | Jan | +1.1% | Jun | -1.4% |
| coupe | 16,916 | 8 | Mar | +1.3% | Jun | -1.2% |
| crew cab | 15,996 | 8 | Jan | +0.8% | May | -0.6% |
| wagon | 15,507 | 8 | Mar | +1.5% | Jun | -2.2% |
| convertible | 10,102 | 8 | Mar | +0.7% | Jun | -0.7% |
| supercrew | 8,871 | 8 | Jan | +0.4% | Feb | -0.4% |
| g sedan | 7,412 | 8 | Jan | +2.6% | Jun | -2.7% |
| supercab | 5,144 | 8 | Jan | +0.5% | Feb | -0.5% |
| regular cab | 4,658 | 8 | Jan | +0.7% | Dec | -1.0% |

## Beispiel-Faktoren

### convertible

| Monat | Faktor | Effekt | Beobachtungen | Sicherheit |
|---|---:|---:|---:|---|
| Jan | 1.0031 | +0.3% | 2,439 | high |
| Feb | 1.0009 | +0.1% | 2,925 | high |
| Mar | 1.0071 | +0.7% | 884 | medium |
| Apr | 0.9984 | -0.2% | 32 | low |
| May | 1.0003 | +0.0% | 1,082 | high |
| Jun | 0.9932 | -0.7% | 1,875 | high |
| Jul | 0.9988 | -0.1% | 37 | low |
| Aug | 1.0000 | +0.0% | 0 | no_data |
| Sep | 1.0000 | +0.0% | 0 | no_data |
| Oct | 1.0000 | +0.0% | 0 | no_data |
| Nov | 1.0000 | +0.0% | 0 | no_data |
| Dec | 0.9987 | -0.1% | 828 | medium |

### suv

| Monat | Faktor | Effekt | Beobachtungen | Sicherheit |
|---|---:|---:|---:|---|
| Jan | 1.0121 | +1.2% | 36,371 | high |
| Feb | 1.0072 | +0.7% | 41,275 | high |
| Mar | 1.0184 | +1.8% | 11,729 | high |
| Apr | 0.9573 | -4.3% | 371 | medium |
| May | 0.9789 | -2.1% | 13,634 | high |
| Jun | 0.9779 | -2.2% | 25,765 | high |
| Jul | 0.9921 | -0.8% | 380 | medium |
| Aug | 1.0000 | +0.0% | 0 | no_data |
| Sep | 1.0000 | +0.0% | 0 | no_data |
| Oct | 1.0000 | +0.0% | 0 | no_data |
| Nov | 1.0000 | +0.0% | 0 | no_data |
| Dec | 0.9968 | -0.3% | 10,276 | high |

### sedan

| Monat | Faktor | Effekt | Beobachtungen | Sicherheit |
|---|---:|---:|---:|---|
| Jan | 1.0155 | +1.5% | 60,160 | high |
| Feb | 1.0093 | +0.9% | 69,657 | high |
| Mar | 1.0250 | +2.5% | 19,567 | high |
| Apr | 0.9807 | -1.9% | 579 | medium |
| May | 0.9680 | -3.2% | 22,208 | high |
| Jun | 0.9634 | -3.7% | 42,955 | high |
| Jul | 0.9939 | -0.6% | 464 | medium |
| Aug | 1.0000 | +0.0% | 0 | no_data |
| Sep | 1.0000 | +0.0% | 0 | no_data |
| Oct | 1.0000 | +0.0% | 0 | no_data |
| Nov | 1.0000 | +0.0% | 0 | no_data |
| Dec | 1.0034 | +0.3% | 18,135 | high |

### coupe

| Monat | Faktor | Effekt | Beobachtungen | Sicherheit |
|---|---:|---:|---:|---|
| Jan | 1.0030 | +0.3% | 4,230 | high |
| Feb | 1.0063 | +0.6% | 5,029 | high |
| Mar | 1.0126 | +1.3% | 1,401 | high |
| Apr | 0.9984 | -0.2% | 37 | low |
| May | 0.9926 | -0.7% | 1,618 | high |
| Jun | 0.9877 | -1.2% | 3,085 | high |
| Jul | 0.9987 | -0.1% | 69 | low |
| Aug | 1.0000 | +0.0% | 0 | no_data |
| Sep | 1.0000 | +0.0% | 0 | no_data |
| Oct | 1.0000 | +0.0% | 0 | no_data |
| Nov | 1.0000 | +0.0% | 0 | no_data |
| Dec | 0.9978 | -0.2% | 1,447 | high |

## Einordnung

- Die korrigierten Effekte sind deutlich kleiner als beim Vergleich roher
  Monatspreise. Das ist plausibel, weil teurere oder jüngere Fahrzeuge in
  einzelnen Monaten nun nicht mehr als Saisonalität fehlinterpretiert werden.
- Die Daten stammen fast vollständig aus Dezember 2014 bis Juli 2015. Stage 3
  bleibt daher eine konservative Heuristik, kein kausaler Nachweis.
