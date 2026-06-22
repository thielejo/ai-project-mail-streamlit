# Stage 3 Evaluation: Seasonal Adjustment

## Methode

Stage 3 ergänzt den Stage-2-Marktpreis um einen saisonalen Faktor nach
Karosserieform und Verkaufsmonat:

```
final_price = stage2_price x seasonal_factor(body, month)
```

Die Berechnung vergleicht CPI-normalisierte Verkaufspreise mit Vorhersagen des
Stage-1-Modells für die feste Referenz `2015-02`. Dadurch werden
Unterschiede im Fahrzeugmix (Modell, Alter, Laufleistung und Zustand) weitgehend
herausgerechnet. Der Zielmonat fließt nicht doppelt in Stage 1 und Stage 3 ein.

Pro Karosserieform werden die monatlichen Medianabweichungen relativ zum
Gesamtmedian berechnet. Alle Effekte werden mit einer Prior-Stärke von
1,000 Beobachtungen Richtung 1.0 gedämpft und auf
0.85 bis 1.15 begrenzt.

## Datenabdeckung

- Beobachtete Verkaufsmonate: 1, 2, 3, 4, 5, 6, 7, 12
- August bis November fehlen im Datensatz vollständig und erhalten deshalb
  neutral den Faktor 1.0. Für diese Monate wird keine Empfehlung behauptet.
- Ein Monat wird nur ab 100 Beobachtungen als
  bester oder schwächster Verkaufsmonat berücksichtigt.

## Getrennte 80/20-Prüfung der Saisonregel

| Kennzahl | Ohne Stage 3 | Mit Stage 3 | Änderung |
|---|---:|---:|---:|
| MAE auf CPI-normalisierten Preisen | $1,895.03 | $1,870.20 | -1.31% |

Die Faktoren wurden dabei nur aus den 80% Regel-Trainingsdaten abgeleitet und
auf den übrigen 105,806 Zeilen geprüft. Dies ist eine Prüfung
der Saisonregel, kein vollständig unabhängiger neuer Stage-1-Modelltest.

## Wichtigste Muster

| Karosserie | Beobachtungen | Monate mit Daten | Bester Monat | Effekt | Schwächster Monat | Effekt |
|---|---:|---:|---|---:|---|---:|
| sedan | 234,034 | 8 | Mar | +3.8% | Dec | -7.1% |
| suv | 139,924 | 8 | Mar | +2.9% | Dec | -8.4% |
| hatchback | 25,651 | 8 | Jan | +2.9% | Jun | -4.9% |
| minivan | 24,626 | 8 | Mar | +1.5% | Dec | -6.4% |
| coupe | 16,955 | 8 | Mar | +2.2% | Dec | -4.9% |
| crew cab | 16,005 | 8 | Jun | +1.2% | Dec | -4.1% |
| wagon | 15,521 | 8 | Mar | +2.1% | Dec | -5.1% |
| convertible | 10,148 | 8 | Mar | +2.0% | Dec | -5.1% |
| supercrew | 8,882 | 8 | Jun | +2.9% | Dec | -2.9% |
| g sedan | 7,417 | 8 | Jan | +2.0% | Jun | -2.4% |
| supercab | 5,152 | 8 | Jun | +1.1% | Dec | -2.4% |
| regular cab | 4,668 | 8 | Mar | +1.0% | Dec | -3.4% |

## Beispiel-Faktoren

### convertible

| Monat | Faktor | Effekt | Beobachtungen | Sicherheit |
|---|---:|---:|---:|---|
| Jan | 0.9861 | -1.4% | 2,449 | high |
| Feb | 1.0087 | +0.9% | 2,935 | high |
| Mar | 1.0197 | +2.0% | 888 | medium |
| Apr | 1.0032 | +0.3% | 32 | low |
| May | 1.0155 | +1.5% | 1,088 | high |
| Jun | 1.0029 | +0.3% | 1,886 | high |
| Jul | 1.0013 | +0.1% | 38 | low |
| Aug | 1.0000 | +0.0% | 0 | no_data |
| Sep | 1.0000 | +0.0% | 0 | no_data |
| Oct | 1.0000 | +0.0% | 0 | no_data |
| Nov | 1.0000 | +0.0% | 0 | no_data |
| Dec | 0.9487 | -5.1% | 832 | medium |

### suv

| Monat | Faktor | Effekt | Beobachtungen | Sicherheit |
|---|---:|---:|---:|---|
| Jan | 1.0048 | +0.5% | 36,390 | high |
| Feb | 1.0165 | +1.6% | 41,323 | high |
| Mar | 1.0290 | +2.9% | 11,742 | high |
| Apr | 0.9595 | -4.0% | 371 | medium |
| May | 0.9923 | -0.8% | 13,646 | high |
| Jun | 0.9904 | -1.0% | 25,788 | high |
| Jul | 0.9988 | -0.1% | 380 | medium |
| Aug | 1.0000 | +0.0% | 0 | no_data |
| Sep | 1.0000 | +0.0% | 0 | no_data |
| Oct | 1.0000 | +0.0% | 0 | no_data |
| Nov | 1.0000 | +0.0% | 0 | no_data |
| Dec | 0.9158 | -8.4% | 10,284 | high |

### sedan

| Monat | Faktor | Effekt | Beobachtungen | Sicherheit |
|---|---:|---:|---:|---|
| Jan | 1.0085 | +0.8% | 60,233 | high |
| Feb | 1.0217 | +2.2% | 69,757 | high |
| Mar | 1.0381 | +3.8% | 19,589 | high |
| Apr | 0.9789 | -2.1% | 580 | medium |
| May | 0.9775 | -2.3% | 22,242 | high |
| Jun | 0.9742 | -2.6% | 43,012 | high |
| Jul | 0.9996 | -0.0% | 464 | medium |
| Aug | 1.0000 | +0.0% | 0 | no_data |
| Sep | 1.0000 | +0.0% | 0 | no_data |
| Oct | 1.0000 | +0.0% | 0 | no_data |
| Nov | 1.0000 | +0.0% | 0 | no_data |
| Dec | 0.9286 | -7.1% | 18,157 | high |

### coupe

| Monat | Faktor | Effekt | Beobachtungen | Sicherheit |
|---|---:|---:|---:|---|
| Jan | 0.9961 | -0.4% | 4,242 | high |
| Feb | 1.0157 | +1.6% | 5,037 | high |
| Mar | 1.0220 | +2.2% | 1,403 | high |
| Apr | 1.0006 | +0.1% | 37 | low |
| May | 0.9957 | -0.4% | 1,625 | high |
| Jun | 0.9996 | -0.0% | 3,094 | high |
| Jul | 1.0014 | +0.1% | 69 | low |
| Aug | 1.0000 | +0.0% | 0 | no_data |
| Sep | 1.0000 | +0.0% | 0 | no_data |
| Oct | 1.0000 | +0.0% | 0 | no_data |
| Nov | 1.0000 | +0.0% | 0 | no_data |
| Dec | 0.9514 | -4.9% | 1,448 | high |

## Einordnung

- Die korrigierten Effekte sind deutlich kleiner als beim Vergleich roher
  Monatspreise. Das ist plausibel, weil teurere oder jüngere Fahrzeuge in
  einzelnen Monaten nun nicht mehr als Saisonalität fehlinterpretiert werden.
- Die Daten stammen fast vollständig aus Dezember 2014 bis Juli 2015. Stage 3
  bleibt daher eine konservative Heuristik, kein kausaler Nachweis.
