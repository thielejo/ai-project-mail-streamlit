# Stage 3 Evaluation: Seasonal Adjustment

## Methode

Stage 3 ergänzt den Stage-2-Marktpreis um einen saisonalen Faktor nach
Karosserieform und Verkaufsmonat:

```
final_price = stage2_price x seasonal_factor(body, month)
```

Die Berechnung vergleicht CPI-normalisierte Verkaufspreise mit Vorhersagen des
zeitneutralen Stage-1-V2-Modells. V2 enthält bewusst keinen Verkaufsmonat. Dadurch werden
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
- Eine Best-/Worst-Month-Empfehlung wird nur ausgegeben, wenn mindestens
  2 Monate jeweils wenigstens
  100 Beobachtungen besitzen. Seltene
  Karosserieformen erhalten ausdrücklich keine belastbare Empfehlung.
- Aktuell erfüllen 20 von 45 Karosserieformen
  diese Mindestanforderung.

## Getrennte 80/20-Prüfung der Saisonregel

| Kennzahl | Ohne Stage 3 | Mit Stage 3 | Änderung |
|---|---:|---:|---:|
| MAE auf CPI-normalisierten Preisen | $1,353.15 | $1,339.84 | -0.98% |

Die Faktoren wurden dabei nur aus den 80% Regel-Trainingsdaten abgeleitet und
auf den übrigen 105,688 Zeilen geprüft. Dies ist eine Prüfung
der Saisonregel, kein vollständig unabhängiger neuer Stage-1-Modelltest.

## Wichtigste Muster

| Karosserie | Beobachtungen | Monate mit Daten | Bester Monat | Effekt | Schwächster Monat | Effekt |
|---|---:|---:|---|---:|---|---:|
| sedan | 233,725 | 8 | Mar | +3.2% | Jun | -2.6% |
| suv | 139,801 | 8 | Mar | +2.0% | Apr | -4.5% |
| hatchback | 25,620 | 8 | Jan | +2.1% | Jun | -4.5% |
| minivan | 24,619 | 8 | Mar | +1.4% | Dec | -2.0% |
| coupe | 16,916 | 8 | Mar | +1.8% | Dec | -1.6% |
| crew cab | 15,996 | 8 | Mar | +0.4% | Dec | -1.0% |
| wagon | 15,507 | 8 | Mar | +1.7% | Jun | -1.6% |
| convertible | 10,102 | 8 | Mar | +1.3% | Dec | -1.6% |
| supercrew | 8,871 | 8 | Jun | +0.9% | Dec | -0.7% |
| g sedan | 7,412 | 8 | Jan | +1.7% | Jun | -2.2% |
| supercab | 5,144 | 8 | Jun | +1.0% | Dec | -0.6% |
| regular cab | 4,658 | 8 | Mar | +0.6% | Dec | -1.9% |

## Beispiel-Faktoren

### convertible

| Monat | Faktor | Effekt | Beobachtungen | Sicherheit |
|---|---:|---:|---:|---|
| Jan | 0.9847 | -1.5% | 2,439 | high |
| Feb | 1.0045 | +0.4% | 2,925 | high |
| Mar | 1.0126 | +1.3% | 884 | medium |
| Apr | 1.0005 | +0.0% | 32 | low |
| May | 1.0092 | +0.9% | 1,082 | high |
| Jun | 1.0052 | +0.5% | 1,875 | high |
| Jul | 1.0003 | +0.0% | 37 | low |
| Aug | 1.0000 | +0.0% | 0 | no_data |
| Sep | 1.0000 | +0.0% | 0 | no_data |
| Oct | 1.0000 | +0.0% | 0 | no_data |
| Nov | 1.0000 | +0.0% | 0 | no_data |
| Dec | 0.9835 | -1.7% | 828 | medium |

### suv

| Monat | Faktor | Effekt | Beobachtungen | Sicherheit |
|---|---:|---:|---:|---|
| Jan | 0.9998 | -0.0% | 36,371 | high |
| Feb | 1.0105 | +1.1% | 41,275 | high |
| Mar | 1.0204 | +2.0% | 11,729 | high |
| Apr | 0.9552 | -4.5% | 371 | medium |
| May | 0.9879 | -1.2% | 13,634 | high |
| Jun | 0.9874 | -1.3% | 25,765 | high |
| Jul | 0.9995 | -0.0% | 380 | medium |
| Aug | 1.0000 | +0.0% | 0 | no_data |
| Sep | 1.0000 | +0.0% | 0 | no_data |
| Oct | 1.0000 | +0.0% | 0 | no_data |
| Nov | 1.0000 | +0.0% | 0 | no_data |
| Dec | 0.9795 | -2.0% | 10,276 | high |

### sedan

| Monat | Faktor | Effekt | Beobachtungen | Sicherheit |
|---|---:|---:|---:|---|
| Jan | 1.0041 | +0.4% | 60,160 | high |
| Feb | 1.0146 | +1.5% | 69,657 | high |
| Mar | 1.0317 | +3.2% | 19,567 | high |
| Apr | 0.9749 | -2.5% | 579 | medium |
| May | 0.9769 | -2.3% | 22,208 | high |
| Jun | 0.9735 | -2.6% | 42,955 | high |
| Jul | 0.9968 | -0.3% | 464 | medium |
| Aug | 1.0000 | +0.0% | 0 | no_data |
| Sep | 1.0000 | +0.0% | 0 | no_data |
| Oct | 1.0000 | +0.0% | 0 | no_data |
| Nov | 1.0000 | +0.0% | 0 | no_data |
| Dec | 0.9910 | -0.9% | 18,135 | high |

### coupe

| Monat | Faktor | Effekt | Beobachtungen | Sicherheit |
|---|---:|---:|---:|---|
| Jan | 0.9872 | -1.3% | 4,230 | high |
| Feb | 1.0093 | +0.9% | 5,029 | high |
| Mar | 1.0177 | +1.8% | 1,401 | high |
| Apr | 1.0002 | +0.0% | 37 | low |
| May | 0.9984 | -0.2% | 1,618 | high |
| Jun | 0.9997 | -0.0% | 3,085 | high |
| Jul | 1.0018 | +0.2% | 69 | low |
| Aug | 1.0000 | +0.0% | 0 | no_data |
| Sep | 1.0000 | +0.0% | 0 | no_data |
| Oct | 1.0000 | +0.0% | 0 | no_data |
| Nov | 1.0000 | +0.0% | 0 | no_data |
| Dec | 0.9839 | -1.6% | 1,447 | high |

## Einordnung

- Die korrigierten Effekte sind deutlich kleiner als beim Vergleich roher
  Monatspreise. Das ist plausibel, weil teurere oder jüngere Fahrzeuge in
  einzelnen Monaten nun nicht mehr als Saisonalität fehlinterpretiert werden.
- Die Daten stammen fast vollständig aus Dezember 2014 bis Juli 2015. Stage 3
  bleibt daher eine konservative Heuristik, kein kausaler Nachweis.
