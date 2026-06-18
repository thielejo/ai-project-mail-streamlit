# Data Cleaning: Manheim Car Prices

This document summarizes the cleaning steps applied to the raw Manheim vehicle auction dataset (`car_prices.csv`) and the resulting output file (`car_prices_clean.csv`).

## Input

- Source file: `car_prices.csv`
- Original rows: 558,837
- Original columns: 16
- Data period: vehicle transactions from 2014-2015

## Output

- Cleaned file: `car_prices_clean.csv`
- Cleaned rows: 558,743
- Removed rows: 94
- Reproducible script: `scripts/clean_car_prices.py`

## Cleaning Steps Applied

### 1. Repaired malformed CSV rows

During import, 26 rows had one additional field because the `trim` value contained an unquoted comma:

```text
SE PZEV w/Connectivity, Navitgation
```

These rows were repaired by joining the split `trim` fields back together instead of dropping the rows.

### 2. Removed rows with missing target or key mileage value

Rows were removed if either of the following fields was missing:

- `sellingprice`
- `odometer`

This removed 94 rows.

### 3. Filled missing transmission values

Missing or empty `transmission` values were replaced with:

```text
unknown
```

Final `transmission` counts:

| transmission | rows |
|---|---:|
| automatic | 475,880 |
| unknown | 65,326 |
| manual | 17,537 |

### 4. Standardized categorical columns

The following categorical columns were normalized:

- `make`
- `model`
- `trim`
- `body`
- `transmission`
- `vin`
- `state`
- `color`
- `interior`
- `seller`

Applied transformations:

- removed leading and trailing whitespace
- converted text to lowercase
- collapsed repeated whitespace into a single space
- converted empty strings to missing values

Example:

```text
SUV -> suv
Sedan -> sedan
```

### 5. Corrected known typo

The typo `navitgation` was corrected to `navigation` in categorical text values.

### 6. Converted sale date

The `saledate` column was converted from the original auction timestamp format into a consistent datetime format.

Original example:

```text
Tue Dec 16 2014 12:30:00 GMT-0800 (PST)
```

Cleaned format:

```text
2014-12-16 20:30:00+0000
```

The cleaned date is stored in UTC.

## Outliers

Outliers were intentionally kept in the cleaned dataset for now.

The following potential outliers are still present:

| condition | rows retained |
|---|---:|
| `sellingprice < 500` | 5,318 |
| `sellingprice > 150,000` | 25 |
| `odometer > 500,000` | 81 |

These can be filtered later during feature engineering or model-specific preprocessing if needed.

## Validation Checks

After cleaning:

- missing `sellingprice`: 0
- missing `odometer`: 0
- missing `transmission`: 0
- invalid `saledate` values after parsing: 0
- remaining `navitgation` typo values: 0

## How to Reproduce

Run the cleaning script from the project root:

```bash
python3 scripts/clean_car_prices.py
```

This will regenerate `car_prices_clean.csv`.
