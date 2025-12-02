# Mehurcki

Zvočni detektor mehurčkov

Projekt za predmet Matematika z računalnikom na UL FMF.

## Predpriprava

```
uv sync
```

## Uporaba

### 1. Vizualizacija zvočnega posnetka
```
uv run src/main.py visualize_waveform data/output_2025-10-03_21:51:08.wav
```

### 2. Testiranje posameznega detektorja
```
uv run src/main.py train_detector constant
```
(ne prikazuje vizualizacij)

### 3. Vizualizacija detekcij
```
uv run src/main.py train_detector constant --waveform data/output_2025-10-03_21:51:08.wav
```
### 4. Testiranje vseh detektorjev
```
uv run src/main.py train_detector
```
(ne prikazuje vizualizacij)

# Primeri

## Anotacije podatkov
![Ena sama anotacija mehurcka](media/single-annotation.png)
![Srednje tezek primer z vmesnim smrcanjem](media/snoring.png)
![Tezaven primer s hrupom](media/difficult-case.png)

## Detekcije

TODO
