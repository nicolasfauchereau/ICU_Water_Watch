# calculating and mapping the SPI

```
$ for ndays in 30 60 90 180 360; do papermill calculates_GPM-IMERG_accumulations_SPI.ipynb calculates_GPM-IMERG_accumulations_SPI.ipynb -p ndays ${ndays}; done
```

There is also a `lag` option, in days (relative to current local date), to calculate and map the SPI for previous period endings, e.g. 

```
$ for ndays in 30 60 90 180 360; do papermill calculates_GPM-IMERG_accumulations_SPI.ipynb calculates_GPM-IMERG_accumulations_SPI.ipynb -p ndays ${ndays} -p lag 7; done
```

