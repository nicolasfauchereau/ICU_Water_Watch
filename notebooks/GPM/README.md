# calculating and mapping the SPI

```
$ for ndays in 30 60 90 180 360; do papermill calculates_GPM-IMERG_accumulations_SPI.ipynb calculates_GPM-IMERG_accumulations_SPI.ipynb -p ndays ${ndays}; done
```

