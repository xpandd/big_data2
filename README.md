## Pradžia

Žemiau pateikti nurodymai padės jums įdiegti ir paleisti programos kopiją savo lokaliame kompiuteryje ar serveryje.
******************************************************************************************************************
### Docker vaizdo nuoroda:
docker pull astagr/studentu_prognozavimas:latest

### Instrukcijos paleidimui:
docker run -it astagr/studentu_prognozavimas:latest
******************************************************************************************************************
### Reikalavimai
- Docker
- Python 3.8 ar naujesnė versija
- Įsitikinkite, kad jūsų sistemoje įdiegtas Docker

*****************************
# MOKINIŲ PASIEKIMŲ PROGNOZĖ
*****************************
## Apie Projektą
Ši analizė skirta analizuoti ir prognozuoti mokinių pasiekimus, remiantis jų mokymosi duomenimis. Programa parašyta Python programavimo kalba ir supakuota į Docker konteinerį, leidžiantį ją lengvai diegti ir naudoti skirtingose platformose.


********************
Dockerfile turinys:
********************
FROM python:3.8-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "./Prediction.py"]


*************************
requirements.txt turinys
*************************
lightgbm
catboost
matplotlib
numpy
pandas
seaborn
scipy
statsmodels
xgboost
scikit-learn


*****************
Pridedamas failas
******************
Python failas "Prediction.py"


****************************
Problemos atliekant užduotį:
****************************
Sukiausia buvo pritaikyti Jupyter Notebook kodą į py formatą, testuojant Docker vaizdą vis išlysdavo klaidos, nors Jupyter aplinkoje kodas puikiai veikė. 
Teko dali kodo pašalinti, nes CNN kodą labai ilgai kraudavo.

************************
##  Assignment_2 atliko:
************************
- **Asta Gražytė-Skominienė** - ** - https://github.com/xpandd/big_data2.git 




