# plankifier
Code for plankton dataset creation and classification

Use the script `master.sh` to understand what is contained in this repository.

---


---

### Da fare

#### Prioritario
- Trainare senza immagini, solo features. Usare Random Forest e MLP
- Train SmallTom (lui non usa maxpooling, batchnorm e dropout!)
- Train SmallNet
- Train SmallVGG
- Capire quale di questi e` un buon baseline model
- Scegliere quale dei due e` un buon baseline model
- Train VGG pretrained
- Train models on mixed data
- Vedere perche` non mi accetta batch size maggiore di 16 nel modello di Tom
- Condizioni iniziali

#### Analisi basiche
Fatte per MLP.

#### Analisi avanzate
- Vedere se ottengo migliore generalizzazione mediando i pesi finali

#### Parametri da analizzare

##### Prima
- _Taglia immagini_. Vedere effetto della taglia delle immagini (sto vedendo che immagini piu` grandi diminuiscono il tempo di convergenza). Plot: Final-accuracy vs image-size. Plot: Convergence time(wallclock and iterations) vs image-size. 
- _Data Augmentation_. Plot: miglioramento della predizione con ogni tipo di data augmentation

##### Dopo
- _Batch size_. Tempo di esecuzione e accuracy vs batch size, per ogni modello
- _Numero di classi_. Plot: accuracy in funzione del numero di classi
- _Class imbalance_. Trovare qualche modo di misurare la performance in funzione dell'imbalance. Occhio che quando faccio training non posso splittare a casaccio, devo assicurarmi che ogni classe sia ben rappresentata.

#### Organizzazione dati
- Scripts che raccoglie dati da Q e li mette nella stessa cartella appropriata
- Script che associa features e immagini
- Script che prende dati dalla cartella appropriata, riscala le immagini e le associa alle label (e alle features) per creare il dataset

#### Organizzazione workflow
- Implementare semi-supervised e active learning

---

## Risorse

- Pretrained tricks (pretrained, bottleneck features, fine-tuning)

`https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html`

- Mixed data (dataset composto da features+immagini)

`https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/`

