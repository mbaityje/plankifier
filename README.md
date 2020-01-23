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

## Cose che ho imparato

- Some of the images have one dimension that is extremely larger than the other. When I try to reduce them to squares maintaining the proportions, the smaller dimension becomes ridiculously small. Careful with that, cause I haven't thought about it deeply yet.

- Training images on a stupid MLP gives decent results. If to that dataset, we simply add the information on whether that image was resized or not, the test accuracy improves sizably. This probably means that resizing is often taking away useful information (which can be understood by reading the previous point).

- Con un grande class imbalance, non serve a niente usare grandi batch sizes, perché la rete assegnerà a ogni esempio la classe maggioritaria. Una idea intuitiva del motivo è che, se il 90% degli esempi è dinobryon, la label media del batch sarà dinobryon per tutti gli elementi del batch, inclusi quelli di classi diverse. **Corollario:** sta cosa è stata sicuramente pubblicata, ma se non lo è stata devo mettermici subito.

## Risorse

- Pretrained tricks (pretrained, bottleneck features, fine-tuning)

`https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html`

- Mixed data (dataset composto da features+immagini)

`https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/`

- Per class imbalance: ensemble cross validation, Class Weight/Importance, Over-Predict a Label than Under-Predict

`https://towardsdatascience.com/working-with-highly-imbalanced-datasets-in-machine-learning-projects-c70c5f2a7b16` (ma poi i suoi risultati fanno pena) 
`https://analyticsindiamag.com/5-important-techniques-to-process-imbalanced-data-in-machine-learning/`
