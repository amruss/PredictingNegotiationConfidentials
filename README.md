### Context Model

To train run a context model run:

```
python train_context model_name --layers --batch_size --learning_rate --iterations

```

To load it after training run:

```
torch.load(FILENAME)
```

### Sentiment Model

```
import nltk.sentiment
nltk.download('vader_lexicon')
nltk.sentiment.util.demo_vader_instance("string")
```


### Logistic Regressor

Run LogisticRegressor.py or RewardPredictor.py


### Confidentials Predictor

To train the model run:

```
python train model_name --layers --batch_size --learning_rate --iterations

```