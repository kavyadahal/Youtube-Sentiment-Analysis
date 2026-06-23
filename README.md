# YouTube Sentiment Analysis with DVC, MLflow, and DagsHub

This is the DagsHub version of the YouTube Sentiment Analysis MLOps project.
It does not require any cloud account or card.

## 1. Create and activate environment

```bash
conda create -n youtube_sentiment python=3.10 -y
conda activate youtube_sentiment
pip install -r requirements.txt
```

## 2. Connect to your DagsHub repo

This project is configured for:

```text
DagsHub owner: kavyadahal
DagsHub repo: Youtube-Sentiment-Analysis
```

The code uses:

```python
import dagshub

dagshub.init(repo_owner="kavyadahal", repo_name="Youtube-Sentiment-Analysis", mlflow=True)
```

When DagsHub asks you to log in, use your DagsHub account/token.
Do not put your token inside the code.

## 3. Test MLflow logging

```bash
python scripts/mlflow_test.py
```

Then open your DagsHub repository and check the **Experiments** tab.

## 4. Run the full DVC pipeline

```bash
dvc repro
```

This runs:

1. data ingestion
2. data preprocessing
3. model building
4. model evaluation and MLflow logging
5. model registration

## 5. Push DVC data/model files to DagsHub

The DVC remote URL is already set in `.dvc/config`.
Set your DagsHub token locally only:

```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user kavyadahal
dvc remote modify origin --local password YOUR_DAGSHUB_TOKEN
dvc push
```

## 6. Run the Flask API

```bash
python flask_app/app.py
```

Test endpoint:

```text
http://localhost:5000/predict
```

Example JSON body:

```json
{
  "comments": ["This video is awesome!", "Very bad explanation."]
}
```

## 7. Git workflow

```bash
git add .
git commit -m "Use DagsHub for MLflow and DVC"
git push origin main
```

## Notes

- No cloud account is needed.
- No card is needed.
- No cloud CLI setup is needed.
- Keep your DagsHub token private.
