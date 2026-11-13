   .venv\Scripts\python -m src.features.extract_daic_audio --config configs/daic_models.yaml --mode turn_moments
   .venv\Scripts\python -m src.features.extract_daic_text --config configs/daic_models.yaml --variant participant
   .venv\Scripts\python -m src.features.extract_daic_text --config configs/daic_models.yaml --variant with_ellie
   Write-Host "ALL EXTRACTION DONE"