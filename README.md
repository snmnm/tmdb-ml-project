## Movie Rating Predictor

Random Forest modeli kullanarak bir filmin IMDb benzeri puanını tahmin eden ML projesi.

### Pipeline
	•	EDA
	•	Baseline model
	•	Feature engineering
	•	Hyperparameter tuning
	•	Model evaluation
	•	Gradio ile mini uygulama

### Sonuçlar
	•	Baseline R²: 0.6235 → Optimized R²: 0.6283
	•	MAE: 0.5425 → 0.5359
	•	RMSE: 0.7535 → 0.7486

### Kullanılan Özellikler
	•	Log budget, log revenue, log popularity, log vote count
	•	Movie age
	•	Genre count
	•	Keyword count
	•	Ratio features (budget_per_minute, popularity_per_vote)

### App

Basit Gradio arayüzü ile tahmin alınabilir.
