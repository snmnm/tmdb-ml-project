## Live Demo  
Bu projenin çalışan Streamlit uygulamasına buradan ulaşabilirsiniz:

https://huggingface.co/spaces/sudesagmen/movie-rating-prediction


# Movie Rating Predictor

Filmlerin temel özelliklerini kullanarak IMDb benzeri puan tahmini yapan bir makine öğrenimi modeli. Proje; veri temizleme, özellik mühendisliği, model eğitimi ve değerlendirme aşamalarını içerir. Son aşamada Gradio ile basit bir tahmin arayüzü sunulmaktadır.

## Proje Açıklaması
Film bütçesi, hasılatı, süre, popülarite, oy sayısı ve metaveri bilgileri kullanılarak bir Random Forest regresyon modeli eğitilmiştir. Amaç, yapım aşamasındaki filmler için puan tahmini yapabilmektir.

## Pipeline
- EDA
- Baseline model
- Feature engineering
- Hyperparameter tuning
- Model evaluation
- Gradio arayüzü ile inference

## Sonuçlar
- R²: 0.6235 → 0.6283  
- MAE: 0.5425 → 0.5359  
- RMSE: 0.7535 → 0.7486  

## Kullanılan Teknolojiler
Python, Pandas, Scikit-Learn, Matplotlib, Joblib, Gradio

## Local Kurulum
```bash
pip install -r requirements.txt
python src/train.py
python app.py
```

## Repo Yapısı
```
src/
notebooks/
data/
models/
app.py
requirements.txt
README.md
```

## İletişim
sudesagmen@gmail.com
