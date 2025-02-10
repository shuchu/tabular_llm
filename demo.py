from be_great import GReaT
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True).frame

model = GReaT(llm='distilgpt2', batch_size=32,  epochs=50, fp16=True)
model.fit(data)
synthetic_data = model.sample(n_samples=100)
