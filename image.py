from ultralytics import FastSAM
# from ultralytics.models.fastsam import FastSAMPrompt

src = "media/beach.jpg"
model = FastSAM("FastSAM-s.pt")
results: list = model.predict(source=src, show=True)
# reference https://docs.ultralytics.com/models/fast-sam/