import torch

cls = torch.load("./trained_classifier.pth.tar")

print("\nnumber of trained epochs:\n-------------------------\n")
print(cls.current_epoch)

print("\nClassifier Setup:\n-------------------------\n")
print(cls.model)

print("\nAll Parameters:\n-------------------------\n")
for param in cls.model.parameters():
  print(param.data)
