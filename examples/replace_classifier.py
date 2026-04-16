"""Replace the ImageNet classifier heads for a new class count."""

from grlnet import GRLNetWeights, grlnet_stabhrec40

num_classes = 20

model = grlnet_stabhrec40(weights=GRLNetWeights.DEFAULT)
model.reset_classifier(num_classes=num_classes)

print(f"ready for transfer learning with {model.num_classes} classes")
