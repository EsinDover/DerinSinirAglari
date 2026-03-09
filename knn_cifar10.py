import numpy as np
import pickle

print("CIFAR-10 veri seti yükleniyor...")

f = open("cifar-10-batches-py/data_batch_1", 'rb')
batch = pickle.load(f, encoding='bytes')
f.close()

data = np.array(batch[b'data'])
labels = np.array(batch[b'labels'])

print("Toplam veri sayısı:", len(data))

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

print("\nMesafe türünü seçin")
print("1 - L1 (Manhattan)")
print("2 - L2 (Euclidean)")

distance_type = input("Seçiminiz: ")

k = int(input("k değerini girin: "))

test_index = int(input("Test edilecek görüntü indexi (0-9999): "))

test_image = data[test_index]

print("\nk-NN hesaplanıyor...")

distances = []

i = 0
while i < len(data):
    if i != test_index:
        if distance_type == "1":
            distance = np.sum(np.abs(data[i] - test_image))
        else:
            distance = np.sqrt(np.sum((data[i] - test_image) ** 2))
        distances.append((distance, labels[i]))
    i += 1


distances.sort(key=lambda x: x[0])

neighbors = distances[:k]

votes = {}
for neighbor in neighbors:
    label = neighbor[1]
    if label in votes:
        votes[label] += 1
    else:
        votes[label] = 1

predicted_class = max(votes, key=votes.get)

print("\nSONUÇ")
print("Tahmin edilen sınıf:", classes[predicted_class])
print("Gerçek sınıf:", classes[labels[test_index]])