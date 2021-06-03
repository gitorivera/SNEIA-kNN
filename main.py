import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# introducimos la semila para hacer que lo aleatorio sea igual cada vez que se ejecute
np.random.seed(42)

# vamos a crear un funci'on que cree los datos aleatorios y las etiquetas
def generate_data(num_samples, num_features=2):
    data_size = (num_samples, num_features)
    data = np.random.randint(0,100,size=data_size)
    data = data.astype(np.float32)
    labels_size = (num_samples,1)
    labels = np.random.randint(0,2,size=labels_size)
    return data,labels

new_data, new_labels = generate_data(10,2)
class_1 = new_data[new_labels.ravel()==0]
class_2 = new_data[new_labels.ravel()==1]
plt.scatter(class_1[:,0], class_1[:,1], c='b', marker='s')
plt.scatter(class_2[:,0],class_2[:,1], c='r', marker='^')

# crearemos el clasificador
knn = cv2.ml.KNearest_create()
knn.train(new_data, cv2.ml.ROW_SAMPLE, new_labels)
# muestra aleatoria para evaluar
test_sample, _ = generate_data(1)
plt.scatter(test_sample[:,0], test_sample[:,1], c = 'g', marker ='o')
plt.show()

ret, results, neighbor, dist = knn.findNearest(test_sample,1)
print(results)
